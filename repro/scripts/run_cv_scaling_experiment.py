from __future__ import annotations

import gc
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from repro.eval.metrics import compute_metrics
from repro.interactions.shapley_gene import compute_all_nn_interaction_scores
from repro.models.paper_models import GeneInteractionNN, l1_penalty
from repro.simulators.complex_simulator import ComplexSimulator
from repro.training.nn_dataset import GeneDataset, collate_gene_batch
from repro.training.train_ensemble import (
    compute_ensemble_nn_scores,
    train_nn_ensemble,
)
from repro.training.train_nn import make_gene_loaders, train_model
from repro.utils.profiling import profile_callable


@dataclass
class ExperimentConfig:
    # Start small, then replace with [40000, 80000, 120000].
    sample_sizes: tuple[int, ...] = (5000, 10000, 20000, 40000)

    n_genes: int = 10
    snps_per_gene: int = 20

    causal_prop: float = 0.5
    snr: float = 0.1
    main_interaction_ratio: float = 1.0

    # CV settings.
    n_folds: int = 3
    l1_candidates: tuple[float, ...] = (
        1e-4,
        1e-5,
        1e-6,
    )

    # CV should remain light because every candidate is trained per fold.
    cv_num_epochs: int = 80
    cv_patience: int = 10

    # Final ensemble settings.
    ensemble_size: int = 5
    final_num_epochs: int = 150
    final_patience: int = 10

    gene_hidden_dim: int = 10
    predictor_hidden_dim: int = 100
    learning_rate: float = 0.005

    batch_size: int = 4096
    num_subset_samples: int = 100

    seed: int = 42


TRUE_INTERACTION_PAIRS = {
    (0, 1),
    (6, 8),
    (2, 4),
    (3, 7),
    (5, 6),
    (7, 9),
}


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_snps_by_gene(
    X: np.ndarray,
    n_genes: int,
    snps_per_gene: int,
) -> list[np.ndarray]:
    """
    Split a flat SNP matrix into one matrix per gene.

    X:
        shape (n_samples, n_genes * snps_per_gene)

    Output:
        list of n_genes arrays,
        each of shape (n_samples, snps_per_gene)
    """

    expected_features = n_genes * snps_per_gene

    if X.shape[1] != expected_features:
        raise ValueError(
            f"Expected {expected_features} SNP columns, "
            f"but received {X.shape[1]}."
        )

    gene_blocks = []

    for gene_index in range(n_genes):
        start = gene_index * snps_per_gene
        end = start + snps_per_gene
        gene_blocks.append(X[:, start:end])

    return gene_blocks


def make_loader_from_indices(
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    n_genes: int,
    snps_per_gene: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """
    Create a DataLoader from selected sample indices.
    """

    X_subset = X[indices]
    y_subset = y[indices]

    gene_blocks = split_snps_by_gene(
        X_subset,
        n_genes=n_genes,
        snps_per_gene=snps_per_gene,
    )

    dataset = GeneDataset(gene_blocks, y_subset)

    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        collate_fn=collate_gene_batch,
    )


@torch.no_grad()
def calculate_loader_mse(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> float:
    """
    Calculate ordinary prediction MSE on a validation fold.

    The L1 penalty is not included in this validation metric because
    CV should select the model that best predicts held-out samples.
    """

    model.eval()
    model.to(device)

    criterion = nn.MSELoss(reduction="sum")
    total_squared_error = 0.0
    total_samples = 0

    for gene_inputs, y_batch in loader:
        gene_inputs = [
            tensor.to(device)
            for tensor in gene_inputs
        ]
        y_batch = y_batch.to(device)

        prediction = model(gene_inputs)

        total_squared_error += criterion(
            prediction,
            y_batch,
        ).item()

        total_samples += y_batch.shape[0]

    if total_samples == 0:
        raise RuntimeError("Validation DataLoader is empty.")

    return float(total_squared_error / total_samples)


def cross_validate_l1(
    X: np.ndarray,
    y: np.ndarray,
    config: ExperimentConfig,
    device: str,
) -> tuple[float, dict[str, Any]]:
    """
    Select the L1 coefficient using K-fold cross-validation.

    For every candidate lambda:
      1. split samples into K folds
      2. train on K-1 folds
      3. evaluate MSE on the held-out fold
      4. average validation MSE across folds

    The candidate with the lowest average validation MSE is selected.
    """

    kfold = KFold(
        n_splits=config.n_folds,
        shuffle=True,
        random_state=config.seed,
    )

    cv_results: dict[str, Any] = {}

    for l1_candidate in config.l1_candidates:
        print("\n" + "=" * 80)
        print(f"Testing L1 lambda = {l1_candidate}")
        print("=" * 80)

        fold_losses: list[float] = []

        for fold_index, (train_indices, val_indices) in enumerate(
            kfold.split(X),
            start=1,
        ):
            fold_seed = (
                config.seed
                + fold_index
                + int(abs(np.log10(l1_candidate)) * 100)
            )
            set_all_seeds(fold_seed)

            print(
                f"\nL1={l1_candidate} "
                f"| fold {fold_index}/{config.n_folds}"
            )

            train_loader = make_loader_from_indices(
                X=X,
                y=y,
                indices=train_indices,
                n_genes=config.n_genes,
                snps_per_gene=config.snps_per_gene,
                batch_size=config.batch_size,
                shuffle=True,
            )

            val_loader = make_loader_from_indices(
                X=X,
                y=y,
                indices=val_indices,
                n_genes=config.n_genes,
                snps_per_gene=config.snps_per_gene,
                batch_size=config.batch_size,
                shuffle=False,
            )

            model = GeneInteractionNN(
                gene_input_dims=[
                    config.snps_per_gene
                ] * config.n_genes,
                gene_hidden_dim=config.gene_hidden_dim,
                predictor_hidden_dim=config.predictor_hidden_dim,
            )

            model, _ = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                l1_penalty_fn=l1_penalty,
                lr=config.learning_rate,
                l1_lambda=l1_candidate,
                num_epochs=config.cv_num_epochs,
                patience=config.cv_patience,
                device=device,
            )

            fold_mse = calculate_loader_mse(
                model=model,
                loader=val_loader,
                device=device,
            )

            print(
                f"Validation MSE for fold {fold_index}: "
                f"{fold_mse:.6f}"
            )

            fold_losses.append(fold_mse)

            del model
            del train_loader
            del val_loader
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mean_mse = float(np.mean(fold_losses))
        std_mse = float(np.std(fold_losses, ddof=1))

        cv_results[str(l1_candidate)] = {
            "fold_validation_mse": fold_losses,
            "mean_validation_mse": mean_mse,
            "std_validation_mse": std_mse,
        }

        print(
            f"\nL1={l1_candidate}: "
            f"mean validation MSE={mean_mse:.6f}, "
            f"std={std_mse:.6f}"
        )

    best_l1 = min(
        config.l1_candidates,
        key=lambda candidate: cv_results[str(candidate)][
            "mean_validation_mse"
        ],
    )

    print("\n" + "#" * 80)
    print(f"Selected best L1 lambda: {best_l1}")
    print("#" * 80)

    return float(best_l1), cv_results


def run_one_dataset_size(
    n_samples: int,
    config: ExperimentConfig,
    device: str,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Run the complete workflow for one dataset size:

    1. generate synthetic data
    2. run 3-fold CV for L1
    3. train final ensemble
    4. compute ensemble interaction scores
    5. evaluate AUROC and AP
    6. save time and memory measurements
    """

    print("\n" + "=" * 100)
    print(f"RUNNING DATASET SIZE: {n_samples}")
    print("=" * 100)

    set_all_seeds(config.seed)

    simulator = ComplexSimulator(
        n_samples=n_samples,
        n_genes=config.n_genes,
        snps_per_gene=config.snps_per_gene,
        causal_prop=config.causal_prop,
        snr=config.snr,
        main_interaction_ratio=config.main_interaction_ratio,
        seed=config.seed,
    )

    generated_data, generation_resources = profile_callable(
        simulator.generate,
        profiling_device="cpu",
    )

    X, y, _ = generated_data

    print(
        f"Generated X={X.shape}, y={y.shape} "
        f"in {generation_resources.wall_time_seconds:.2f}s"
    )

    cv_output, cv_resources = profile_callable(
        cross_validate_l1,
        X=X,
        y=y,
        config=config,
        device=device,
        profiling_device=device,
    )

    best_l1, cv_results = cv_output

    train_loader, val_loader = make_gene_loaders(
        X=X,
        y=y,
        n_genes=config.n_genes,
        snps_per_gene=config.snps_per_gene,
        batch_size=config.batch_size,
        val_fraction=0.2,
        seed=config.seed,
    )

    ensemble_output, training_resources = profile_callable(
        train_nn_ensemble,
        train_loader=train_loader,
        val_loader=val_loader,
        gene_input_dims=[
            config.snps_per_gene
        ] * config.n_genes,
        ensemble_size=config.ensemble_size,
        gene_hidden_dim=config.gene_hidden_dim,
        predictor_hidden_dim=config.predictor_hidden_dim,
        lr=config.learning_rate,
        l1_lambda=best_l1,
        num_epochs=config.final_num_epochs,
        patience=config.final_patience,
        device=device,
        base_seed=config.seed,
        profiling_device=device,
    )

    models, histories = ensemble_output

    interaction_scores, scoring_resources = profile_callable(
        compute_ensemble_nn_scores,
        models=models,
        X=X,
        n_genes=config.n_genes,
        snps_per_gene=config.snps_per_gene,
        num_subset_samples=config.num_subset_samples,
        device=device,
        base_seed=config.seed,
        profiling_device=device,
    )

    evaluation_metrics = compute_metrics(
        interaction_scores,
        TRUE_INTERACTION_PAIRS,
    )

    sorted_pairs = sorted(
        interaction_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top_10_pairs = [
        {
            "pair": str(pair),
            "score": float(score),
            "is_true": pair in TRUE_INTERACTION_PAIRS,
        }
        for pair, score in sorted_pairs[:10]
    ]

    result = {
        "n_samples": n_samples,
        "n_genes": config.n_genes,
        "snps_per_gene": config.snps_per_gene,
        "snr": config.snr,
        "causal_prop": config.causal_prop,
        "main_interaction_ratio": (
            config.main_interaction_ratio
        ),
        "ensemble_size": config.ensemble_size,
        "num_subset_samples": config.num_subset_samples,
        "selected_l1_lambda": best_l1,
        "auroc": evaluation_metrics["auroc"],
        "ap": evaluation_metrics["ap"],
        "generation_seconds": (
            generation_resources.wall_time_seconds
        ),
        "cv_seconds": cv_resources.wall_time_seconds,
        "training_seconds": (
            training_resources.wall_time_seconds
        ),
        "interaction_scoring_seconds": (
            scoring_resources.wall_time_seconds
        ),
        "total_seconds": (
            generation_resources.wall_time_seconds
            + cv_resources.wall_time_seconds
            + training_resources.wall_time_seconds
            + scoring_resources.wall_time_seconds
        ),
        "generation_peak_ram_mb": (
            generation_resources.peak_ram_mb
        ),
        "cv_peak_ram_mb": cv_resources.peak_ram_mb,
        "training_peak_ram_mb": (
            training_resources.peak_ram_mb
        ),
        "scoring_peak_ram_mb": (
            scoring_resources.peak_ram_mb
        ),
        "training_peak_gpu_allocated_mb": (
            training_resources.peak_gpu_allocated_mb
        ),
        "training_peak_gpu_reserved_mb": (
            training_resources.peak_gpu_reserved_mb
        ),
        "scoring_peak_gpu_allocated_mb": (
            scoring_resources.peak_gpu_allocated_mb
        ),
        "cv_results": cv_results,
        "top_10_pairs": top_10_pairs,
        "interaction_scores": {
            str(pair): float(score)
            for pair, score in interaction_scores.items()
        },
    }

    run_output_file = (
        output_dir
        / f"result_n{n_samples}.json"
    )

    with open(
        run_output_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(result, file, indent=2)

    print("\nFinal result:")
    print(
        f"n={n_samples}, "
        f"best_l1={best_l1}, "
        f"AUROC={evaluation_metrics['auroc']:.4f}, "
        f"AP={evaluation_metrics['ap']:.4f}"
    )

    del models
    del histories
    del train_loader
    del val_loader
    del X
    del y

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main() -> None:
    config = ExperimentConfig()

    output_dir = Path(
        "repro/outputs/cv_scaling_experiment"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)
    print("Experiment configuration:")
    print(json.dumps(asdict(config), indent=2))

    all_results: list[dict[str, Any]] = []

    for n_samples in config.sample_sizes:
        result = run_one_dataset_size(
            n_samples=n_samples,
            config=config,
            device=device,
            output_dir=output_dir,
        )
        all_results.append(result)

        summary_rows = []

        for run_result in all_results:
            summary_rows.append(
                {
                    key: value
                    for key, value in run_result.items()
                    if key not in {
                        "cv_results",
                        "top_10_pairs",
                        "interaction_scores",
                    }
                }
            )

        summary_dataframe = pd.DataFrame(
            summary_rows
        )

        summary_dataframe.to_csv(
            output_dir / "scaling_summary.csv",
            index=False,
        )

        with open(
            output_dir / "all_results.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                all_results,
                file,
                indent=2,
            )

    print("\nExperiment complete.")
    print(
        "Saved results to:",
        output_dir.resolve(),
    )


if __name__ == "__main__":
    main()