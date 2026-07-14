from __future__ import annotations

import gc
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from repro.interactions.shapley_gene import compute_all_nn_interaction_scores
from repro.models.paper_models import (
    GeneInteractionNN,
    MainEffectNN,
    l1_penalty,
)
from repro.permutation.perm_i import build_perm_i_target
from repro.simulators.complex_simulator import ComplexSimulator
from repro.training.train_ensemble import (
    compute_ensemble_nn_scores,
    train_nn_ensemble,
)
from repro.training.train_nn import (
    make_gene_loaders,
    predict_model,
    train_model,
)
from repro.utils.profiling import profile_callable


# ---------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------

# Start with 5000. Later use 40000, 80000, or 120000.
N_SAMPLES = 40000

N_GENES = 10
SNPS_PER_GENE = 20

CAUSAL_PROP = 0.5
SNR = 0.1
MAIN_INTERACTION_RATIO = 1.0

DATASET_SEED = 42

# Number of independently shuffled Perm I null targets.
# Use 5 for debugging, then 20, 50, 100, or more.
N_PERMUTATIONS = 100

# Observed and null interaction ensembles.
# Use 2 for debugging. Increase later.
OBSERVED_ENSEMBLE_SIZE = 5
NULL_ENSEMBLE_SIZE = 1 #Why use null ensemble size 1 instead of 2? 100 permutations are already expensive & one null model per permutation gives 100 independent null scores

# Shapley Monte Carlo subset samples.
# Use 20 for debugging. Increase later.
NUM_SUBSET_SAMPLES = 50

BATCH_SIZE = 2048
LEARNING_RATE = 0.005

NUM_EPOCHS = 100
PATIENCE = 10

GENE_HIDDEN_DIM = 10
PREDICTOR_HIDDEN_DIM = 100

VAL_FRACTION = 0.2

# Result produced by run_cv_scaling_experiment.py.
CV_RESULT_FILE = Path(
    f"repro/outputs/cv_scaling_experiment/result_n{N_SAMPLES}.json"
)

OUTPUT_DIR = Path(
    f"repro/outputs/perm_i_pvalues_n{N_SAMPLES}"
)


TRUE_INTERACTION_PAIRS = {
    (0, 1),
    (6, 8),
    (2, 4),
    (3, 7),
    (5, 6),
    (7, 9),
}


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
    """
    Set Python, NumPy, and PyTorch random seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# CV result loading
# ---------------------------------------------------------------------

def load_selected_l1(result_file: Path) -> float:
    """
    Load the L1 coefficient selected by cross-validation.

    The file is created by:
        repro/scripts/run_cv_scaling_experiment.py
    """
    if not result_file.exists():
        raise FileNotFoundError(
            f"CV result file was not found:\n{result_file}\n\n"
            "Run the CV scaling script for this sample size first."
        )

    with open(result_file, "r", encoding="utf-8") as file:
        result = json.load(file)

    if "selected_l1_lambda" not in result:
        raise KeyError(
            "The CV result JSON does not contain "
            "'selected_l1_lambda'."
        )

    selected_l1 = float(result["selected_l1_lambda"])

    print(f"Loaded CV-selected L1 lambda: {selected_l1}")

    return selected_l1


# ---------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------

def generate_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Regenerate the same complex synthetic dataset used for CV.
    """
    simulator = ComplexSimulator(
        n_samples=N_SAMPLES,
        n_genes=N_GENES,
        snps_per_gene=SNPS_PER_GENE,
        causal_prop=CAUSAL_PROP,
        snr=SNR,
        main_interaction_ratio=MAIN_INTERACTION_RATIO,
        seed=DATASET_SEED,
    )

    X, y, true_gene_latents = simulator.generate()

    print("Generated dataset:")
    print("X:", X.shape)
    print("y:", y.shape)
    print("True latent gene matrix:", true_gene_latents.shape)

    return X, y, true_gene_latents


# ---------------------------------------------------------------------
# Main-effect NN
# ---------------------------------------------------------------------

def train_main_effect_model(
    X: np.ndarray,
    y: np.ndarray,
    selected_l1: float,
    device: str,
) -> tuple[MainEffectNN, np.ndarray, dict[str, list[float]]]:
    """
    Train the main-effect-only neural network.

    It estimates:
        y_hat_main

    Perm I then computes:
        residual = y - y_hat_main
        y_null = y_hat_main + permuted(residual)
    """
    train_loader, val_loader = make_gene_loaders(
        X=X,
        y=y,
        n_genes=N_GENES,
        snps_per_gene=SNPS_PER_GENE,
        batch_size=BATCH_SIZE,
        val_fraction=VAL_FRACTION,
        seed=DATASET_SEED,
    )

    model = MainEffectNN(
        gene_input_dims=[SNPS_PER_GENE] * N_GENES,
        gene_hidden_dim=GENE_HIDDEN_DIM,
    )

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        l1_penalty_fn=l1_penalty,
        lr=LEARNING_RATE,
        l1_lambda=selected_l1,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        device=device,
    )

    y_hat_main = predict_model(
        model=model,
        X=X,
        n_genes=N_GENES,
        snps_per_gene=SNPS_PER_GENE,
        device=device,
    )

    return model, y_hat_main, history


# ---------------------------------------------------------------------
# Interaction ensemble
# ---------------------------------------------------------------------

def train_interaction_ensemble_and_score(
    X: np.ndarray,
    y: np.ndarray,
    selected_l1: float,
    ensemble_size: int,
    base_seed: int,
    device: str,
) -> tuple[
    dict[tuple[int, int], float],
    list[dict[str, list[float]]],
]:
    """
    Train an interaction NN ensemble and compute averaged interaction scores.

    Each ensemble member:
      1. learns its own gene representations;
      2. computes its own Shapley-style pair scores.

    Final scores are averaged across ensemble members.
    """
    train_loader, val_loader = make_gene_loaders(
        X=X,
        y=y,
        n_genes=N_GENES,
        snps_per_gene=SNPS_PER_GENE,
        batch_size=BATCH_SIZE,
        val_fraction=VAL_FRACTION,
        seed=DATASET_SEED,
    )

    models, histories = train_nn_ensemble(
        train_loader=train_loader,
        val_loader=val_loader,
        gene_input_dims=[SNPS_PER_GENE] * N_GENES,
        ensemble_size=ensemble_size,
        gene_hidden_dim=GENE_HIDDEN_DIM,
        predictor_hidden_dim=PREDICTOR_HIDDEN_DIM,
        lr=LEARNING_RATE,
        l1_lambda=selected_l1,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        device=device,
        base_seed=base_seed,
    )

    scores = compute_ensemble_nn_scores(
        models=models,
        X=X,
        n_genes=N_GENES,
        snps_per_gene=SNPS_PER_GENE,
        num_subset_samples=NUM_SUBSET_SAMPLES,
        device=device,
        base_seed=base_seed,
    )

    del models
    del train_loader
    del val_loader

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return scores, histories


# ---------------------------------------------------------------------
# Empirical p-values
# ---------------------------------------------------------------------

def calculate_empirical_pvalues(
    observed_scores: dict[tuple[int, int], float],
    null_scores: dict[tuple[int, int], list[float]],
) -> dict[tuple[int, int], float]:
    """
    Compute one-sided empirical p-values with the +1 correction.

    p_ij =
        (1 + number of null scores >= observed score)
        / (1 + number of permutations)

    The +1 correction prevents a p-value of exactly zero.
    """
    pvalues: dict[tuple[int, int], float] = {}

    for pair, observed_score in observed_scores.items():
        pair_null_scores = np.asarray(
            null_scores[pair],
            dtype=float,
        )

        count_at_least_as_large = int(
            np.sum(pair_null_scores >= observed_score)
        )

        pvalue = (
            1 + count_at_least_as_large
        ) / (
            1 + len(pair_null_scores)
        )

        pvalues[pair] = float(pvalue)

    return pvalues


def benjamini_hochberg(
    pvalues: dict[tuple[int, int], float],
) -> dict[tuple[int, int], float]:
    """
    Benjamini-Hochberg false-discovery-rate correction.

    Multiple gene pairs are tested simultaneously, so raw p-values
    should not be interpreted independently.
    """
    pairs = list(pvalues.keys())
    raw_values = np.asarray(
        [pvalues[pair] for pair in pairs],
        dtype=float,
    )

    order = np.argsort(raw_values)
    sorted_pvalues = raw_values[order]

    number_of_tests = len(sorted_pvalues)

    adjusted_sorted = np.empty(
        number_of_tests,
        dtype=float,
    )

    running_minimum = 1.0

    for reverse_index in range(
        number_of_tests - 1,
        -1,
        -1,
    ):
        rank = reverse_index + 1

        adjusted_value = (
            sorted_pvalues[reverse_index]
            * number_of_tests
            / rank
        )

        running_minimum = min(
            running_minimum,
            adjusted_value,
        )

        adjusted_sorted[reverse_index] = min(
            running_minimum,
            1.0,
        )

    adjusted_original_order = np.empty(
        number_of_tests,
        dtype=float,
    )
    adjusted_original_order[order] = adjusted_sorted

    return {
        pair: float(adjusted_original_order[index])
        for index, pair in enumerate(pairs)
    }


# ---------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------

def serialize_pair_scores(
    values: dict[tuple[int, int], Any],
) -> dict[str, Any]:
    """
    Convert tuple dictionary keys to strings for JSON serialization.
    """
    serialized = {}

    for pair, value in values.items():
        if isinstance(value, list):
            serialized[str(pair)] = [
                float(item)
                for item in value
            ]
        else:
            serialized[str(pair)] = float(value)

    return serialized


def build_pair_table(
    observed_scores: dict[tuple[int, int], float],
    null_scores: dict[tuple[int, int], list[float]],
    raw_pvalues: dict[tuple[int, int], float],
    adjusted_pvalues: dict[tuple[int, int], float],
) -> pd.DataFrame:
    """
    Create a CSV-friendly table with one row per gene pair.
    """
    rows = []

    for pair in sorted(observed_scores):
        pair_null = np.asarray(
            null_scores[pair],
            dtype=float,
        )

        rows.append(
            {
                "gene_i": pair[0],
                "gene_j": pair[1],
                "pair": str(pair),
                "is_true_interaction": (
                    pair in TRUE_INTERACTION_PAIRS
                ),
                "observed_score": observed_scores[pair],
                "null_mean": float(np.mean(pair_null)),
                "null_std": float(np.std(pair_null)),
                "null_max": float(np.max(pair_null)),
                "raw_pvalue": raw_pvalues[pair],
                "bh_adjusted_pvalue": adjusted_pvalues[pair],
                "significant_raw_0.05": (
                    raw_pvalues[pair] <= 0.05
                ),
                "significant_bh_0.05": (
                    adjusted_pvalues[pair] <= 0.05
                ),
            }
        )

    dataframe = pd.DataFrame(rows)

    return dataframe.sort_values(
        by=[
            "bh_adjusted_pvalue",
            "raw_pvalue",
            "observed_score",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    set_all_seeds(DATASET_SEED)

    selected_l1 = load_selected_l1(
        CV_RESULT_FILE
    )

    generated_output, generation_resources = profile_callable(
        generate_dataset,
        profiling_device="cpu",
    )

    X, y, _ = generated_output

    print("\nTraining observed interaction ensemble...")

    observed_output, observed_resources = profile_callable(
        train_interaction_ensemble_and_score,
        X=X,
        y=y,
        selected_l1=selected_l1,
        ensemble_size=OBSERVED_ENSEMBLE_SIZE,
        base_seed=DATASET_SEED,
        device=device,
        profiling_device=device,
    )

    observed_scores, observed_histories = observed_output

    print("\nTraining main-effect model...")

    main_effect_output, main_effect_resources = profile_callable(
        train_main_effect_model,
        X=X,
        y=y,
        selected_l1=selected_l1,
        device=device,
        profiling_device=device,
    )

    main_effect_model, y_hat_main, main_history = (
        main_effect_output
    )

    residuals = y - y_hat_main

    print("\nResidual summary:")
    print("Mean:", float(np.mean(residuals)))
    print("Std:", float(np.std(residuals)))

    null_scores: dict[
        tuple[int, int],
        list[float],
    ] = {
        pair: []
        for pair in observed_scores
    }

    permutation_resource_rows = []

    for permutation_index in range(N_PERMUTATIONS):
        permutation_number = permutation_index + 1
        permutation_seed = (
            1000
            + DATASET_SEED
            + permutation_index
        )

        print("\n" + "=" * 90)
        print(
            f"Perm I null run "
            f"{permutation_number}/{N_PERMUTATIONS}"
        )
        print("=" * 90)

        y_null = build_perm_i_target(
            y_true=y,
            y_hat_main=y_hat_main,
            seed=permutation_seed,
        )

        null_output, null_resources = profile_callable(
            train_interaction_ensemble_and_score,
            X=X,
            y=y_null,
            selected_l1=selected_l1,
            ensemble_size=NULL_ENSEMBLE_SIZE,
            base_seed=permutation_seed,
            device=device,
            profiling_device=device,
        )

        permutation_scores, _ = null_output

        for pair, score in permutation_scores.items():
            null_scores[pair].append(
                float(score)
            )

        permutation_resource_rows.append(
            {
                "permutation": permutation_number,
                "seed": permutation_seed,
                **null_resources.to_dict(),
            }
        )

        del y_null
        del permutation_scores

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raw_pvalues = calculate_empirical_pvalues(
        observed_scores=observed_scores,
        null_scores=null_scores,
    )

    adjusted_pvalues = benjamini_hochberg(
        raw_pvalues
    )

    pair_table = build_pair_table(
        observed_scores=observed_scores,
        null_scores=null_scores,
        raw_pvalues=raw_pvalues,
        adjusted_pvalues=adjusted_pvalues,
    )

    pair_table.to_csv(
        OUTPUT_DIR / "pair_pvalues.csv",
        index=False,
    )

    pd.DataFrame(
        permutation_resource_rows
    ).to_csv(
        OUTPUT_DIR
        / "permutation_resource_usage.csv",
        index=False,
    )

    significant_raw = pair_table[
        pair_table["significant_raw_0.05"]
    ]

    significant_adjusted = pair_table[
        pair_table["significant_bh_0.05"]
    ]

    summary = {
        "configuration": {
            "n_samples": N_SAMPLES,
            "n_genes": N_GENES,
            "snps_per_gene": SNPS_PER_GENE,
            "causal_prop": CAUSAL_PROP,
            "snr": SNR,
            "main_interaction_ratio": (
                MAIN_INTERACTION_RATIO
            ),
            "dataset_seed": DATASET_SEED,
            "selected_l1_lambda": selected_l1,
            "n_permutations": N_PERMUTATIONS,
            "observed_ensemble_size": (
                OBSERVED_ENSEMBLE_SIZE
            ),
            "null_ensemble_size": (
                NULL_ENSEMBLE_SIZE
            ),
            "num_subset_samples": (
                NUM_SUBSET_SAMPLES
            ),
            "num_epochs": NUM_EPOCHS,
            "patience": PATIENCE,
            "device": device,
        },
        "resource_usage": {
            "generation": (
                generation_resources.to_dict()
            ),
            "observed_interaction_ensemble": (
                observed_resources.to_dict()
            ),
            "main_effect_model": (
                main_effect_resources.to_dict()
            ),
        },
        "number_significant_raw_0.05": int(
            len(significant_raw)
        ),
        "number_significant_bh_0.05": int(
            len(significant_adjusted)
        ),
        "observed_scores": serialize_pair_scores(
            observed_scores
        ),
        "null_scores": serialize_pair_scores(
            null_scores
        ),
        "raw_pvalues": serialize_pair_scores(
            raw_pvalues
        ),
        "bh_adjusted_pvalues": serialize_pair_scores(
            adjusted_pvalues
        ),
    }

    with open(
        OUTPUT_DIR / "perm_i_pvalue_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\n" + "#" * 90)
    print("PERM I P-VALUE EXPERIMENT COMPLETE")
    print("#" * 90)

    print(
        "Raw significant pairs at alpha=0.05:",
        len(significant_raw),
    )

    print(
        "BH-adjusted significant pairs at FDR=0.05:",
        len(significant_adjusted),
    )

    print("\nTop pairs by adjusted p-value:")

    columns_to_show = [
        "pair",
        "is_true_interaction",
        "observed_score",
        "null_mean",
        "raw_pvalue",
        "bh_adjusted_pvalue",
    ]

    print(
        pair_table[
            columns_to_show
        ].head(10).to_string(index=False)
    )

    print(
        "\nSaved outputs to:",
        OUTPUT_DIR.resolve(),
    )


if __name__ == "__main__":
    main()