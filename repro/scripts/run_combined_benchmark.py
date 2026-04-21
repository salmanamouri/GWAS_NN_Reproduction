from pathlib import Path
import json

from repro.simulators.complex_simulator import ComplexSimulator

from repro.baselines.topsnp_lr import run_topsnp_lr
from repro.baselines.topsnp_lasso import run_topsnp_lasso
from repro.baselines.topsnp_xgb import run_topsnp_xgb
from repro.baselines.pca_lr import run_pca_lr
from repro.baselines.pca_lasso import run_pca_lasso
from repro.baselines.pca_xgb import run_pca_xgb

from repro.models.paper_models import GeneInteractionNN, l1_penalty
from repro.training.train_nn import make_gene_loaders, train_model, extract_gene_layer
from repro.interactions.shapley_gene import compute_all_nn_interaction_scores
from repro.eval.metrics import compute_metrics

import torch

"""
    This is the first full comparison runner:
        same dataset
        same ground truth
        same evaluation metrics
        all methods in one place

    That makes the results directly comparable.
"""

def main():
    out_dir = Path("repro/outputs/combined_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    # same debug setting for fair comparison
    sim = ComplexSimulator(
        n_samples=5000,
        n_genes=10,
        snps_per_gene=20,
        causal_prop=0.5,
        snr=0.1,
        main_interaction_ratio=1.0,
        seed=42,
    )

    X, y, G_true = sim.generate()
    true_pairs = {(0, 1), (6, 8), (2, 4), (3, 7), (5, 6), (7, 9)}

    results = {}

    # -------- Baselines --------
    baseline_methods = {
        "topsnp_lr": run_topsnp_lr,
        "topsnp_lasso": run_topsnp_lasso,
        "topsnp_xgb": run_topsnp_xgb,
        "pca_lr": run_pca_lr,
        "pca_lasso": run_pca_lasso,
        "pca_xgb": run_pca_xgb,
    }

    for name, fn in baseline_methods.items():
        print(f"Running baseline: {name}")
        result = fn(X, y, n_genes=10, snps_per_gene=20)
        metrics = compute_metrics(result["scores"], true_pairs)

        results[name] = {
            "type": "baseline",
            "metrics": metrics,
            "scores": {str(k): float(v) for k, v in result["scores"].items()},
        }

        print(f"{name}: {metrics}")

    # -------- Neural Network --------
    print("Running neural method...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = make_gene_loaders(
        X=X,
        y=y,
        n_genes=10,
        snps_per_gene=20,
        batch_size=1024,
        val_fraction=0.2,
        seed=42,
    )

    nn_model = GeneInteractionNN(
        gene_input_dims=[20] * 10,
        gene_hidden_dim=10,
        predictor_hidden_dim=100,
    )

    nn_model, history = train_model(
        nn_model,
        train_loader,
        val_loader,
        l1_penalty_fn=l1_penalty,
        lr=0.005,
        l1_lambda=1e-5,
        num_epochs=300,
        patience=20,
        device=device,
    )

    gene_layer = extract_gene_layer(
        nn_model,
        X,
        n_genes=10,
        snps_per_gene=20,
        device=device,
    )

    nn_scores = compute_all_nn_interaction_scores(
        model=nn_model,
        gene_layer=gene_layer,
        num_subset_samples=50,
        device=device,
        seed=42,
    )

    nn_metrics = compute_metrics(nn_scores, true_pairs)

    results["nn_shapley"] = {
        "type": "neural",
        "metrics": nn_metrics,
        "gene_layer_shape": list(gene_layer.shape),
        "scores": {str(k): float(v) for k, v in nn_scores.items()},
    }

    print(f"nn_shapley: {nn_metrics}")

    with open(out_dir / "combined_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()