from pathlib import Path
import json
import torch

from repro.simulators.simple_simulator import SimpleSimulator
from repro.baselines.topsnp_lr import run_topsnp_lr
from repro.baselines.topsnp_lasso import run_topsnp_lasso
from repro.baselines.topsnp_xgb import run_topsnp_xgb
from repro.baselines.pca_lr import run_pca_lr
from repro.baselines.pca_lasso import run_pca_lasso
from repro.baselines.pca_xgb import run_pca_xgb
from repro.eval.metrics import compute_metrics

from repro.training.train_nn import make_gene_loaders
from repro.training.train_ensemble import train_nn_ensemble, compute_ensemble_nn_scores


def main():
    out_dir = Path("repro/outputs/simple_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = SimpleSimulator(
        n_samples=40000,
        n_genes=10,
        snps_per_gene=20,
        causal_prop=0.5,
        snr=0.1,
        main_interaction_ratio=1.0,
        seed=42,
    )

    X, y, metadata = sim.generate()
    true_pairs = set(tuple(p) for p in metadata["true_pairs"])

    results = {}

    # baselines
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
        }
        print(f"{name}: {metrics}")

    # neural ensemble
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running neural method...")

    train_loader, val_loader = make_gene_loaders(
        X=X,
        y=y,
        n_genes=10,
        snps_per_gene=20,
        batch_size=4096,
        val_fraction=0.2,
        seed=42,
    )

    models, histories = train_nn_ensemble(
        train_loader=train_loader,
        val_loader=val_loader,
        gene_input_dims=[20] * 10,
        ensemble_size=10, #20
        gene_hidden_dim=10,
        predictor_hidden_dim=100,
        lr=0.005,
        l1_lambda=1e-5,
        num_epochs=150, #300
        patience=10, #20
        device=device,
        base_seed=42,
    )

    nn_scores = compute_ensemble_nn_scores(
        models=models,
        X=X,
        n_genes=10,
        snps_per_gene=20,
        num_subset_samples=500,
        device=device,
        base_seed=42,
    )

    nn_metrics = compute_metrics(nn_scores, true_pairs)
    results["nn_shapley"] = {
        "type": "neural",
        "metrics": nn_metrics,
    }
    print(f"nn_shapley: {nn_metrics}")

    with open(out_dir / "simple_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()