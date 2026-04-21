from pathlib import Path
import json

from repro.simulators.complex_simulator import ComplexSimulator
from repro.baselines.topsnp_lr import run_topsnp_lr
from repro.baselines.topsnp_lasso import run_topsnp_lasso
from repro.baselines.topsnp_xgb import run_topsnp_xgb
from repro.baselines.pca_lr import run_pca_lr
from repro.baselines.pca_lasso import run_pca_lasso
from repro.baselines.pca_xgb import run_pca_xgb


def serialize_scores(scores):
    return {str(k): float(v) for k, v in scores.items()}


def main():
    out_dir = Path("repro/outputs/baseline_smoke_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = ComplexSimulator(
        n_samples=5000,
        n_genes=10,
        snps_per_gene=20,
        causal_prop=0.5,
        snr=0.1,
        main_interaction_ratio=1.0,
        seed=42,
    )

    X, y, G = sim.generate()

    methods = {
        "topsnp_lr": run_topsnp_lr,
        "topsnp_lasso": run_topsnp_lasso,
        "topsnp_xgb": run_topsnp_xgb,
        "pca_lr": run_pca_lr,
        "pca_lasso": run_pca_lasso,
        "pca_xgb": run_pca_xgb,
    }

    results = {}
    for name, fn in methods.items():
        print(f"Running {name}...")
        result = fn(X, y, n_genes=10, snps_per_gene=20)
        results[name] = serialize_scores(result["scores"])

    with open(out_dir / "baseline_scores.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()