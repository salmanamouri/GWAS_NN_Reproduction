import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

from repro.simulators.complex_simulator import ComplexSimulator
from repro.permutation.perm_i import build_perm_i_target, empirical_p_value
from repro.permutation.perm_t import build_perm_t_target
from repro.permutation.perm_r import build_perm_r_target
from repro.eval.power_specificity import classify_significant_pairs, compute_tpr_tnr


def simple_pairwise_scores(G: np.ndarray, y: np.ndarray):
    """
    Proxy interaction score for the benchmark scaffold.
    Uses the absolute correlation between y and Gi*Gj.
    """
    n_genes = G.shape[1]
    scores = {}

    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            interaction_term = G[:, i] * G[:, j]
            corr = np.corrcoef(interaction_term, y)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            scores[(i, j)] = abs(float(corr))

    return scores


"""Perm I uses predicted main effects and Perm R uses residuals both require a fitted main-effect model"""
def fit_main_effect_model(G: np.ndarray, y: np.ndarray):
    model = LinearRegression()
    model.fit(G, y)
    y_hat = model.predict(G)
    return model, y_hat


def benchmark_one_method(method_name, y, y_hat_main, G, observed_scores, n_permutations=50):
    """
    Build null scores and p-values for one permutation method.
    This keeps all three methods under the same interface
    """
    null_scores = {pair: [] for pair in observed_scores.keys()}

    for k in range(n_permutations):
        seed = 1000 + k

        if method_name == "PermI":
            y_null = build_perm_i_target(y_true=y, y_hat_main=y_hat_main, seed=seed)
        elif method_name == "PermT":
            y_null = build_perm_t_target(y_true=y, seed=seed)
        elif method_name == "PermR":
            y_null = build_perm_r_target(y_true=y, y_hat_main=y_hat_main, seed=seed)
        else:
            raise ValueError(f"Unknown method: {method_name}")

        perm_scores = simple_pairwise_scores(G, y_null)

        for pair, score in perm_scores.items():
            null_scores[pair].append(score)

    pvalues = {
        pair: empirical_p_value(observed_scores[pair], np.array(null_scores[pair]))
        for pair in observed_scores
    }

    return null_scores, pvalues


def main():
    out_dir = Path("repro/outputs/perm_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Complex simulation: this has true interactions
    sim = ComplexSimulator(
        n_samples=5000,              # debugging size first
        n_genes=10,
        snps_per_gene=20,
        causal_prop=0.5,
        snr=0.1,
        main_interaction_ratio=1.0,
        seed=42,
    )

    X, y, G = sim.generate()

    # True interactions from the paper's complex simulator
    true_pairs = {(0, 1), (6, 8), (2, 4), (3, 7), (5, 6), (7, 9)}

    # All tested pairs
    observed_scores = simple_pairwise_scores(G, y)
    all_pairs = set(observed_scores.keys())

    # Fit main-effect model once
    _, y_hat_main = fit_main_effect_model(G, y)

    results = {}

    for method_name in ["PermI", "PermT", "PermR"]:
        print(f"Running {method_name}...")

        null_scores, pvalues = benchmark_one_method(
            method_name=method_name,
            y=y,
            y_hat_main=y_hat_main,
            G=G,
            observed_scores=observed_scores,
            n_permutations=50,
        )

        significant_pairs = classify_significant_pairs(pvalues, alpha=0.05)
        #TPR = power and TNR = specifity
        metrics = compute_tpr_tnr(
            significant_pairs=significant_pairs,
            true_pairs=true_pairs,
            all_pairs=all_pairs
        )

        results[method_name] = {
            "observed_scores": {str(k): v for k, v in observed_scores.items()},
            "null_scores": {str(k): v for k, v in null_scores.items()},
            "pvalues": {str(k): v for k, v in pvalues.items()},
            "metrics": metrics,
            "significant_pairs": [str(p) for p in sorted(significant_pairs)],
        }

        print(f"{method_name} metrics:", metrics)

    with open(out_dir / "perm_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved benchmark results to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()