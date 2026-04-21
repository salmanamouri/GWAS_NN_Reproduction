import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

from repro.simulators.null_simulator import NullSimulator
from repro.permutation.perm_i import build_perm_i_target, empirical_p_value


def simple_pairwise_scores(G: np.ndarray, y: np.ndarray):
    """
    Proxy pairwise interaction scores under the null calibration step.

    For each pair (i,j), use the absolute correlation between y and Gi*Gj
    as a simple score. Under the null, these should not be systematically large.

    This is a temporary scaffold for calibration debugging before replacing
    it with the NN + Shapley score pipeline.
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


def fit_main_effect_model(G: np.ndarray, y: np.ndarray):
    """
    Main-effect-only predictor.
    For the calibration scaffold we use linear regression on G.
    Later this will be replaced by the paper-faithful MainEffectNN.
    """
    model = LinearRegression()
    model.fit(G, y)
    y_hat = model.predict(G)
    return model, y_hat


def main():
    out_dir = Path("repro/outputs/null_calibration")
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = NullSimulator(
        n_samples=5000,
        n_genes=10,
        snps_per_gene=20,
        causal_prop=0.5,
        snr=0.1,
        seed=42,
    )

    X, y, G = sim.generate()

    # observed scores on null data
    observed_scores = simple_pairwise_scores(G, y)

    # fit main-effect model
    _, y_hat_main = fit_main_effect_model(G, y)

    # build null distribution for each pair using Perm I
    n_permutations = 50
    null_scores = {pair: [] for pair in observed_scores.keys()}

    for k in range(n_permutations):
        y_null = build_perm_i_target(y_true=y, y_hat_main=y_hat_main, seed=1000 + k)
        perm_scores = simple_pairwise_scores(G, y_null)

        for pair, score in perm_scores.items():
            null_scores[pair].append(score)

    # empirical p-values
    pvalues = {
        pair: empirical_p_value(observed_scores[pair], np.array(null_scores[pair]))
        for pair in observed_scores
    }

    # save results
    with open(out_dir / "observed_scores.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in observed_scores.items()}, f, indent=2)

    with open(out_dir / "null_scores.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in null_scores.items()}, f, indent=2)

    with open(out_dir / "pvalues.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in pvalues.items()}, f, indent=2)

    print(f"Saved results to: {out_dir.resolve()}")
    print(f"Number of tested gene pairs: {len(pvalues)}")
    print(f"Mean p-value: {np.mean(list(pvalues.values())):.4f}")


if __name__ == "__main__":
    main()