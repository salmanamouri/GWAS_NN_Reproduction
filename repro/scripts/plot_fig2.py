import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    in_file = Path("repro/outputs/perm_benchmark/perm_benchmark_results.json")
    out_dir = Path("repro/outputs/fig2")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(in_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    methods = ["PermI", "PermT", "PermR"]
    tpr_values = [results[m]["metrics"]["tpr"] for m in methods]
    tnr_values = [results[m]["metrics"]["tnr"] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    plt.figure(figsize=(7, 4.5))
    plt.bar(x - width / 2, tpr_values, width, label="TPR (Power)")
    plt.bar(x + width / 2, tnr_values, width, label="TNR (Specificity)")
    plt.xticks(x, methods)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Permutation method comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig2_permutation_comparison.png", dpi=200)
    plt.close()

    # simple text summary
    summary_lines = []
    for m in methods:
        metrics = results[m]["metrics"]
        summary_lines.append(
            f"{m}: "
            f"TP={metrics['tp']}, FN={metrics['fn']}, "
            f"FP={metrics['fp']}, TN={metrics['tn']}, "
            f"TPR={metrics['tpr']:.4f}, TNR={metrics['tnr']:.4f}"
        )

    with open(out_dir / "fig2_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print("Saved:")
    print(out_dir / "fig2_permutation_comparison.png")
    print(out_dir / "fig2_summary.txt")


if __name__ == "__main__":
    main()