import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main():
    in_file = Path("repro/outputs/figure3/figure3_results.json")
    out_dir = Path("repro/outputs/figure3")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(in_file, "r") as f:
        results = json.load(f)

    methods = list(results.keys())

    auroc = [results[m]["metrics"]["auroc"] for m in methods]
    ap = [results[m]["metrics"]["ap"] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, auroc, width, label="AUROC")
    plt.bar(x + width/2, ap, width, label="Average Precision")

    plt.xticks(x, methods, rotation=30)
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("Figure 3 — Baseline comparison")
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_dir / "fig3_baseline_comparison.png", dpi=200)
    plt.close()

    print("Figure 3 saved!")


if __name__ == "__main__":
    main()