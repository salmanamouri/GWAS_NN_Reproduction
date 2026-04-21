import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    print("Starting plot script...")

    out_dir = Path("repro/outputs/null_calibration")
    print(f"Reading from: {out_dir.resolve()}")

    with open(out_dir / "pvalues.json", "r", encoding="utf-8") as f:
        pvalue_dict = json.load(f)

    pvalues = np.array(list(pvalue_dict.values()), dtype=float)

    print("Number of p-values:", len(pvalues))
    print("Mean p-value:", pvalues.mean())

    # Histogram
    plt.figure()
    plt.hist(pvalues, bins=10, range=(0, 1))
    plt.xlabel("p-value")
    plt.ylabel("Count")
    plt.title("Null p-value histogram")
    plt.savefig(out_dir / "pvalue_histogram.png")
    plt.close()

    # Calibration curve
    thresholds = np.linspace(0.01, 0.5, 50)
    empirical = [(pvalues <= t).mean() for t in thresholds]

    plt.figure()
    plt.plot(thresholds, empirical, label="Empirical")
    plt.plot(thresholds, thresholds, linestyle="--", label="Ideal")
    plt.xlabel("Threshold")
    plt.ylabel("False positive rate")
    plt.legend()
    plt.title("Calibration curve")
    plt.savefig(out_dir / "calibration_curve.png")
    plt.close()

    print("Plots saved!")


if __name__ == "__main__":
    main()