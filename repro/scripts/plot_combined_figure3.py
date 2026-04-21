from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


def main():
    in_file = Path("repro/outputs/combined_benchmark/combined_results.json")
    out_dir = Path("repro/outputs/combined_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(in_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    methods = list(results.keys())
    auroc = [results[m]["metrics"]["auroc"] for m in methods]
    ap = [results[m]["metrics"]["ap"] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    plt.figure(figsize=(11, 5.5))
    plt.bar(x - width / 2, auroc, width, label="AUROC")
    plt.bar(x + width / 2, ap, width, label="Average Precision")
    plt.xticks(x, methods, rotation=30, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Combined benchmark: NN vs baselines")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "combined_figure3.png", dpi=200)
    plt.close()

    # text summary
    lines = []
    for m in methods:
        metrics = results[m]["metrics"]
        lines.append(
            f"{m}: AUROC={metrics['auroc']:.4f}, AP={metrics['ap']:.4f}"
        )

    with open(out_dir / "combined_figure3_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Saved:")
    print(out_dir / "combined_figure3.png")
    print(out_dir / "combined_figure3_summary.txt")


if __name__ == "__main__":
    main()