from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


def main():
    in_file = Path("repro/outputs/simple_benchmark/simple_benchmark_results.json")
    out_dir = Path("repro/outputs/simple_benchmark")
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
    plt.title("Simple interaction benchmark (Figure 4-style)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig4_simple_benchmark.png", dpi=200)
    plt.close()

    print("Saved:")
    print(out_dir / "fig4_simple_benchmark.png")


if __name__ == "__main__":
    main()