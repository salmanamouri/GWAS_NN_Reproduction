from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def list_python_files():
    return sorted([p for p in ROOT.glob("*.py")])


def extract_imports(file_path: Path):
    imports = []
    pattern_import = re.compile(r"^\s*import\s+(.+)$")
    pattern_from = re.compile(r"^\s*from\s+([A-Za-z0-9_\.]+)\s+import\s+(.+)$")

    for line in file_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m1 = pattern_import.match(line)
        m2 = pattern_from.match(line)
        if m1:
            imports.append(f"import {m1.group(1).strip()}")
        elif m2:
            imports.append(f"from {m2.group(1).strip()} import {m2.group(2).strip()}")
    return imports


def search_keywords(file_path: Path, keywords):
    matches = []
    text = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for i, line in enumerate(text, start=1):
        for kw in keywords:
            if kw in line:
                matches.append((i, kw, line.strip()))
    return matches


def main():
    py_files = list_python_files()
    print("Python files found in repo root:\n")
    for p in py_files:
        print("-", p.name)

    print("\n" + "=" * 80 + "\n")

    keywords = [
        "learning_rate",
        "batch_size",
        "num_epoch",
        "Softplus",
        "ReLU",
        "Adam",
        "MSELoss",
        "Shapley",
        "permutation",
        "residual",
        "Main_effect",
        "SparseNN",
    ]

    for p in py_files:
        print(f"\nFILE: {p.name}")
        print("-" * 80)

        imports = extract_imports(p)
        if imports:
            print("Imports:")
            for imp in imports:
                print("  ", imp)

        matches = search_keywords(p, keywords)
        if matches:
            print("\nKeyword matches:")
            for line_no, kw, line in matches:
                print(f"  [line {line_no:>4}] {kw}: {line}")


if __name__ == "__main__":
    main()