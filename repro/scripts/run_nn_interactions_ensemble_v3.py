from pathlib import Path
import json
import torch

from repro.simulators.complex_simulator import ComplexSimulator
from repro.training.train_nn import make_gene_loaders
from repro.training.train_ensemble import train_nn_ensemble, compute_ensemble_nn_scores
from repro.eval.metrics import compute_metrics


def serialize_scores(score_dict):
    return {str(k): float(v) for k, v in score_dict.items()}


def main():
    out_dir = Path("repro/outputs/nn_ensemble_v3")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # -------- Scaled dataset config --------
    n_samples = 40000 #This is the first paper-scale sample size. gives the NN much more data to learn (gene encoders and more stable rankings)
    n_genes = 10
    snps_per_gene = 20
    causal_prop = 0.5
    snr = 0.1
    main_interaction_ratio = 1.0

    sim = ComplexSimulator(
        n_samples=n_samples,
        n_genes=n_genes,
        snps_per_gene=snps_per_gene,
        causal_prop=causal_prop,
        snr=snr,
        main_interaction_ratio=main_interaction_ratio,
        seed=42,
    )

    X, y, G_true = sim.generate()
    true_pairs = {(0, 1), (6, 8), (2, 4), (3, 7), (5, 6), (7, 9)}

    train_loader, val_loader = make_gene_loaders(
        X=X,
        y=y,
        n_genes=n_genes,
        snps_per_gene=snps_per_gene,
        batch_size=4096, #A practical compromise on CPU. If memory is okay, you can try 8192
        val_fraction=0.2,
        seed=42,
    )

    ensemble_size = 10 #less than 50 (paper like) but much better than 5
    num_subset_samples = 200 #reduces monte carlo noise in the Shapley like score

    models, histories = train_nn_ensemble(
        train_loader=train_loader,
        val_loader=val_loader,
        gene_input_dims=[snps_per_gene] * n_genes,
        ensemble_size=ensemble_size,
        gene_hidden_dim=10,
        predictor_hidden_dim=100,
        lr=0.005,
        l1_lambda=1e-5,
        num_epochs=300,
        patience=20,
        device=device,
        base_seed=42,
    )

    scores = compute_ensemble_nn_scores(
        models=models,
        X=X,
        n_genes=n_genes,
        snps_per_gene=snps_per_gene,
        num_subset_samples=num_subset_samples,
        device=device,
        base_seed=42,
    )

    metrics = compute_metrics(scores, true_pairs)

    # top ranked pairs
    ranked_pairs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top10 = [(str(pair), float(score)) for pair, score in ranked_pairs[:10]]

    results = {
        "dataset": {
            "n_samples": n_samples,
            "n_genes": n_genes,
            "snps_per_gene": snps_per_gene,
            "causal_prop": causal_prop,
            "snr": snr,
            "main_interaction_ratio": main_interaction_ratio,
            "generator": "python_complex_simulator",
        },
        "ensemble": {
            "ensemble_size": ensemble_size,
            "num_subset_samples": num_subset_samples,
            "aggregation": "average_final_scores",
        },
        "metrics": metrics,
        "top10_pairs": top10,
        "scores": serialize_scores(scores),
    }

    with open(out_dir / "nn_ensemble_v3_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Ensemble V3 NN interaction metrics:", metrics)
    print("Top 10 pairs:")
    for pair, score in top10:
        print(pair, score)
    print(f"Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()