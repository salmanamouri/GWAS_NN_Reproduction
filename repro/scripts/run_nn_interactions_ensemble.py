from pathlib import Path
import json
import torch

from repro.simulators.complex_simulator import ComplexSimulator
from repro.training.train_nn import make_gene_loaders
from repro.training.train_ensemble import train_nn_ensemble, extract_ensemble_gene_layers
from repro.interactions.shapley_gene import compute_all_nn_interaction_scores
from repro.eval.metrics import compute_metrics


def serialize_scores(score_dict):
    return {str(k): float(v) for k, v in score_dict.items()}


def main():
    out_dir = Path("repro/outputs/nn_ensemble")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # -------- Dataset config --------
    n_samples = 20000
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
        batch_size=2048,
        val_fraction=0.2,
        seed=42,
    )

    ensemble_size = 5

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

    gene_layer = extract_ensemble_gene_layers(
        models=models,
        X=X,
        n_genes=n_genes,
        snps_per_gene=snps_per_gene,
        device=device,
    )

    print("Ensemble gene layer shape:", gene_layer.shape)

    # use one trained model’s predictor with averaged gene layer
    # simplest practical version: use first model for predictor head
    scores = compute_all_nn_interaction_scores(
        model=models[0],
        gene_layer=gene_layer,
        num_subset_samples=100,
        device=device,
        seed=42,
    )

    metrics = compute_metrics(scores, true_pairs)

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
            "num_subset_samples": 100,
        },
        "metrics": metrics,
        "gene_layer_shape": list(gene_layer.shape),
        "scores": serialize_scores(scores),
    }

    with open(out_dir / "nn_ensemble_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Ensemble NN interaction metrics:", metrics)
    print(f"Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()