from pathlib import Path
import json

import torch

from repro.simulators.complex_simulator import ComplexSimulator
from repro.models.paper_models import GeneInteractionNN, l1_penalty
from repro.training.train_nn import make_gene_loaders, train_model, extract_gene_layer
from repro.interactions.shapley_gene import compute_all_nn_interaction_scores
from repro.eval.metrics import compute_metrics


def serialize_scores(score_dict):
    return {str(k): float(v) for k, v in score_dict.items()}


def main():
    out_dir = Path("repro/outputs/nn_interactions")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    sim = ComplexSimulator(
        n_samples=5000,
        n_genes=10,
        snps_per_gene=20,
        causal_prop=0.5,
        snr=0.1,
        main_interaction_ratio=1.0,
        seed=42,
    )

    X, y, G_true = sim.generate()
    true_pairs = {(0, 1), (6, 8), (2, 4), (3, 7), (5, 6), (7, 9)}

    train_loader, val_loader = make_gene_loaders(
        X=X,
        y=y,
        n_genes=10,
        snps_per_gene=20,
        batch_size=1024,
        val_fraction=0.2,
        seed=42,
    )

    model = GeneInteractionNN(
        gene_input_dims=[20] * 10,
        gene_hidden_dim=10,
        predictor_hidden_dim=100,
    )

    model, history = train_model(
        model,
        train_loader,
        val_loader,
        l1_penalty_fn=l1_penalty,
        lr=0.005,
        l1_lambda=1e-5,
        num_epochs=300,
        patience=20,
        device=device,
    )

    gene_layer = extract_gene_layer(model, X, n_genes=10, snps_per_gene=20, device=device)
    print("Gene layer shape:", gene_layer.shape)

    scores = compute_all_nn_interaction_scores(
        model=model,
        gene_layer=gene_layer,
        num_subset_samples=50,
        device=device,
        seed=42,
    )

    metrics = compute_metrics(scores, true_pairs)

    results = {
        "metrics": metrics,
        "gene_layer_shape": list(gene_layer.shape),
        "scores": serialize_scores(scores),
    }

    with open(out_dir / "nn_interaction_scores.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("NN interaction metrics:", metrics)
    print(f"Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()