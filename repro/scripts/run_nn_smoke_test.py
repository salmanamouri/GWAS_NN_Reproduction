from pathlib import Path
import json
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import torch

from repro.simulators.complex_simulator import ComplexSimulator
from repro.models.paper_models import GeneInteractionNN, MainEffectNN, l1_penalty
from repro.training.train_nn import make_gene_loaders, predict_model, extract_gene_layer, split_snps_by_gene, train_model


def main():
    out_dir = Path("repro/outputs/nn_smoke_test")
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

    train_loader, val_loader = make_gene_loaders(
        X=X,
        y=y,
        n_genes=10,
        snps_per_gene=20,
        batch_size=1024,
        val_fraction=0.2,
        seed=42,
    )

    gene_input_dims = [20] * 10

    print("\nTraining interaction NN...")
    interaction_model = GeneInteractionNN(
        gene_input_dims=gene_input_dims,
        gene_hidden_dim=10,
        predictor_hidden_dim=100,
    )

    interaction_model, interaction_history = train_model(
        interaction_model,
        train_loader,
        val_loader,
        l1_penalty_fn=l1_penalty,
        lr=0.005,
        l1_lambda=1e-5,
        num_epochs=300,
        patience=20,
        device=device,
    )

    y_pred = predict_model(interaction_model, X, n_genes=10, snps_per_gene=20, device=device)
    mse = float(mean_squared_error(y, y_pred))
    r2 = float(r2_score(y, y_pred))

    print("\nTraining main-effect NN...")
    main_model = MainEffectNN(
        gene_input_dims=gene_input_dims,
        gene_hidden_dim=10,
    )

    main_model, main_history = train_model(
        main_model,
        train_loader,
        val_loader,
        l1_penalty_fn=l1_penalty,
        lr=0.005,
        l1_lambda=1e-5,
        num_epochs=300,
        patience=20,
        device=device,
    )

    y_pred_main = predict_model(main_model, X, n_genes=10, snps_per_gene=20, device=device)
    mse_main = float(mean_squared_error(y, y_pred_main))
    r2_main = float(r2_score(y, y_pred_main))

    gene_repr = extract_gene_layer(interaction_model, X, n_genes=10, snps_per_gene=20, device=device)

    results = {
        "interaction_model": {
            "mse": mse,
            "r2": r2,
        },
        "main_effect_model": {
            "mse": mse_main,
            "r2": r2_main,
        },
        "gene_layer_shape": list(gene_repr.shape),
    }

    with open(out_dir / "nn_smoke_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nResults:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()