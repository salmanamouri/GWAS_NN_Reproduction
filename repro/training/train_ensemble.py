import numpy as np
import torch

from repro.models.paper_models import GeneInteractionNN, l1_penalty
from repro.training.train_nn import train_model, extract_gene_layer
from repro.interactions.shapley_gene import compute_all_nn_interaction_scores


def set_torch_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

"""
Before
We averaged:
    gene layers
Now
We average:
    final interaction scores
"""
def train_nn_ensemble(
    train_loader,
    val_loader,
    gene_input_dims,
    ensemble_size=5,
    gene_hidden_dim=10,
    predictor_hidden_dim=100,
    lr=0.005,
    l1_lambda=1e-5,
    num_epochs=300,
    patience=20,
    device="cpu",
    base_seed=42,
):
    models = []
    histories = []

    for k in range(ensemble_size):
        seed = base_seed + k
        set_torch_seed(seed)

        print(f"\nTraining ensemble model {k+1}/{ensemble_size} (seed={seed})")

        model = GeneInteractionNN(
            gene_input_dims=gene_input_dims,
            gene_hidden_dim=gene_hidden_dim,
            predictor_hidden_dim=predictor_hidden_dim,
        )

        model, history = train_model(
            model,
            train_loader,
            val_loader,
            l1_penalty_fn=l1_penalty,
            lr=lr,
            l1_lambda=l1_lambda,
            num_epochs=num_epochs,
            patience=patience,
            device=device,
        )

        models.append(model)
        histories.append(history)

    return models, histories


def compute_ensemble_nn_scores(
    models,
    X,
    n_genes,
    snps_per_gene,
    num_subset_samples=100,
    device="cpu",
    base_seed=42,
):
    """
    Compute NN interaction scores for each ensemble member separately,
    then average the scores across models.
    """
    all_score_dicts = []

    for idx, model in enumerate(models):
        print(f"Computing interaction scores for ensemble model {idx+1}/{len(models)}")

        gene_layer = extract_gene_layer(
            model,
            X,
            n_genes=n_genes,
            snps_per_gene=snps_per_gene,
            device=device,
        )

        scores = compute_all_nn_interaction_scores(
            model=model,
            gene_layer=gene_layer,
            num_subset_samples=num_subset_samples,
            device=device,
            seed=base_seed + idx,
        )

        all_score_dicts.append(scores)

    # average per-pair scores
    pairs = all_score_dicts[0].keys()
    avg_scores = {
        pair: float(np.mean([score_dict[pair] for score_dict in all_score_dicts]))
        for pair in pairs
    }

    return avg_scores