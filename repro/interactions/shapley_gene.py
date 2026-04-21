import numpy as np
import torch


@torch.no_grad()
def predictor_from_gene_layer(model, gene_layer_tensor):
    """
    We don't want to use the whole model (the SNP encoder) we only need gene-layer predictor.
    The paper defines interaction scores between gene-layer nodes.
    Evaluate only the predictor head of the interaction model.
    Input shape: (batch, n_genes)
    Output shape: (batch,)
    """
    out = model.predictor(gene_layer_tensor)
    return out.reshape(-1)


def sample_random_subset(num_genes: int, i: int, j: int, rng: np.random.Generator):
    """
    Sample a random subset S (subset of the other genes) from all genes except i and j.
    Shapley interaction averages over subsets of the other features.
    """
    candidates = [k for k in range(num_genes) if k not in (i, j)]
    mask = rng.integers(0, 2, size=len(candidates))
    S = [c for c, m in zip(candidates, mask) if m == 1]
    return S


@torch.no_grad()
def shapley_interaction_score_pair(
    model,
    gene_layer: np.ndarray,
    i: int,
    j: int,
    baseline: np.ndarray,
    num_subset_samples: int = 50,
    device: str = "cpu",
    seed: int = 42,
):
    """
    Monte Carlo approximation of Shapley interaction score between gene i and j
    on the learned gene layer.
    
    Keep some other genes present
    Set the rest to baseline
    compare prediction with and without i,j
    average the interaction effect.

    gene_layer: shape (n_samples, n_genes)
    baseline: shape (n_genes,)
    """
    rng = np.random.default_rng(seed)

    model.eval()
    model = model.to(device)

    G = torch.tensor(gene_layer, dtype=torch.float32, device=device)
    B = torch.tensor(baseline, dtype=torch.float32, device=device).view(1, -1)
    B = B.repeat(G.shape[0], 1)

    n_samples, n_genes = gene_layer.shape
    deltas = []

    for _ in range(num_subset_samples):
        S = sample_random_subset(n_genes, i, j, rng)

        x_S = B.clone()
        x_Si = B.clone()
        x_Sj = B.clone()
        x_Sij = B.clone()

        # put S genes back
        if len(S) > 0:
            x_S[:, S] = G[:, S]
            x_Si[:, S] = G[:, S]
            x_Sj[:, S] = G[:, S]
            x_Sij[:, S] = G[:, S]

        # add i / j when appropriate
        x_Si[:, i] = G[:, i]
        x_Sj[:, j] = G[:, j]
        x_Sij[:, i] = G[:, i]
        x_Sij[:, j] = G[:, j]

        f_S = predictor_from_gene_layer(model, x_S)
        f_Si = predictor_from_gene_layer(model, x_Si)
        f_Sj = predictor_from_gene_layer(model, x_Sj)
        f_Sij = predictor_from_gene_layer(model, x_Sij)

        delta = f_Sij - f_Si - f_Sj + f_S
        deltas.append(delta.cpu().numpy())

    deltas = np.stack(deltas, axis=0)   # (num_subset_samples, n_samples)
    score = float(np.mean(np.abs(deltas)))
    return score


def compute_all_nn_interaction_scores(
    model,
    gene_layer: np.ndarray,
    num_subset_samples: int = 50,
    device: str = "cpu",
    seed: int = 42,
):
    """
    Runs that for every gene pair.
    Compute NN-based interaction scores for all gene pairs.
    """
    n_genes = gene_layer.shape[1]
    baseline = gene_layer.mean(axis=0)

    scores = {}
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            scores[(i, j)] = shapley_interaction_score_pair(
                model=model,
                gene_layer=gene_layer,
                i=i,
                j=j,
                baseline=baseline,
                num_subset_samples=num_subset_samples,
                device=device,
                seed=seed + i * 100 + j,
            )
    return scores