from typing import List
import torch
import torch.nn as nn


class GeneMLP(nn.Module):
    """
    This is the paper s gene-level encoder:
        -takes all SNPs of one gene
        -learns one scalar gene representation
    Per-gene MLP:
    mi -> 10 -> 1
    """
    def __init__(self, input_dim: int, hidden_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class GeneInteractionNN(nn.Module):
    """
    This is the full interaction model:
        -gene layer
        -nonliniear phenotype predictor
    Paper-style interaction model:
    - one GeneMLP per gene
    - concatenate gene representations
    - phenotype MLP: num_genes -> 100 -> 1
    """
    def __init__(self, gene_input_dims: List[int], gene_hidden_dim: int = 10, predictor_hidden_dim: int = 100):
        super().__init__()

        self.num_genes = len(gene_input_dims)

        self.gene_mlps = nn.ModuleList([
            GeneMLP(dim, hidden_dim=gene_hidden_dim)
            for dim in gene_input_dims
        ])

        self.predictor = nn.Sequential(
            nn.Linear(self.num_genes, predictor_hidden_dim),
            nn.ReLU(),
            nn.Linear(predictor_hidden_dim, 1)
        )

    def get_gene_layer(self, gene_inputs: List[torch.Tensor]):
        gene_reprs = [mlp(x) for mlp, x in zip(self.gene_mlps, gene_inputs)]
        return torch.cat(gene_reprs, dim=1)

    def forward(self, gene_inputs: List[torch.Tensor]):
        gene_layer = self.get_gene_layer(gene_inputs)
        y_hat = self.predictor(gene_layer)
        return y_hat


class MainEffectNN(nn.Module):
    """
    This is the null model:
        -same gene encoders 
        -linear head only
    Paper-style main-effect model:
    - same gene encoders
    - linear layer after gene layer
    """
    def __init__(self, gene_input_dims: List[int], gene_hidden_dim: int = 10):
        super().__init__()

        self.num_genes = len(gene_input_dims)

        self.gene_mlps = nn.ModuleList([
            GeneMLP(dim, hidden_dim=gene_hidden_dim)
            for dim in gene_input_dims
        ])

        self.linear_head = nn.Linear(self.num_genes, 1)

    def get_gene_layer(self, gene_inputs: List[torch.Tensor]):
        gene_reprs = [mlp(x) for mlp, x in zip(self.gene_mlps, gene_inputs)]
        return torch.cat(gene_reprs, dim=1)

    def forward(self, gene_inputs: List[torch.Tensor]):
        gene_layer = self.get_gene_layer(gene_inputs)
        y_hat = self.linear_head(gene_layer)
        return y_hat


def l1_penalty(model: nn.Module):
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        penalty = penalty + p.abs().sum()
    return penalty