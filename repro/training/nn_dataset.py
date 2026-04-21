from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset

"""
NN takes a list of per-gene SNP matrices, not one flat matrix. 
So we need a dataset class that:
    -splits each sample into gene blocks
    -returns them in the right format for the model
"""
class GeneDataset(Dataset):
    """
    Dataset where each sample is returned as:
    ([gene1_snps, gene2_snps, ..., geneM_snps], y)
    """
    def __init__(self, gene_blocks: List[np.ndarray], y: np.ndarray):
        self.gene_blocks = [torch.tensor(g, dtype=torch.float32) for g in gene_blocks]
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        n = self.y.shape[0]
        for g in self.gene_blocks:
            if g.shape[0] != n:
                raise ValueError("All gene blocks must have same number of samples")

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        x = [g[idx] for g in self.gene_blocks]
        y = self.y[idx]
        return x, y


def collate_gene_batch(batch):
    num_genes = len(batch[0][0])

    xs = []
    for g in range(num_genes):
        xs.append(torch.stack([item[0][g] for item in batch], dim=0))

    y = torch.stack([item[1] for item in batch], dim=0)
    return xs, y