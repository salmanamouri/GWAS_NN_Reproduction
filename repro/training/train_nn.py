from typing import List, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from repro.training.nn_dataset import GeneDataset, collate_gene_batch

"""
    This gives you:

    train/validation split
    data loaders
    early stopping
    L1 regularization
    prediction
    access to the learned gene layer
    That last part is important because later the interaction score is computed between gene-layer nodes, not raw SNPs.
"""

def split_snps_by_gene(X: np.ndarray, n_genes: int, snps_per_gene: int):
    gene_blocks = []
    start = 0
    for _ in range(n_genes):
        end = start + snps_per_gene
        gene_blocks.append(X[:, start:end])
        start = end
    return gene_blocks


def make_train_val_split(X: np.ndarray, y: np.ndarray, val_fraction: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)

    split = int((1 - val_fraction) * len(y))
    train_idx = idx[:split]
    val_idx = idx[split:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    return X_train, y_train, X_val, y_val


def make_gene_loaders(
    X: np.ndarray,
    y: np.ndarray,
    n_genes: int,
    snps_per_gene: int,
    batch_size: int = 1024,
    val_fraction: float = 0.2,
    seed: int = 42,
):
    X_train, y_train, X_val, y_val = make_train_val_split(X, y, val_fraction=val_fraction, seed=seed)

    gene_blocks_train = split_snps_by_gene(X_train, n_genes=n_genes, snps_per_gene=snps_per_gene)
    gene_blocks_val = split_snps_by_gene(X_val, n_genes=n_genes, snps_per_gene=snps_per_gene)

    train_ds = GeneDataset(gene_blocks_train, y_train)
    val_ds = GeneDataset(gene_blocks_val, y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_gene_batch
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_gene_batch
    )

    return train_loader, val_loader


def train_model(
    model,
    train_loader,
    val_loader,
    l1_penalty_fn,
    lr: float = 0.005,
    l1_lambda: float = 1e-5,
    num_epochs: int = 300,
    patience: int = 20,
    device: str = "cpu",
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    wait = 0

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        for xs, y in train_loader:
            xs = [x.to(device) for x in xs]
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(xs)
            loss = criterion(pred, y) + l1_lambda * l1_penalty_fn(model)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xs, y in val_loader:
                xs = [x.to(device) for x in xs]
                y = y.to(device)

                pred = model(xs)
                loss = criterion(pred, y) + l1_lambda * l1_penalty_fn(model)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch+1:03d} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


@torch.no_grad()
def predict_model(model, X: np.ndarray, n_genes: int, snps_per_gene: int, device: str = "cpu"):
    model.eval()
    model = model.to(device)

    gene_blocks = split_snps_by_gene(X, n_genes=n_genes, snps_per_gene=snps_per_gene)
    xs = [torch.tensor(g, dtype=torch.float32).to(device) for g in gene_blocks]
    pred = model(xs).cpu().numpy().reshape(-1)
    return pred


@torch.no_grad()
def extract_gene_layer(model, X: np.ndarray, n_genes: int, snps_per_gene: int, device: str = "cpu"):
    model.eval()
    model = model.to(device)

    gene_blocks = split_snps_by_gene(X, n_genes=n_genes, snps_per_gene=snps_per_gene)
    xs = [torch.tensor(g, dtype=torch.float32).to(device) for g in gene_blocks]
    gene_repr = model.get_gene_layer(xs).cpu().numpy()
    return gene_repr