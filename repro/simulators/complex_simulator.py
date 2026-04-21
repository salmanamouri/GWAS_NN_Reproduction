"""Data used: Artificial genotype–phenotype data generated from the paper’s Equation (1)"""

import numpy as np
import pandas as pd

class ComplexSimulator:
    """
    Implements the complex interaction simulator from the paper.
    """

    def __init__(
        self,
        n_samples=5000,
        n_genes=10,
        snps_per_gene=20,
        causal_prop=0.5,
        snr=0.1,
        main_interaction_ratio=1.0,
        seed=42,
    ):
        self.n_samples = n_samples
        self.n_genes = n_genes
        self.snps_per_gene = snps_per_gene
        self.causal_prop = causal_prop
        self.snr = snr
        self.ratio = main_interaction_ratio
        np.random.seed(seed)

    # -------------------------
    # STEP 1 — Generate SNPs: Mimics genotype data {0,1,2}
    # -------------------------
    def generate_snps(self):
        genes = []
        for _ in range(self.n_genes):
            maf = np.random.uniform(0.05, 0.5, self.snps_per_gene)
            X = np.zeros((self.n_samples, self.snps_per_gene))

            for j, p in enumerate(maf):
                X[:, j] = np.random.binomial(2, p, self.n_samples)

            X = X - X.mean(axis=0)
            genes.append(X)

        return genes

    # ------------------------------------------------------------------------------------
    # STEP 2 — Gene latent values: Implements gene = sparse linear combination of SNPs
    # ------------------------------------------------------------------------------------
    def generate_gene_latents(self, genes):
        G = []

        for X in genes:
            alpha = np.random.normal(0, 1, X.shape[1])
            mask = np.random.binomial(1, self.causal_prop, X.shape[1])

            if mask.sum() == 0:
                mask[np.random.randint(0, X.shape[1])] = 1

            g = X @ (alpha * mask)
            G.append(g)

        return np.stack(G, axis=1)

    # -------------------------
    # STEP 3 — Complex interactions: EXACT reproduction of Equation (1)
    # -------------------------
    def interaction_effect(self, G):
        return (
            np.maximum(G[:, 0], G[:, 1]) +
            np.maximum(G[:, 6], G[:, 8]) +
            (G[:, 2] - G[:, 4]) ** 2 +
            (G[:, 3] - G[:, 7]) ** 2 +
            G[:, 5] * G[:, 6] +
            G[:, 7] * G[:, 9]
        )

    # -------------------------
    # STEP 4 — Main effects: Adds additive genetic effect
    # -------------------------
    def main_effect(self, G):
        w = np.random.normal(0, 1, G.shape[1])
        return G @ w

    # -------------------------
    # STEP 5 — Add noise: Enforces Signal-to-Noise ratio
    # -------------------------
    def add_noise(self, main, interaction):
        signal = main + interaction
        var_signal = np.var(signal)

        noise_var = var_signal / self.snr
        noise = np.random.normal(0, np.sqrt(noise_var), size=signal.shape)

        return noise

    # -------------------------
    # FULL PIPELINE
    # -------------------------
    def generate(self):
        genes = self.generate_snps()
        G = self.generate_gene_latents(genes)

        interaction = self.interaction_effect(G)
        main = self.main_effect(G)

        # scale main effect
        main = main * self.ratio

        noise = self.add_noise(main, interaction)

        y = main + interaction + noise

        # flatten SNPs
        X = np.concatenate(genes, axis=1)

        return X, y, G