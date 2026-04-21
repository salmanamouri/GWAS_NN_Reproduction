import numpy as np


class NullSimulator:
    """
    Null simulator: NO interactions
    Only main effects + noise
    if there's an interaction detected then the method is not working correctly
    """

    def __init__(
        self,
        n_samples=5000,
        n_genes=10,
        snps_per_gene=20,
        causal_prop=0.5,
        snr=0.1,
        seed=42,
    ):
        self.n_samples = n_samples
        self.n_genes = n_genes
        self.snps_per_gene = snps_per_gene
        self.causal_prop = causal_prop
        self.snr = snr
        np.random.seed(seed)

    # -------------------------
    # SAME SNP GENERATION
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

    # -------------------------
    # SAME GENE LATENTS
    # -------------------------
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
    # MAIN EFFECT ONLY
    # -------------------------
    def main_effect(self, G):
        w = np.random.normal(0, 1, G.shape[1])
        return G @ w

    # -------------------------
    # NOISE
    # -------------------------
    def add_noise(self, main):
        var_signal = np.var(main)
        noise_var = var_signal / self.snr

        noise = np.random.normal(0, np.sqrt(noise_var), size=main.shape)
        return noise

    # -------------------------
    # FULL PIPELINE
    # -------------------------
    def generate(self):
        genes = self.generate_snps()
        G = self.generate_gene_latents(genes)

        main = self.main_effect(G)
        noise = self.add_noise(main)

        y = main + noise

        X = np.concatenate(genes, axis=1)

        return X, y, G