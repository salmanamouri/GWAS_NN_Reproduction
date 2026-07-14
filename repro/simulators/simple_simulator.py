import numpy as np


class SimpleSimulator:
    """
    Paper-style simple interaction simulator:
    - interactions are driven by one SNP per gene pair
    - main effects can still depend on multiple SNPs
    - simpler than the complex simulator
    """

    def __init__(
        self,
        n_samples=40000,
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
        self.main_interaction_ratio = main_interaction_ratio
        np.random.seed(seed)

    def generate_snps(self):
        genes = []
        for _ in range(self.n_genes):
            maf = np.random.uniform(0.05, 0.5, self.snps_per_gene)
            X = np.zeros((self.n_samples, self.snps_per_gene), dtype=np.float32)

            for j, p in enumerate(maf):
                X[:, j] = np.random.binomial(2, p, self.n_samples)

            X = X - X.mean(axis=0)
            genes.append(X)

        return genes

    def generate_main_effects(self, genes):
        """
        Main effects may depend on multiple SNPs within each gene.
        """
        gene_effects = []
        for X in genes:
            alpha = np.random.normal(0, 1, X.shape[1]).astype(np.float32)
            mask = np.random.binomial(1, self.causal_prop, X.shape[1]).astype(np.float32)
            if mask.sum() == 0:
                mask[np.random.randint(0, X.shape[1])] = 1.0
            g = X @ (alpha * mask)
            gene_effects.append(g)

        G_main = np.stack(gene_effects, axis=1)
        w = np.random.normal(0, 1, self.n_genes).astype(np.float32)
        return G_main @ w

    def generate_simple_interactions(self, genes):
        """
        Simple SNP-level interactions:
        one selected SNP per interacting gene pair.
        """
        # same true pairs as complex simulator for consistency
        true_pairs = [(0, 1), (6, 8), (2, 4), (3, 7), (5, 6), (7, 9)]

        interaction = np.zeros(self.n_samples, dtype=np.float32)
        chosen_snps = {}

        for (i, j) in true_pairs:
            si = np.random.randint(0, self.snps_per_gene)
            sj = np.random.randint(0, self.snps_per_gene)
            chosen_snps[(i, j)] = (si, sj)

            xi = genes[i][:, si]
            xj = genes[j][:, sj]

            beta = np.random.normal(0, 1)
            interaction += beta * (xi * xj)

        return interaction, true_pairs, chosen_snps

    def add_noise(self, signal):
        var_signal = np.var(signal)
        noise_var = var_signal / self.snr
        noise = np.random.normal(0, np.sqrt(noise_var), size=signal.shape).astype(np.float32)
        return noise

    def generate(self):
        genes = self.generate_snps()
        main = self.generate_main_effects(genes)
        interaction, true_pairs, chosen_snps = self.generate_simple_interactions(genes)

        # scale main effects relative to interaction term
        main_std = np.std(main) + 1e-8
        int_std = np.std(interaction) + 1e-8
        main = main * (self.main_interaction_ratio * int_std / main_std)

        signal = main + interaction
        noise = self.add_noise(signal)

        y = signal + noise
        X = np.concatenate(genes, axis=1)

        metadata = {
            "true_pairs": true_pairs,
            "chosen_snps": {str(k): v for k, v in chosen_snps.items()},
        }

        return X, y, metadata