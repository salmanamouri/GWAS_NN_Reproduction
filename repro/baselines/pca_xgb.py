from repro.baselines.common import (
    split_snps_by_gene,
    pca_representation_per_gene,
    fit_xgb_pairwise_screen,
)


def run_pca_xgb(X, y, n_genes=10, snps_per_gene=20, n_components=1):
    genes = split_snps_by_gene(X, snps_per_gene=snps_per_gene, n_genes=n_genes)
    G_repr = pca_representation_per_gene(genes, n_components=n_components)
    scores = fit_xgb_pairwise_screen(G_repr, y)
    return {
        "representation": "pca",
        "model_type": "xgb",
        "scores": scores,
    }