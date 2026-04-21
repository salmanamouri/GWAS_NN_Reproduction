from repro.baselines.common import (
    split_snps_by_gene,
    pca_representation_per_gene,
    fit_lasso_interaction_model,
)


def run_pca_lasso(X, y, n_genes=10, snps_per_gene=20, n_components=1, alpha=0.001):
    genes = split_snps_by_gene(X, snps_per_gene=snps_per_gene, n_genes=n_genes)
    G_repr = pca_representation_per_gene(genes, n_components=n_components)
    model, scores = fit_lasso_interaction_model(G_repr, y, alpha=alpha)
    return {
        "representation": "pca",
        "model_type": "lasso",
        "scores": scores,
    }