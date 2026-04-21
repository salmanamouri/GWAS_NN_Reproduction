from repro.baselines.common import (
    split_snps_by_gene,
    select_top_snp_per_gene,
    fit_xgb_pairwise_screen,
)


def run_topsnp_xgb(X, y, n_genes=10, snps_per_gene=20):
    genes = split_snps_by_gene(X, snps_per_gene=snps_per_gene, n_genes=n_genes)
    G_repr = select_top_snp_per_gene(genes, y)
    scores = fit_xgb_pairwise_screen(G_repr, y)
    return {
        "representation": "top_snp",
        "model_type": "xgb",
        "scores": scores,
    }