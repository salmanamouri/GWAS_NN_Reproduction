from repro.baselines.common import (
    split_snps_by_gene,
    select_top_snp_per_gene,
    fit_lr_interaction_model,
)


def run_topsnp_lr(X, y, n_genes=10, snps_per_gene=20):
    genes = split_snps_by_gene(X, snps_per_gene=snps_per_gene, n_genes=n_genes)
    G_repr = select_top_snp_per_gene(genes, y)
    model, scores = fit_lr_interaction_model(G_repr, y)
    return {
        "representation": "top_snp",
        "model_type": "lr",
        "scores": scores,
    }