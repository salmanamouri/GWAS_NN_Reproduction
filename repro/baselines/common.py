import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def split_snps_by_gene(X: np.ndarray, snps_per_gene: int, n_genes: int):
    """
    Split full SNP matrix X into a list of per-gene SNP matrices.
    Assumes equal SNP count per gene for now.
    """
    genes = []
    start = 0
    for _ in range(n_genes):
        end = start + snps_per_gene
        genes.append(X[:, start:end])
        start = end
    return genes


def select_top_snp_per_gene(genes, y):
    """
    For each gene, choose the SNP with maximum absolute correlation with y.
    Returns gene representation matrix of shape (n_samples, n_genes).
    """
    reps = []

    for Xg in genes:
        corrs = []
        for j in range(Xg.shape[1]):
            c = np.corrcoef(Xg[:, j], y)[0, 1]
            if np.isnan(c):
                c = 0.0
            corrs.append(abs(c))

        best_idx = int(np.argmax(corrs))
        reps.append(Xg[:, best_idx])

    return np.stack(reps, axis=1)


def pca_representation_per_gene(genes, n_components=1):
    """
    For each gene, run PCA on its SNP block and keep n_components per gene.
    If n_components=1, final shape = (n_samples, n_genes).
    """
    reps = []

    for Xg in genes:
        pca = PCA(n_components=n_components)
        Zg = pca.fit_transform(Xg)
        reps.append(Zg)

    return np.concatenate(reps, axis=1)


def build_pairwise_linear_design(G_repr: np.ndarray):
    """
    Build main effects + pairwise multiplicative interaction terms.

    Input:
        G_repr shape = (n_samples, n_gene_features)

    Output:
        design matrix with:
        [main effects | pairwise products]
        plus list of pair indices corresponding to the interaction columns
    """
    n_samples, n_features = G_repr.shape
    interaction_terms = []
    interaction_pairs = []

    for i in range(n_features):
        for j in range(i + 1, n_features):
            interaction_terms.append((G_repr[:, i] * G_repr[:, j]).reshape(-1, 1))
            interaction_pairs.append((i, j))

    if interaction_terms:
        interactions = np.concatenate(interaction_terms, axis=1)
        design = np.concatenate([G_repr, interactions], axis=1)
    else:
        design = G_repr.copy()

    return design, interaction_pairs


def fit_lr_interaction_model(G_repr: np.ndarray, y: np.ndarray):
    """
    Fit linear regression on:
    [main effects | pairwise products]

    Return pairwise scores as abs(coefficients of interaction terms).
    """
    design, interaction_pairs = build_pairwise_linear_design(G_repr)

    model = LinearRegression()
    model.fit(design, y)

    coef = model.coef_.reshape(-1)
    interaction_coef = coef[G_repr.shape[1]:]

    scores = {
        pair: abs(float(c))
        for pair, c in zip(interaction_pairs, interaction_coef)
    }

    return model, scores


def fit_lasso_interaction_model(G_repr: np.ndarray, y: np.ndarray, alpha=0.001):
    """
    Same design as LR, but use Lasso.
    """
    design, interaction_pairs = build_pairwise_linear_design(G_repr)

    model = Lasso(alpha=alpha, max_iter=20000)
    model.fit(design, y)

    coef = model.coef_.reshape(-1)
    interaction_coef = coef[G_repr.shape[1]:]

    scores = {
        pair: abs(float(c))
        for pair, c in zip(interaction_pairs, interaction_coef)
    }

    return model, scores


def fit_xgb_pairwise_screen(G_repr: np.ndarray, y: np.ndarray, random_state=42):
    """
    XGB does not give simple per-pair coefficients.
    For this baseline scaffold, we score each pair separately by fitting
    a small XGB on [Gi, Gj, Gi*Gj] and using the achieved fit quality as score.
    """
    if not HAS_XGB:
        raise ImportError("xgboost is not installed.")

    n_features = G_repr.shape[1]
    scores = {}

    for i in range(n_features):
        for j in range(i + 1, n_features):
            pair_X = np.column_stack([
                G_repr[:, i],
                G_repr[:, j],
                G_repr[:, i] * G_repr[:, j]
            ])

            model = XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                objective="reg:squarederror",
                verbosity=0,
            )
            model.fit(pair_X, y)
            pred = model.predict(pair_X)
            score = max(0.0, float(r2_score(y, pred)))
            scores[(i, j)] = score

    return scores