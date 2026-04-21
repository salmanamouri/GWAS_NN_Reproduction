# mathematical core of Perm I y^π=y^_​main​+r^π
#y^_main is the fitted main effect
import numpy as np


def build_perm_i_target(y_true: np.ndarray, y_hat_main: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Perm I from the paper:
    y_null = predicted_main + permuted_residual

    residual = y_true - y_hat_main
    permuted_residual = permutation(residual)
    """
    rng = np.random.default_rng(seed)

    residual = y_true - y_hat_main
    residual_perm = rng.permutation(residual)
    y_null = y_hat_main + residual_perm

    return y_null


def empirical_p_value(observed_score: float, null_scores: np.ndarray) -> float:
    """
    One-sided empirical p-value:
    proportion of null scores >= observed score
    """
    null_scores = np.asarray(null_scores)
    return float((1 + np.sum(null_scores >= observed_score)) / (len(null_scores) + 1))