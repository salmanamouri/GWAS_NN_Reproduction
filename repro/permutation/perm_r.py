import numpy as np


def build_perm_r_target(y_true: np.ndarray, y_hat_main: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Perm R: residual-only permutation strategy:
    compute residual = y - y_hat_main
    then use permuted residual as the regression target (shuffle them) then use shuffled residuals alone as target 

    This removes the main-effect structure from the target.
    """
    rng = np.random.default_rng(seed)
    residual = y_true - y_hat_main
    return rng.permutation(residual)