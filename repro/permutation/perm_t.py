import numpy as np

""" simplest null construction:

shuffle y directly
destroy everything:
main effects
interactions
structure

The paper includes this as a comparison and argues it is not appropriate for neural networks.
"""


def build_perm_t_target(y_true: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Perm T:
    permute the target directly.
    """
    rng = np.random.default_rng(seed)
    return rng.permutation(y_true)