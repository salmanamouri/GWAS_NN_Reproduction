import numpy as np


def compute_uniform_bins(pvalues, n_bins=10):
    """
    Histogram counts for p-values in equal-width bins between 0 and 1.
    """
    counts, edges = np.histogram(pvalues, bins=n_bins, range=(0.0, 1.0))
    return counts, edges


def calibration_curve(pvalues, thresholds=None):
    """
    For each threshold alpha, compute:
    empirical false positive rate = proportion(p <= alpha)
    Under perfect calibration under the null, this should match alpha.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.50, 50)

    pvalues = np.asarray(pvalues)
    empirical_fpr = np.array([(pvalues <= a).mean() for a in thresholds])

    return thresholds, empirical_fpr