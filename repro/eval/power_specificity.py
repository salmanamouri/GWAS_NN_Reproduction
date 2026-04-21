import numpy as np

"""Figure 2 is about:

power = true positive rate
specificity = true negative rate

So we need a clean function that computes those from p-values and known ground-truth interactions."""

def classify_significant_pairs(pvalues: dict, alpha: float = 0.05) -> set:
    """
    Return the set of pairs declared significant at threshold alpha.
    """
    return {pair for pair, p in pvalues.items() if p < alpha}


def compute_tpr_tnr(significant_pairs: set, true_pairs: set, all_pairs: set):
    """
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    """
    false_pairs = all_pairs - true_pairs

    tp = len(significant_pairs & true_pairs)
    fn = len(true_pairs - significant_pairs)

    fp = len(significant_pairs & false_pairs)
    tn = len(false_pairs - significant_pairs)

    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "tpr": float(tpr),
        "tnr": float(tnr),
    }