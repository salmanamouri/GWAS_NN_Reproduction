import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def scores_to_labels_and_values(score_dict, true_pairs):
    """
    Convert:
        {(i,j): score}
    into:
        y_true (0/1), y_score
    """
    y_true = []
    y_score = []

    for pair, score in score_dict.items():
        y_score.append(score)
        y_true.append(1 if pair in true_pairs else 0)

    return np.array(y_true), np.array(y_score)


def compute_metrics(score_dict, true_pairs):
    y_true, y_score = scores_to_labels_and_values(score_dict, true_pairs)

    # avoid crash if constant
    if len(np.unique(y_true)) < 2:
        return {"auroc": None, "ap": None}

    auroc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    return {
        "auroc": float(auroc),
        "ap": float(ap)
    }