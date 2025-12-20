import random
import numpy as np
import torch

def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_auc(y_true, y_pred):
    """
    AUC sans sklearn (robuste).
    Attention: y_true doit être {0,1}.
    """
    y_true = np.asarray(y_true).astype(np.int32)
    y_pred = np.asarray(y_pred).astype(np.float64)

    order = np.argsort(y_pred)
    y_true_sorted = y_true[order]

    n_pos = y_true_sorted.sum()
    n_neg = len(y_true_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # rank sum method
    ranks = np.arange(1, len(y_true_sorted) + 1)
    rank_sum_pos = ranks[y_true_sorted == 1].sum()

    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)
