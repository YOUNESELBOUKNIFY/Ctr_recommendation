import torch
import random
import numpy as np

def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Metrics
from sklearn.metrics import roc_auc_score, log_loss

def compute_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

def compute_logloss(y_true, y_pred):
    return log_loss(y_true, y_pred)
