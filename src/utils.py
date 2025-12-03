import torch
import random
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

def set_seed(seed=2025):
    """Fixe la graine aléatoire pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Rend les calculs déterministes (peut ralentir un peu l'entraînement)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def compute_auc(y_true, y_pred):
    """
    Calcule l'AUC (Area Under Curve).
    Gère le cas rare où y_true n'a qu'une seule classe.
    """
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        # Cela arrive si y_true contient seulement des 0 ou seulement des 1
        return 0.5

def compute_logloss(y_true, y_pred):
    """Calcule la LogLoss (Binary Cross Entropy)."""
    # epsilon est géré automatiquement par sklearn, mais on s'assure des types
    return log_loss(y_true, y_pred, labels=[0, 1])