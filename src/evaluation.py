import numpy as np
from sklearn.metrics import log_loss

def logloss_score(y_true, y_pred):
    return log_loss(y_true, y_pred)

def decision_accuracy(ec_list, chosen):
    """Percentagem de decisões ótimas."""
    ok = sum(1 for ec, a in zip(ec_list, chosen) if a == min(ec, key=ec.get))
    return ok / len(chosen)
