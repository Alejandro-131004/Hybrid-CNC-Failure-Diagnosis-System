import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, brier_score_loss, log_loss
)

from .bn_model import infer_overheat_prob
from .utils import load_cfg


# ----------------------------------------------------------
# Basic evaluation helpers
# ----------------------------------------------------------

def logloss_score(y_true, y_pred):
    """Computes standard log-loss score."""
    return log_loss(y_true, y_pred)


def decision_accuracy(ec_list, chosen):
    """
    Computes the percentage of optimal decisions.
    Each 'ec' is a dict of expected costs for each action.
    """
    ok = sum(1 for ec, a in zip(ec_list, chosen) if a == min(ec, key=ec.get))
    return ok / len(chosen)


# ----------------------------------------------------------
# Full Bayesian Network evaluation
# ----------------------------------------------------------

def evaluate_bn(model, test_data, save_dir="results/bn_evaluation"):
    """
    Full evaluation pipeline for the Bayesian Network.

    Computes:
        - Accuracy
        - Precision
        - Recall
        - F1-score
        - Brier score (probability calibration)
        - Confusion matrix
        - Classification report

    Saves:
        - metrics.json
        - confusion_matrix.png
        - probability_distribution.png
    """

    cfg = load_cfg()
    sensor_cols = cfg["bn"]["sensors"]

    y_true = []
    y_pred = []
    y_prob = []

    # ------------------------------------------------------
    # 1. Inference loop
    # ------------------------------------------------------
    for _, row in test_data.iterrows():
        evidence = {col: int(row[col]) for col in sensor_cols if col in row}
        prob = infer_overheat_prob(model, evidence)

        y_prob.append(prob)
        y_true.append(int(row["spindle_overheat"]))
        y_pred.append(1 if prob >= 0.5 else 0)

    # ------------------------------------------------------
    # 2. Metrics
    # ------------------------------------------------------
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)

    # ------------------------------------------------------
    # 3. Ensure output directory exists
    # ------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------
    # 4. Save metrics as JSON
    # ------------------------------------------------------
    results = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "brier_score": brier,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    # ------------------------------------------------------
    # 5. Plot: Confusion Matrix
    # ------------------------------------------------------
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    # ------------------------------------------------------
    # 6. Plot: Distribution of Probability Estimates
    # ------------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.hist([p for p, t in zip(y_prob, y_true) if t == 0],
             bins=20, alpha=0.7, label="True label = 0")
    plt.hist([p for p, t in zip(y_prob, y_true) if t == 1],
             bins=20, alpha=0.7, label="True label = 1")
    plt.title("Distribution of Predicted Overheat Probabilities")
    plt.xlabel("P(overheat = 1)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "probability_distribution.png"))
    plt.close()

    return results, y_true, y_pred, y_prob
