import numpy as np
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, brier_score_loss, log_loss
)
from pgmpy.inference import VariableElimination
from .bn_model import infer_overheat_prob
from .utils import load_cfg

# ----------------------------------------------------------
# Full Bayesian Network evaluation
# ----------------------------------------------------------

def evaluate_bn(model, test_data, discretizer=None, cfg=None, save_dir="results/bn_evaluation", debug=False):
    """
    Full evaluation pipeline for the Bayesian Network.
    Computes metrics, prints them to console, and saves JSON/Plots.
    """

    # 1. Ensure Config is loaded
    if cfg is None:
        cfg = load_cfg()
        
    sensor_cols = cfg["bn"]["sensors"]
    
    # 2. Retrieve Threshold (Use 0.3 as default for better Recall if not set in yaml)
    threshold = cfg["bn"].get("bn_threshold", 0.3)
    
    print(f"\n[INFO] Starting Evaluation with Binarization Threshold: {threshold:.2f}")

    y_true = []
    y_pred = []
    y_prob = []
    
    # Instantiate inference engine once
    infer = VariableElimination(model)

    # ------------------------------------------------------
    # 3. Inference Loop
    # ------------------------------------------------------
    if debug:
        print(f"Processing {len(test_data)} test samples...")
        
    for _, row in test_data.iterrows():
        # Prepare discrete evidence
        evidence = {col: int(row[col]) for col in sensor_cols if col in row}
        
        # Calculate P(Overheat=1 | evidence)
        # We use the imported helper, or call infer.query directly if needed
        try:
            prob = infer_overheat_prob(model, evidence)
        except:
            prob = 0.0

        y_prob.append(prob)
        y_true.append(int(row["spindle_overheat"]))
        
        # Apply Threshold
        y_pred.append(1 if prob >= threshold else 0)

    # ------------------------------------------------------
    # 4. Metrics Calculation
    # ------------------------------------------------------
    acc = accuracy_score(y_true, y_pred)
    # Zero division handling: default to 0
    prec = precision_score(y_true, y_pred, zero_division=0, pos_label=1)
    rec = recall_score(y_true, y_pred, zero_division=0, pos_label=1)
    f1 = f1_score(y_true, y_pred, zero_division=0, pos_label=1)
    brier = brier_score_loss(y_true, y_prob)
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Get report as dict for JSON saving and string for printing
    report_dict = classification_report(y_true, y_pred, target_names=['Normal (0)', 'Overheat (1)'], zero_division=0, output_dict=True)
    report_text = classification_report(y_true, y_pred, target_names=['Normal (0)', 'Overheat (1)'], zero_division=0)

    # ------------------------------------------------------
    # 5. CONSOLE OUTPUT
    # ------------------------------------------------------
    print("\n" + "="*45)
    print(f"   BAYESIAN NETWORK EVALUATION (TH={threshold:.2f})")
    print("="*45)
    
    print("\n--- Confusion Matrix ---")
    cm_df = pd.DataFrame(cm, 
                         index=['Actual 0 (Normal)', 'Actual 1 (Overheat)'], 
                         columns=['Predicted 0', 'Predicted 1'])
    print(cm_df)

    print("\n--- Classification Report ---")
    print(report_text)

    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Brier Score (Calibration): {brier:.4f}")

    # ------------------------------------------------------
    # 6. Save Results (JSON)
    # ------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)

    results = {
        "threshold_used": threshold,
        "accuracy": acc,
        "precision_overheat": report_dict['Overheat (1)']['precision'],
        "recall_overheat": report_dict['Overheat (1)']['recall'],
        "f1_score_overheat": report_dict['Overheat (1)']['f1-score'],
        "brier_score": brier,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_text,
    }

    json_path = os.path.join(save_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n[INFO] Metrics JSON saved to: {json_path}")

    # ------------------------------------------------------
    # 7. Save Plots
    # ------------------------------------------------------
    
    # Confusion Matrix Plot
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix (Th={threshold})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center', fontsize=14, 
                     color="white" if cm[i, j] > cm.max()/2 else "black")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    # Probability Distribution Plot
    plt.figure(figsize=(8, 4))
    plt.hist([p for p, t in zip(y_prob, y_true) if t == 0],
             bins=20, alpha=0.6, label="Actual Normal (0)", color='blue')
    plt.hist([p for p, t in zip(y_prob, y_true) if t == 1],
             bins=20, alpha=0.6, label="Actual Overheat (1)", color='red')
    
    plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold {threshold}')
    
    plt.title("Distribution of Predicted Overheat Probabilities")
    plt.xlabel("P(Overheat = 1)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "probability_distribution.png"))
    plt.close()
    
    print(f"[INFO] Plots saved to: {save_dir}/")
    print("="*45 + "\n")

    return {'Summary': results}