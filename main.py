"""
main.py
Entry point for the Hybrid CNC Diagnosis System
"""

import sys
import random
import numpy as np
import pandas as pd
import os

from src.integration import run_demo, run_real
# [UPDATED] Added evaluate_root_causes to imports
from src.evaluation import evaluate_bn, evaluate_root_causes
from src.config import load_cfg, path_for

# Global variables to store loaded items
MODEL = None
TEST_DATA = None
DISCRETIZER = None
CFG = None

# ============================================================
# INPUT HELPERS
# ============================================================

def ask_demo_or_real():
    """Ask if using MANUAL CPDs (demo) or REAL learned model."""
    while True:
        user = input("Run with MANUAL CPDs (demo) or REAL learned model? (demo/real) [D=demo, R=real]: ").strip().lower()
        if user == "d":
            return "demo"
        if user == "r":
            return "real"
        print("Invalid input. Please enter D or R.\n")


def ask_yes_no(prompt):
    """Ask Y/N."""
    while True:
        v = input(prompt).strip().lower()
        if v == "y":
            return True
        if v == "n":
            return False
        print("Invalid input. Enter Y or N.\n")


def ask_mode():
    """Ask for mode 1, 2, or 3."""
    while True:
        v = input(
            "Select mode:\n"
            "1. Single Example Diagnosis (Manual Input)\n"
            "2. Live Diagnosis (Random Faulty Sample)\n"
            "3. Evaluate on Test Set\n"
            "Enter choice (1/2/3): "
        ).strip()
        if v in ("1", "2", "3"):
            return int(v)
        print("Invalid choice. Please enter 1, 2, or 3.\n")


# ============================================================
# MAIN
# ============================================================

def main():
    global MODEL, TEST_DATA, DISCRETIZER, CFG
    
    # MODE: DEMO or REAL
    mode = ask_demo_or_real()

    # DEBUG
    debug = ask_yes_no("Enable DEBUG mode? (y/n): ")

    # Retraining only applies to REAL mode
    force_retrain = False
    if mode == "real":
        force_retrain = ask_yes_no("Force re-training (ignore cached model)? (y/n): ")

    # Choose WHAT to do
    choice = ask_mode()

    # ============================================================
    # DEMO MODE (manual CPDs)
    # ============================================================
    if mode == "demo":
        if choice == 1:
            print("\n[DEMO] Manual single example diagnosis.\n")
            evidence = {
                "vibration_rms": int(input("Vibration (0/1): ")),
                "spindle_temp": int(input("SpindleTemp (0/1): ")),
                "coolant_flow": int(input("CoolantFlow (0/1): ")),
            }
            run_demo(evidence, debug=debug)

        elif choice == 2:
            print("\n[DEMO] Live diagnosis (random simulated example).\n")
            evidence = {
                "vibration_rms": random.randint(0, 1),
                "spindle_temp": random.randint(0, 1),
                "coolant_flow": random.randint(0, 1),
            }
            print("[Random Evidence]:", evidence)
            run_demo(evidence, debug=debug)

        elif choice == 3:
            print("\n[DEMO] Evaluation is NOT available in manual mode.")
            print("Switch to REAL mode for full performance evaluation.\n")
        
        return # End DEMO mode


    # ============================================================
    # REAL MODE
    # ============================================================

    # Capture all 4 return values from run_real: model, test_data_disc, discretizer, cfg
    # TEST_DATA now contains 'actual_cause' if integration.py was updated correctly
    MODEL, TEST_DATA, DISCRETIZER, CFG = run_real(
        evidence=None,
        debug=debug,
        force_retrain=force_retrain,
        return_test_data=True
    )
    
    # ------------------------------------------------------------
    # MODE 1 — Manual Example
    # ------------------------------------------------------------
    if choice == 1:
        print("\n[REAL] Manual single example diagnosis.\n")
        sensors = CFG["bn"]["sensors"]

        evidence = {}
        for s in sensors:
            val = input(f"{s} (numeric value or 0/1): ").strip()
            try:
                evidence[s] = float(val)
            except:
                evidence[s] = 0

        result = run_demo(
            evidence=evidence,
            debug=debug,
            model_override=MODEL,
            discretizer=DISCRETIZER
        )

        print("\n[REAL] Result:", result)

    # ------------------------------------------------------------
    # MODE 2 — Live Diagnosis (Random Test Sample)
    # ------------------------------------------------------------
    elif choice == 2:
        print("[REAL] Live diagnosis (random test sample).")

        sensor_cols = CFG["bn"]["sensors"]

        # Load RAW continuous dataset (Only needed here for the pretty print)
        tel = pd.read_csv(path_for(CFG, "telemetry"))
        lab = pd.read_csv(path_for(CFG, "labels"))
        df_raw = tel.join(lab["spindle_overheat"])

        # Filter for ACTUAL FAULTS to ensure the demo is interesting
        # We look for rows in TEST_DATA where the target 'spindle_overheat' is 1
        faulty_data = TEST_DATA[TEST_DATA['spindle_overheat'] == 1]
        
        # If faults exist in test set, pick one. Otherwise random.
        if not faulty_data.empty:
            idx = faulty_data.sample(1).index[0]
            if debug:
                print(f"[INFO] Demo: Selected a known FAULTY sample (Index {idx})")
        else:
            idx = TEST_DATA.sample(1).index[0]

        sample_disc = TEST_DATA.loc[idx]     # DISCRETE for BN inference
        sample_raw = df_raw.loc[idx]         # CONTINUOUS only for the display

        # Discrete evidence for BN (exclude actual_cause if present)
        evidence = {col: int(sample_disc[col]) for col in sensor_cols if col in sample_disc.index}

        # Pretty continuous evidence for printing:
        evidence_readable = {
            col: float(f"{sample_raw[col]:.3f}") if isinstance(sample_raw[col], float) else sample_raw[col]
            for col in sensor_cols if col in sample_raw.index
        }

        print("\n[Evidence]:", evidence_readable)
        
        # Check ground truth if available
        if 'actual_cause' in sample_disc:
            print(f"[Ground Truth Cause]: {sample_disc['actual_cause']}")

        # Run full BN → KG → Decision pipeline
        result = run_demo(
            evidence=evidence,
            debug=debug,
            model_override=MODEL,
            discretizer=DISCRETIZER
        )

        print("\n[Live Diagnosis Result]")
        print(result)

    # ------------------------------------------------------------
    # MODE 3 — Evaluate BN on Test Set
    # ------------------------------------------------------------
    elif choice == 3:
        print("\n[REAL] Evaluating on full test set...\n")
        
        # 1. Standard BN Evaluation (Target: Overheat)
        metrics = evaluate_bn(
            model=MODEL, 
            test_data=TEST_DATA, 
            discretizer=DISCRETIZER, 
            cfg=CFG,
            debug=debug
        )

        # 2. Root Cause Diagnosis Evaluation (Target: Actual Cause)
        # This requires TEST_DATA to have 'actual_cause' column
        evaluate_root_causes(MODEL, TEST_DATA, debug=debug)
        
        # Confirmation print
        if metrics:
            print("\n--- Evaluation Complete ---")
            print("Summary Metrics (Overheat Detection):", metrics.get('Summary', {}))


if __name__ == "__main__":
    main()