"""
main.py
Entry point for the Hybrid CNC Diagnosis System
"""

import sys
import random
from src.integration import run_demo, run_real
from src.evaluation import evaluate_bn
from src.config import load_cfg, path_for
import numpy as np
import pandas as pd

# ============================================================
#  INPUT HELPERS
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
#  MAIN
# ============================================================

def main():

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
    # DEMO MODE  (manual CPDs)
    # ============================================================
    if mode == "demo":

        if choice == 1:
            print("\n[DEMO] Manual single example diagnosis.\n")
            evidence = {
                "Vibration": int(input("Vibration (0/1): ")),
                "SpindleTemp": int(input("SpindleTemp (0/1): ")),
                "CoolantFlow": int(input("CoolantFlow (0/1): ")),
            }
            run_demo(evidence, debug=debug)

        elif choice == 2:
            print("\n[DEMO] Live diagnosis (random simulated example).\n")
            evidence = {
                "Vibration": random.randint(0, 1),
                "SpindleTemp": random.randint(0, 1),
                "CoolantFlow": random.randint(0, 1),
            }
            print("[Random Evidence]:", evidence)
            run_demo(evidence, debug=debug)

        elif choice == 3:
            print("\n[DEMO] Evaluation is NOT available in manual mode.")
            print("Switch to REAL mode for full performance evaluation.\n")

        return  # end DEMO mode


    # ============================================================
    # REAL MODE
    # ============================================================

    # IMPORTANT: load/train the REAL model ONCE
    model, test_data = run_real(
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
        cfg = load_cfg()
        sensors = cfg["bn"]["sensors"]

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
            model_override=model
        )

        print("\n[REAL] Result:", result)

    # ------------------------------------------------------------
    # MODE 2 — Live Diagnosis (Random Test Sample)
    # ------------------------------------------------------------
    elif choice == 2:
        print("[REAL] Live diagnosis (random test sample).")

        cfg = load_cfg()
        sensor_cols = cfg["bn"]["sensors"]

        # Load RAW continuous dataset
        tel = pd.read_csv(path_for(cfg, "telemetry"))
        lab = pd.read_csv(path_for(cfg, "labels"))
        df_raw = tel.join(lab["spindle_overheat"])

        # Pick SAME INDEX on discretized and raw data
        idx = test_data.sample(1).index[0]

        sample_disc = test_data.loc[idx]     # DISCRETE for BN inference
        sample_raw = df_raw.loc[idx]         # CONTINUOUS only for the display

        # Discrete evidence for BN:
        evidence = {col: int(sample_disc[col]) for col in sensor_cols if col in sample_disc.index}

        # Pretty continuous evidence for printing:
        evidence_readable = {
            col: float(f"{sample_raw[col]:.3f}") if isinstance(sample_raw[col], float) else sample_raw[col]
            for col in sensor_cols if col in sample_raw.index
        }

        print("\n[Evidence]:", evidence_readable)

        # Run full BN → KG → Decision pipeline
        result = run_demo(
            evidence=evidence,
            debug=False,
            model_override=model
        )

        print("\n[Live Diagnosis Result]")
        print(result)

    # ------------------------------------------------------------
    # MODE 3 — Evaluate BN on Test Set
    # ------------------------------------------------------------
    elif choice == 3:
        print("\n[REAL] Evaluating on full test set...\n")
        evaluate_bn(model, test_data)


if __name__ == "__main__":
    main()
