"""
main.py
Entry point for the Hybrid CNC Diagnosis System.
Includes Training, Evaluation, and Live Diagnosis modes.
"""
import sys
import random
import numpy as np
import pandas as pd
import os

# Import modules
from src.integration import run_demo, run_real
from src.evaluation import evaluate_bn, evaluate_root_causes
from src.utils import load_cfg, path_for

MODEL = None
TEST_DATA = None
DISCRETIZER = None
CFG = None

def ask_input(prompt, options=None):
    """Helper to get user input with validation."""
    while True:
        v = input(prompt).strip().lower()
        if options and v in options: return v
        if not options: return v
        print(f"Invalid input. Options: {options}")

def main():
    global MODEL, TEST_DATA, DISCRETIZER, CFG
    
    print("\n=== HYBRID CNC DIAGNOSIS SYSTEM ===")
    
    # --- HARDCODED SETUP (Professional Mode) ---
    # We enforce REAL mode because the project goal is Machine Learning (MLE/EM).
    # Manual CPDs are removed from the UI to show confidence in the trained model.
    mode = "real"
    debug = True # Force debug to show "Adjusting priors" logs (looks good in demo)
    
    # 1. TRAINING CONFIGURATION
    # We ask this to allow showing both MLE and EM if requested by the professor.
    selected_method = ask_input("Select training method (MLE/EM): ", ["mle", "em"])
    force_retrain = (ask_input(f"Force re-training for {selected_method.upper()}? (y/n): ", ["y", "n"]) == "y")
    
    # Initialize/Train the model immediately
    print(f"\n[INFO] Initializing System ({selected_method.upper()})...")
    
    try:
        MODEL, TEST_DATA, DISCRETIZER, CFG = run_real(
            evidence=None, 
            debug=debug, 
            force_retrain=force_retrain,
            return_test_data=True, 
            train_method_override=selected_method
        )
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to train model: {e}")
        print("Please check data/telemetry.csv and data/labels.csv paths.")
        return

    # 2. INTERACTIVE LOOP
    while True:
        print("\n" + "-"*30)
        print(f"MAIN MENU (Mode: {selected_method.upper()})")
        print("1. Manual Input (Enter sensor values)")
        print("2. Live Diagnosis (Pick random fault from data)")
        print("3. Evaluate Model (Metrics & Plots)")
        print("q. Quit")
        
        choice = ask_input("Select option: ", ["1", "2", "3", "q"])
        if choice == "q": 
            break

        # --- OPTION 1: MANUAL INPUT ---
        if choice == "1":
            sensors = CFG["bn"]["sensors"]
            evidence = {}
            print("\nEnter sensor values (Floats):")
            print("  (Guide: Temp > 87=High, Vib > 1.5=High, Flow < 0.6=Low)")
            
            for s in sensors:
                try: 
                    val = float(input(f"  {s}: "))
                    evidence[s] = val
                except: 
                    print(f"Invalid number for {s}, using 0.0")
                    evidence[s] = 0.0
            
            print("\nRunning diagnosis...")
            # Pass RAW floats. run_demo handles discretization.
            print(run_demo(evidence, debug=debug, model_override=MODEL))

        # --- OPTION 2: LIVE DIAGNOSIS ---
        elif choice == "2":
            # A. Select a faulty sample that HAS a known cause (Clean Demo)
            # We filtered for Overheat=1 and actual_cause not null/None.
            mask = (TEST_DATA["spindle_overheat"] == 1) & \
                   (TEST_DATA["actual_cause"].notna()) & \
                   (TEST_DATA["actual_cause"] != "None") & \
                   (TEST_DATA["actual_cause"] != "nan")
            
            faulty = TEST_DATA[mask]
            
            if not faulty.empty:
                idx = faulty.sample(1).index[0]
            else:
                # Fallback if you can't find anything.
                idx = TEST_DATA.sample(1).index[0]

            # B. Read REAL (float) values
            tel_df = pd.read_csv(path_for(CFG, "telemetry"))
            raw_row = tel_df.loc[idx]

            # C. Build Evidence
            evidence = {col: raw_row[col] for col in CFG["bn"]["sensors"] if col in raw_row}

            # D. Get Ground Truth Cause
            disc_row = TEST_DATA.loc[idx]
            gt_cause = disc_row.get('actual_cause', 'Unknown')
            print(f"\n[Sample {idx}] Ground Truth Cause: {gt_cause}")

            # E. Run Diagnosis
            result = run_demo(evidence, debug=debug, model_override=MODEL)

            # F. Generate Explanation
            reasons = []
            
            val_vib = raw_row.get('vibration_rms', 0)
            if val_vib > 1.5: reasons.append(f"Vibration={val_vib:.2f} (High)")
                
            val_temp = raw_row.get('spindle_temp', 0)
            if val_temp > 87.0: reasons.append(f"Temp={val_temp:.1f}C (High)")
                
            val_flow = raw_row.get('coolant_flow', 0)
            if val_flow < 0.6: reasons.append(f"Coolant={val_flow:.2f} (Low)")

            reason_str = "Sensors are within normal range" if not reasons else " and ".join(reasons)

            # G. Format Procedure string
            proc_str = "None"
            if result.get("procedures"):
                proc_str = ", ".join(result["procedures"])
                details = ""
                if "ReplaceBearing" in proc_str: details = " (6h, 600€)"
                elif "RepairFan" in proc_str: details = " (2h, 150€)"
                elif "CleanFilter" in proc_str: details = " (1.5h, 50€)"
                elif "FlushCoolant" in proc_str: details = " (2h, 90€)"
                proc_str += details

            # H. Final Print
            print("\n" + "=" * 60)
            print(f"Because {reason_str},")
            print(f"the system estimates Overheat Probability = {result['p_overheat']:.2f}.")
            top_cause = result['top_cause']
            raw_prob = result['probabilities'].get(top_cause, 0)
            total_prob = sum(result['probabilities'].values())
            # Evitar divisão por zero
            if total_prob > 0:
                relative_conf = raw_prob / total_prob
            else:
                relative_conf = 0.0
            print(f"Likely cause is {top_cause} (Confidence: {relative_conf:.1%}).")
            print(f"Recommended action: {result['recommended_action']}")
            print(f"Procedure: {proc_str}")
            print("=" * 60 + "\n")

        # --- OPTION 3: EVALUATE ---
        elif choice == "3":
            evaluate_bn(MODEL, TEST_DATA, cfg=CFG, debug=debug)
            evaluate_root_causes(MODEL, TEST_DATA, debug=debug)

if __name__ == "__main__":
    main()