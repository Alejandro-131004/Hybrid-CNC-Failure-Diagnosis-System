"""
main.py
Entry point for the Hybrid CNC Diagnosis System
"""
import sys
import random
import numpy as np
import pandas as pd
import os

# Import from UTILS now
from src.integration import run_demo, run_real
from src.evaluation import evaluate_bn, evaluate_root_causes
from src.utils import load_cfg, path_for # <--- CORRIGIDO

MODEL = None
TEST_DATA = None
DISCRETIZER = None
CFG = None

def ask_input(prompt, options=None):
    while True:
        v = input(prompt).strip().lower()
        if options and v in options: return v
        if not options: return v
        print(f"Invalid input. Options: {options}")

def main():
    global MODEL, TEST_DATA, DISCRETIZER, CFG
    
    # 1. SETUP MODE
    mode_input = ask_input("Run with MANUAL CPDs (demo) or REAL learned model? [d/r]: ", ["d", "demo", "r", "real"])
    mode = "real" if mode_input in ["r", "real"] else "demo"
    
    debug = (ask_input("Enable DEBUG mode? (y/n): ", ["y", "n"]) == "y")
    
    # 2. REAL MODE CONFIG
    force_retrain = False
    selected_method = "mle" 

    if mode == "real":
        # Ask Method explicitly
        selected_method = ask_input("Select training method (MLE/EM): ", ["mle", "em"])
        force_retrain = (ask_input(f"Force re-training for {selected_method.upper()}? (y/n): ", ["y", "n"]) == "y")

    # 3. OPERATION CHOICE
    choice = int(ask_input("Select:\n1. Manual Input\n2. Live Diagnosis (Random)\n3. Evaluate\nChoice (1/2/3): ", ["1", "2", "3"]))

    # --- DEMO EXECUTION ---
    if mode == "demo":
        if choice == 1:
            evidence = {
                "vibration_rms": int(ask_input("Vibration (0/1): ", ["0", "1"])),
                "spindle_temp": int(ask_input("SpindleTemp (0/1): ", ["0", "1"])),
                "coolant_flow": int(ask_input("CoolantFlow (0/1): ", ["0", "1"])),
            }
            print(run_demo(evidence, debug=debug))
        elif choice == 2:
            evidence = {"vibration_rms": random.randint(0, 1), "spindle_temp": random.randint(0, 1), "coolant_flow": random.randint(0, 1)}
            print(f"[Random]: {evidence}")
            print(run_demo(evidence, debug=debug))
        return

    # --- REAL EXECUTION ---
    MODEL, TEST_DATA, DISCRETIZER, CFG = run_real(
        evidence=None, debug=debug, force_retrain=force_retrain,
        return_test_data=True, train_method_override=selected_method
    )
    
    if choice == 1: # Manual
        sensors = CFG["bn"]["sensors"]
        evidence = {}
        print("\nEnter sensor values:")
        for s in sensors:
            try: evidence[s] = float(input(f"{s}: "))
            except: evidence[s] = 0.0
        print(run_demo(evidence, debug=debug, model_override=MODEL))

    elif choice == 2:  # Live
        # 1. Select a faulty sample (Target=1)
        faulty = TEST_DATA[TEST_DATA["spindle_overheat"] == 1]
        if not faulty.empty:
            idx = faulty.sample(1).index[0]
        else:
            idx = TEST_DATA.sample(1).index[0]

        # 2. Read the REAL (float) values so the print looks nice
        # (TEST_DATA only has 0/1, so we need the original telemetry CSV)
        tel_df = pd.read_csv(path_for(CFG, "telemetry"))
        raw_row = tel_df.loc[idx]

        # 3. Build discrete evidence for the model
        disc_row = TEST_DATA.loc[idx]
        evidence = {col: int(disc_row[col]) for col in CFG["bn"]["sensors"] if col in disc_row}

        print(f"\n[Sample {idx}] Ground Truth Cause: {disc_row.get('actual_cause', 'Unknown')}")

        # 4. Run diagnosis
        result = run_demo(evidence, debug=debug, model_override=MODEL)

        # 5. Generate explanation (professor-style format)
        # Example: "Because Vibration=high and Temp=high..."

        reasons = []
        # Check which sensors are in alarm state (value 1)
        if evidence.get("vibration_rms") == 1:
            reasons.append(f"Vibration={raw_row['vibration_rms']:.2f} (High)")
        if evidence.get("spindle_temp") == 1:
            reasons.append(f"Temp={raw_row['spindle_temp']:.1f}C (High)")
        if evidence.get("coolant_flow") == 1:
            reasons.append(f"Coolant={raw_row['coolant_flow']:.2f} (Low)")

        if not reasons:
            reason_str = "Sensors are within normal range"
        else:
            reason_str = " and ".join(reasons)

        # Format procedure string with simulated cost/time details (for nicer output)
        proc_str = "None"
        if result["procedures"]:
            proc_list = result["procedures"]
            proc_str = ", ".join(proc_list)

            details = ""
            if "ReplaceBearing" in proc_str:
                details = "(6h, 600€)"
            elif "RepairFan" in proc_str:
                details = "(2h, 150€)"
            elif "CleanFilter" in proc_str:
                details = "(1.5h, 50€)"
            elif "FlushCoolant" in proc_str:
                details = "(2h, 90€)"

            proc_str += f" {details}"

        # 6. FINAL PRINT
        print("\n" + "=" * 60)
        print(f"Because {reason_str},")
        print(f"the system estimates Overheat Probability = {result['p_overheat']:.2f}.")
        print(
            f"Likely cause is {result['top_cause']} "
            f"({result['probabilities'].get(result['top_cause'], 0):.1%})."
        )
        print(f"Recommended action: {result['recommended_action']}")
        print(f"Procedure: {proc_str}")
        print("=" * 60 + "\n")

    elif choice == 3: # Evaluate
        evaluate_bn(MODEL, TEST_DATA, cfg=CFG, debug=debug)
        evaluate_root_causes(MODEL, TEST_DATA, debug=debug)

if __name__ == "__main__":
    main()