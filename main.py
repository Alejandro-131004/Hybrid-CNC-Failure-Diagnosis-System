"""
main.py
Entry point for the Hybrid CNC Diagnosis System
"""

from src.integration import run_demo, run_real, evaluate_on_test_set

if __name__ == "__main__":

    # Ask user which mode to run
    mode = input("Run with MANUAL CPDs (demo) or REAL learned model? (demo/real): ").strip().lower()
    debug = input("Enable DEBUG mode? (y/n): ").strip().lower() == "y"
    
    if mode == "real":
        # For real mode, ask about retraining and evaluation
        force_retrain = input("Force re-training (ignore cached model)? (y/n): ").strip().lower() == "y"
        
        print("\nSelect mode:")
        print("1. Single Example Diagnosis (Manual Input)")
        print("2. Live Diagnosis (Random Faulty Sample)")
        print("3. Evaluate on Test Set")
        choice = input("Enter choice (1/2/3): ").strip()
        
        if choice == "3":
            # Get model and test data
            model, test_data = run_real(None, debug=debug, force_retrain=force_retrain, return_test_data=True)
            
            # Evaluate on test set
            results = evaluate_on_test_set(model, test_data, debug=debug)
            
        elif choice == "2":
            # Live Diagnosis
            model, test_data = run_real(None, debug=debug, force_retrain=force_retrain, return_test_data=True)
            
            # Filter for faulty samples (spindle_overheat=1) to make it interesting
            faulty_samples = test_data[test_data["spindle_overheat"] == 1]
            
            if len(faulty_samples) > 0:
                sample = faulty_samples.sample(1).iloc[0]
                print("\n=== [LIVE DIAGNOSIS] ===")
                print(f"Selected random faulty sample (Index: {sample.name})")
                
                # Extract evidence (only sensor columns)
                from src.utils import load_cfg
                cfg = load_cfg()
                sensor_cols = cfg["bn"]["sensors"]
                evidence = {col: int(sample[col]) for col in sensor_cols if col in test_data.columns}
                print(f"Sensors: {evidence}")
                
                # Run diagnosis
                run_real(evidence, debug=debug, force_retrain=False)
            else:
                print("No faulty samples found in test set.")
                
        else:
            # Single example inference
            evidence = {
                "vibration_rms": 1,
                "coolant_flow": 1, 
                "spindle_temp": 0
            }
            result = run_real(evidence, debug=debug, force_retrain=force_retrain)
    else:
        # Demo mode uses manual BN with CamelCase node names
        evidence = {
            "Vibration": 1,
            "CoolantFlow": 1,
            "SpindleTemp": 0
        }
        result = run_demo(evidence, debug=debug)

