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
        evaluate = input("Evaluate on test set? (y/n): ").strip().lower() == "y"
        
        if evaluate:
            # Get model and test data
            model, test_data = run_real(None, debug=debug, force_retrain=force_retrain, return_test_data=True)
            
            # Evaluate on test set
            results = evaluate_on_test_set(model, test_data, debug=debug)
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

