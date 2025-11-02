"""
main.py
Entry point for the Hybrid CNC Diagnosis System
"""

from src.integration import run_demo, run_real

if __name__ == "__main__":

    # Ask user which mode to run
    mode = input("Run with MANUAL CPDs (demo) or REAL learned model? (demo/real): ").strip().lower()
    debug = input("Enable DEBUG mode? (y/n): ").strip().lower() == "y"

    # Example binary evidence (0 = normal, 1 = abnormal)
    evidence = {
        "Vibration": 1,
        "CoolantFlow": 1,
        "SpindleTemp": 0
    }

    if mode == "real":
        result = run_real(evidence, debug=debug)
    else:
        result = run_demo(evidence, debug=debug)
