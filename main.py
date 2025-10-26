"""
main.py
Entry point for the Hybrid CNC Diagnosis System
"""

from src.integration import run_demo

if __name__ == "__main__":
    # === Example sensor evidence ===
    # 0 = normal / low, 1 = high / abnormal
    evidence = {
        "Vibration": 1,      # high vibration
        "CoolantFlow": 1,    # low coolant flow
        "SpindleTemp": 65     # normal temperature
    }

    # Run the integrated reasoning pipeline
    result = run_demo(evidence)

    # Display results
    print("\n=== Posterior Probabilities ===")
    for c, p in result["probabilities"].items():
        print(f"{c}: {p:.2f}")

    print("\n=== Expected Costs ===")
    for a, c in result["expected_costs"].items():
        print(f"{a}: {c:.2f}")

    print("\n=== Recommended Action ===")
    print(result["recommended_action"])
