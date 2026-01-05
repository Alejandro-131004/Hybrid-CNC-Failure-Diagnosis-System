import pandas as pd
import numpy as np
import os

def augment_data():
    tel_path = "data/telemetry.csv"
    lab_path = "data/labels.csv"
    
    print(f"Loading data from: {tel_path}...")
    try:
        telemetry = pd.read_csv(tel_path)
        labels = pd.read_csv(lab_path)
    except FileNotFoundError:
        print("Error: Files not found.")
        return

    # Align datasets
    min_len = min(len(telemetry), len(labels))
    telemetry = telemetry.iloc[:min_len].copy()
    labels = labels.iloc[:min_len].copy()
    n_rows = len(telemetry)
    
    cause_labels = pd.Series(["None"] * n_rows, name="actual_cause")
    n_anomalies = int(n_rows * 0.30)
    np.random.seed(42)
    anomaly_indices = np.random.choice(n_rows, n_anomalies, replace=False)
    
    print(f"Injecting faults in {n_anomalies} samples...")

    idx_fan = anomaly_indices[: n_anomalies // 3]
    idx_bearing = anomaly_indices[n_anomalies // 3 : 2 * n_anomalies // 3]
    idx_filter = anomaly_indices[2 * n_anomalies // 3 :]

    # --- 1. SENSOR VALUE INJECTION (Our part) ---
    # The goal is to create clear scenarios for the Bayesian Network to learn.

    # A. FanFault (High temperature, others normal)
    current_temps = telemetry.loc[idx_fan, "spindle_temp"]
    new_temps = np.maximum(
        current_temps + 40,
        np.random.uniform(95, 120, size=len(idx_fan))
    )
    telemetry.loc[idx_fan, "spindle_temp"] = np.round(new_temps, 2)

    # Cleanup: ensure no noise in other sensors
    telemetry.loc[idx_fan, "vibration_rms"] = np.round(
        np.random.uniform(0.5, 1.2, size=len(idx_fan)), 3
    )
    telemetry.loc[idx_fan, "coolant_flow"] = np.round(
        np.random.uniform(0.8, 1.2, size=len(idx_fan)), 3
    )
    cause_labels.loc[idx_fan] = "FanFault"
    
    # B. BearingWearHigh (High vibration, normal flow)
    new_vibration = np.random.uniform(1.6, 3.5, size=len(idx_bearing))
    telemetry.loc[idx_bearing, "vibration_rms"] = np.round(new_vibration, 3)
    
    current_temps = telemetry.loc[idx_bearing, "spindle_temp"]
    new_temps_b = np.maximum(
        current_temps + 15,
        np.random.uniform(90, 105, size=len(idx_bearing))
    )
    telemetry.loc[idx_bearing, "spindle_temp"] = np.round(new_temps_b, 2)

    # Cleanup: ensure normal flow
    telemetry.loc[idx_bearing, "coolant_flow"] = np.round(
        np.random.uniform(0.8, 1.2, size=len(idx_bearing)), 3
    )
    cause_labels.loc[idx_bearing] = "BearingWearHigh"

    # C. CloggedFilter (Low flow + high vibration to satisfy professor's rule)
    new_flow = np.random.uniform(0.05, 0.4, size=len(idx_filter))  # < 0.6 guaranteed
    telemetry.loc[idx_filter, "coolant_flow"] = np.round(new_flow, 3)
    
    # Professor states that Overheat occurs if Vib > 1.5 AND Flow < 0.6
    # Therefore, for Filter faults we must increase vibration as well.
    new_vib_f = np.random.uniform(1.6, 2.5, size=len(idx_filter))
    telemetry.loc[idx_filter, "vibration_rms"] = np.round(new_vib_f, 3)
    
    current_temps = telemetry.loc[idx_filter, "spindle_temp"]
    new_temps_f = np.maximum(
        current_temps + 20,
        np.random.uniform(92, 115, size=len(idx_filter))
    )
    telemetry.loc[idx_filter, "spindle_temp"] = np.round(new_temps_f, 2)
    cause_labels.loc[idx_filter] = "CloggedFilter"

    # --- 2. PROFESSOR'S LOGIC (Official fix) ---
    # We use EXACTLY her logic to define the target (spindle_overheat)
    cond = (
        (telemetry["spindle_temp"] > 87) |
        ((telemetry["vibration_rms"] > 1.5) &
         (telemetry["coolant_flow"] < 0.6))
    )

    labels["spindle_overheat"] = 0

    # Apply the rule
    # This will capture ALL injected faults (values were forced to satisfy the rule)
    labels.loc[cond, "spindle_overheat"] = 1
    
    # --- SAVE ---
    if not os.path.exists("data"):
        os.makedirs("data")

    telemetry.to_csv("data/telemetry.csv", index=False)
    labels.to_csv("data/labels.csv", index=False)
    cause_labels.to_csv("data/causes_ground_truth.csv", index=False)
    
    print("Augmentation Done. Data is clean and consistent with Professor's logic.")

if __name__ == "__main__":
    augment_data()
