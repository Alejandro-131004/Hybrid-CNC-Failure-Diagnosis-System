import pandas as pd
import numpy as np
import os

def augment_data():
    # File paths
    tel_path = "data/telemetry.csv"
    lab_path = "data/labels.csv"
    
    print(f"Loading data from: {tel_path}...")
    
    # Load original data
    try:
        telemetry = pd.read_csv(tel_path)
        labels = pd.read_csv(lab_path)
    except FileNotFoundError:
        print("Error: Files not found. Please ensure data/telemetry.csv and data/labels.csv exist.")
        return

    # Ensure alignment
    min_len = min(len(telemetry), len(labels))
    telemetry = telemetry.iloc[:min_len]
    labels = labels.iloc[:min_len]

    n_rows = len(telemetry)
    
    # [NEW] Create a column to store the TRUE CAUSE
    # Default is "None" (Normal operation)
    cause_labels = pd.Series(["None"] * n_rows, name="actual_cause")
    
    # --- SELECTION OF 20% FOR FAULTS ---
    n_anomalies = int(n_rows * 0.20)
    np.random.seed(42)  # Fixed seed for reproducibility
    anomaly_indices = np.random.choice(n_rows, n_anomalies, replace=False)
    
    print(f"Generating {n_anomalies} simulated faults (20% of total)...")

    # Divide indices among the 3 fault types
    idx_fan = anomaly_indices[:n_anomalies//3]
    idx_bearing = anomaly_indices[n_anomalies//3 : 2*n_anomalies//3]
    idx_filter = anomaly_indices[2*n_anomalies//3:]

    # --- 1. FanFault Injection ---
    # Symptom: High Spindle Temp
    current_temps = telemetry.loc[idx_fan, 'spindle_temp']
    new_temps = np.maximum(current_temps + 30, np.random.uniform(95, 115, size=len(idx_fan)))
    telemetry.loc[idx_fan, 'spindle_temp'] = np.round(new_temps, 2)
    
    # [CRITICAL] Save the cause label
    cause_labels.loc[idx_fan] = "FanFault" 
    
    # --- 2. BearingWearHigh Injection ---
    # Symptom: High Vibration + High Temp
    new_vibration = np.random.uniform(1.5, 3.5, size=len(idx_bearing))
    telemetry.loc[idx_bearing, 'vibration_rms'] = np.round(new_vibration, 3)
    
    current_temps = telemetry.loc[idx_bearing, 'spindle_temp']
    new_temps_bearing = np.maximum(current_temps + 15, np.random.uniform(90, 100, size=len(idx_bearing)))
    telemetry.loc[idx_bearing, 'spindle_temp'] = np.round(new_temps_bearing, 2)
    
    # [CRITICAL] Save the cause label
    cause_labels.loc[idx_bearing] = "BearingWearHigh"

    # --- 3. CloggedFilter Injection ---
    # Symptom: Low Coolant Flow + High Temp
    new_flow = np.random.uniform(0.1, 0.5, size=len(idx_filter))
    telemetry.loc[idx_filter, 'coolant_flow'] = np.round(new_flow, 3)
    
    current_temps = telemetry.loc[idx_filter, 'spindle_temp']
    new_temps_filter = np.maximum(current_temps + 20, np.random.uniform(92, 105, size=len(idx_filter)))
    telemetry.loc[idx_filter, 'spindle_temp'] = np.round(new_temps_filter, 2)
    
    # [CRITICAL] Save the cause label
    cause_labels.loc[idx_filter] = "CloggedFilter"

    # --- UPDATE OVERHEAT LABELS (Target) ---
    labels.loc[anomaly_indices, 'spindle_overheat'] = 1
    
    # --- SAVE ALL FILES ---
    if not os.path.exists('data'):
        os.makedirs('data')
        
    print("Saving modified files to 'data/'...")
    telemetry.to_csv("data/telemetry.csv", index=False)
    labels.to_csv("data/labels.csv", index=False)
    
    # [IMPORTANT] Save the new ground truth file
    cause_labels.to_csv("data/causes_ground_truth.csv", index=False)
    
    print("Done. 'causes_ground_truth.csv' created successfully.")
    print(f"Total faults injected: {labels['spindle_overheat'].sum()}")

if __name__ == "__main__":
    augment_data()