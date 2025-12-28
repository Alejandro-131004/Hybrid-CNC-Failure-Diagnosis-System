import pandas as pd
import numpy as np
import os

def augment_data():
    # File paths
    tel_path = "data/telemetry.csv"
    lab_path = "data/labels.csv"
    
    print(f"Loading data from: {tel_path}...")
    
    try:
        telemetry = pd.read_csv(tel_path)
        labels = pd.read_csv(lab_path)
    except FileNotFoundError:
        print("Error: Files not found.")
        return

    # Ensure alignment
    min_len = min(len(telemetry), len(labels))
    telemetry = telemetry.iloc[:min_len]
    labels = labels.iloc[:min_len]

    n_rows = len(telemetry)
    
    # 1. GROUND TRUTH (Causas Reais)
    cause_labels = pd.Series(["None"] * n_rows, name="actual_cause")
    
    # --- SELEÇÃO DE 30% PARA FALHAS (Aumentei um pouco para compensar o filtro depois) ---
    n_anomalies = int(n_rows * 0.30)
    np.random.seed(42)
    anomaly_indices = np.random.choice(n_rows, n_anomalies, replace=False)
    
    print(f"Injecting faults in {n_anomalies} samples...")

    idx_fan = anomaly_indices[:n_anomalies//3]
    idx_bearing = anomaly_indices[n_anomalies//3 : 2*n_anomalies//3]
    idx_filter = anomaly_indices[2*n_anomalies//3:]

    # --- INJEÇÃO DE VALORES NOS SENSORES ---
    
    # FanFault -> Temperatura Alta
    current_temps = telemetry.loc[idx_fan, 'spindle_temp']
    new_temps = np.maximum(current_temps + 30, np.random.uniform(95, 115, size=len(idx_fan)))
    telemetry.loc[idx_fan, 'spindle_temp'] = np.round(new_temps, 2)
    cause_labels.loc[idx_fan] = "FanFault" 
    
    # BearingWearHigh -> Vibração Alta + Temp Alta
    # (Ajustei vibração para > 1.5 para bater com a regra da prof)
    new_vibration = np.random.uniform(1.6, 3.5, size=len(idx_bearing))
    telemetry.loc[idx_bearing, 'vibration_rms'] = np.round(new_vibration, 3)
    
    current_temps = telemetry.loc[idx_bearing, 'spindle_temp']
    new_temps_bearing = np.maximum(current_temps + 15, np.random.uniform(90, 100, size=len(idx_bearing)))
    telemetry.loc[idx_bearing, 'spindle_temp'] = np.round(new_temps_bearing, 2)
    cause_labels.loc[idx_bearing] = "BearingWearHigh"

    # CloggedFilter -> Fluxo Baixo + Temp Alta
    # (Ajustei fluxo para < 0.6 para bater com a regra da prof)
    new_flow = np.random.uniform(0.1, 0.5, size=len(idx_filter))
    telemetry.loc[idx_filter, 'coolant_flow'] = np.round(new_flow, 3)
    
    current_temps = telemetry.loc[idx_filter, 'spindle_temp']
    new_temps_filter = np.maximum(current_temps + 20, np.random.uniform(92, 105, size=len(idx_filter)))
    telemetry.loc[idx_filter, 'spindle_temp'] = np.round(new_temps_filter, 2)
    cause_labels.loc[idx_filter] = "CloggedFilter"

    # --- 2. LÓGICA PROBABILÍSTICA DA PROFESSORA ---
    # Aqui decidimos se a falha realmente causou "Overheat" (Target = 1)
    
    # Condição Crítica (Baseada no código que ela deu)
    cond = (
        (telemetry["spindle_temp"] > 87) |
        ((telemetry["vibration_rms"] > 1.5) & (telemetry["coolant_flow"] < 0.6))
    )

    # Reset labels to 0 first
    labels["spindle_overheat"] = 0
    
    # Aplica a máscara aleatória (50% de chance se a condição for cumprida)
    # Isto quebra a correlação perfeita de 100%
    mask = cond & (np.random.rand(len(telemetry)) < 0.5)
    
    labels.loc[mask, "spindle_overheat"] = 1
    
    # --- SAVE ALL FILES ---
    if not os.path.exists('data'):
        os.makedirs('data')
        
    print("Saving modified files to 'data/'...")
    telemetry.to_csv("data/telemetry.csv", index=False)
    # Nota: Salvo como labels.csv porque é o que o sistema espera
    labels.to_csv("data/labels.csv", index=False)
    cause_labels.to_csv("data/causes_ground_truth.csv", index=False)
    
    print("Done.")
    print(f"Total Overheat Positives: {labels['spindle_overheat'].sum()} ({labels['spindle_overheat'].mean()*100:.2f}%)")

if __name__ == "__main__":
    augment_data()