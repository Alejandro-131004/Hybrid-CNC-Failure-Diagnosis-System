import pandas as pd
import numpy as np
import os

def augment_data():
    # Caminhos dos ficheiros
    tel_path = "data/telemetry.csv"
    lab_path = "data/labels.csv"
    
    print(f"A carregar dados de: {tel_path}...")
    
    # Carregar os dados
    try:
        telemetry = pd.read_csv(tel_path)
        labels = pd.read_csv(lab_path)
    except FileNotFoundError:
        print("Erro: Ficheiros não encontrados. Verifique se estão na pasta correta.")
        return

    # Garantir alinhamento
    min_len = min(len(telemetry), len(labels))
    telemetry = telemetry.iloc[:min_len]
    labels = labels.iloc[:min_len]

    n_rows = len(telemetry)
    
    # --- SELEÇÃO DE 20% DOS DADOS PARA FALHAS ---
    n_anomalies = int(n_rows * 0.20)
    anomaly_indices = np.random.choice(n_rows, n_anomalies, replace=False)
    
    print(f"A gerar {n_anomalies} falhas simuladas (20% do total)...")
    print("A calibrar valores para exceder os limites normais (>90ºC)...")

    # --- INJEÇÃO DE FALHAS REALISTAS (Com Arredondamento) ---
    
    # 1. Falha na Ventoinha (FanFault)
    # Sintoma: Temperatura sobe drasticamente (>95ºC)
    idx_fan = anomaly_indices[:n_anomalies//3]
    current_temps = telemetry.loc[idx_fan, 'spindle_temp']
    
    new_temps = np.maximum(current_temps + 30, np.random.uniform(95, 115, size=len(idx_fan)))
    telemetry.loc[idx_fan, 'spindle_temp'] = np.round(new_temps, 2) # Arredondar a 2 casas
    
    # 2. Desgaste do Rolamento (BearingWear)
    # Sintoma: Vibração muito alta + Temperatura alta
    idx_bearing = anomaly_indices[n_anomalies//3 : 2*n_anomalies//3]
    
    # Aumentar vibração
    new_vibration = np.random.uniform(1.5, 3.5, size=len(idx_bearing))
    telemetry.loc[idx_bearing, 'vibration_rms'] = np.round(new_vibration, 3) # Arredondar a 3 casas
    
    # Aumentar temperatura (atrito)
    current_temps = telemetry.loc[idx_bearing, 'spindle_temp']
    new_temps_bearing = np.maximum(current_temps + 15, np.random.uniform(90, 100, size=len(idx_bearing)))
    telemetry.loc[idx_bearing, 'spindle_temp'] = np.round(new_temps_bearing, 2) # Arredondar a 2 casas

    # 3. Filtro Entupido (CloggedFilter)
    # Sintoma: Fluxo de refrigerante baixo + Temperatura alta
    idx_filter = anomaly_indices[2*n_anomalies//3:]
    
    # Reduzir fluxo
    new_flow = np.random.uniform(0.1, 0.5, size=len(idx_filter))
    telemetry.loc[idx_filter, 'coolant_flow'] = np.round(new_flow, 3) # Arredondar a 3 casas
    
    # Aumentar temperatura
    current_temps = telemetry.loc[idx_filter, 'spindle_temp']
    new_temps_filter = np.maximum(current_temps + 20, np.random.uniform(92, 105, size=len(idx_filter)))
    telemetry.loc[idx_filter, 'spindle_temp'] = np.round(new_temps_filter, 2) # Arredondar a 2 casas

    # --- ATUALIZAR AS LABELS ---
    labels.loc[anomaly_indices, 'spindle_overheat'] = 1
    
    # --- GUARDAR NOVOS FICHEIROS ---
    # Cria pasta data se não existir
    if not os.path.exists('data'):
        os.makedirs('data')
        
    new_tel_path = "data/telemetry.csv"
    new_lab_path = "data/labels.csv"

    print(f"A guardar ficheiros modificados na pasta 'data/'...")
    telemetry.to_csv(new_tel_path, index=False)
    labels.to_csv(new_lab_path, index=False)
    
    print("Concluído.")
    print(f"Total de falhas injetadas: {labels['spindle_overheat'].sum()}")

if __name__ == "__main__":
    augment_data()