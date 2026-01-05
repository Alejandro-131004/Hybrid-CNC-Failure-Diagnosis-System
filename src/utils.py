"""
utils.py
Shared utilities for configuration and data loading.
"""
import yaml
import os
import pandas as pd

def load_cfg(path=None):
    """
    Load YAML configuration. 
    Searches in multiple standard locations if path is not provided.
    """
    # Se o utilizador der um caminho específico, tenta esse primeiro
    candidates = []
    if path:
        candidates.append(path)

    # Lista de locais onde o settings.yaml costuma estar
    candidates.extend([
        "settings.yaml",              # Na raiz onde corres o main.py
        "configs/settings.yaml",      # Numa pasta configs
        "../settings.yaml",           # Um nível acima (se correr de src/)
        "../configs/settings.yaml",   # Um nível acima dentro de configs
        "src/settings.yaml"           # Dentro de src (raro, mas possível)
    ])

    for candidate in candidates:
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    # print(f"[DEBUG] Loaded config from: {candidate}") # Descomenta se quiseres ver
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"[WARNING] Found {candidate} but could not read it: {e}")

    # Se chegar aqui, não encontrou nada
    raise FileNotFoundError(
        f"Configuration file 'settings.yaml' not found.\n"
        f"Checked locations: {candidates}\n"
        f"Current working directory: {os.getcwd()}"
    )

def path_for(cfg, key):
    """Builds file path based on config data_dir."""
    data_dir = cfg.get("data_dir", "data")
    
    if "files" not in cfg or key not in cfg["files"]:
         raise KeyError(f"Key '{key}' missing in settings.yaml under 'files:' section.")
         
    filename = cfg["files"][key]
    
    # 1. Tenta o caminho direto (ex: data/telemetry.csv)
    path_direct = os.path.join(data_dir, filename)
    if os.path.exists(path_direct):
        return path_direct
        
    # 2. Tenta um nível acima (ex: ../data/telemetry.csv)
    path_up = os.path.join("..", data_dir, filename)
    if os.path.exists(path_up):
        return path_up

    # 3. Retorna o direto por defeito (para o erro ser informativo se falhar depois)
    return path_direct

def load_csv(cfg, key, parse_dates=None):
    p = path_for(cfg, key)
    try:
        return pd.read_csv(p, parse_dates=parse_dates)
    except FileNotFoundError:
        # Tenta dar uma dica útil
        raise FileNotFoundError(f"Could not find CSV for '{key}'. Expected at: {p}")