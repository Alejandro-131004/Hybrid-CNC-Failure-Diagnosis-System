import os, yaml, pandas as pd

def load_cfg(path="configs/settings.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def path_for(cfg, key):
    return os.path.join(cfg["data_dir"], cfg["files"][key])

def load_csv(cfg, key, parse_dates=None):
    p = path_for(cfg, key)
    return pd.read_csv(p, parse_dates=parse_dates)
