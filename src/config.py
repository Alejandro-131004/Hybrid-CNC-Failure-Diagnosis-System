import yaml
import os

# ---------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------

def load_cfg(path: str = "configs/settings.yaml"):
    """
    Load and return the YAML configuration as a Python dictionary.
    The path defaults to 'settings.yaml' at the project root.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


# ---------------------------------------------------------
# Utility to build file paths from config
# ---------------------------------------------------------

def path_for(cfg: dict, key: str) -> str:
    """
    Given the loaded config and a key inside cfg["files"],
    build the full path under cfg["data_dir"].

    Example:
        telemetry_path = path_for(cfg, "telemetry")
    """
    data_dir = cfg.get("data_dir", "")
    files = cfg.get("files", {})

    if key not in files:
        raise KeyError(f"File key '{key}' not found in configuration.")

    return os.path.join(data_dir, files[key])
