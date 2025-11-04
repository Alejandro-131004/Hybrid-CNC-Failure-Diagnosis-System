"""
decision.py
Implements Bayes Decision Theory for maintenance recommendations.
"""

import pandas as pd
from .utils import load_csv


# ==============================================================
# === Expected cost computation (Bayes Decision Theory)
# ==============================================================

def expected_cost(action: str, p_overheat: float, proc_cost: float | None, cfg) -> float:
    """
    Computes expected cost of an action given the probability of overheat.
    """
    rw = cfg["decision"]["risk_weight"]
    cost_params = cfg["decision"]["cost"]

    if action == "Continue":
        penalty = 500  # assumed equipment damage if overheat occurs
        return 0 + rw * p_overheat * penalty

    if action == "SlowDown":
        return cost_params["SlowDown_per_hour"] + rw * p_overheat * 200

    if action == "ScheduleMaintenance":
        base = cost_params["ScheduleMaintenance_base"]
        return base + (proc_cost or 0) + rw * p_overheat * 50

    raise ValueError(f"Unknown action: {action}")


# ==============================================================
# === Decision selection logic
# ==============================================================

def choose_action(p_overheat: float, top_cause: str | None, cfg, procedures_csv: str):
    """
    Chooses the optimal action minimizing expected cost:
      - Continue
      - SlowDown
      - ScheduleMaintenance
    """
    proc_cost = None
    try:
        df = pd.read_csv(procedures_csv)
        if top_cause and "mitigates_cause" in df.columns and "cost" in df.columns:
            row = df[df["mitigates_cause"] == top_cause]
            if not row.empty:
                proc_cost = float(row["cost"].iloc[0])
    except Exception:
        pass

    # Compute expected cost for each possible action
    options = {
        "Continue": expected_cost("Continue", p_overheat, proc_cost, cfg),
        "SlowDown": expected_cost("SlowDown", p_overheat, proc_cost, cfg),
        "ScheduleMaintenance": expected_cost("ScheduleMaintenance", p_overheat, proc_cost, cfg),
    }

    # Choose action with minimum expected cost
    best_action = min(options, key=options.get)
    return best_action, options
