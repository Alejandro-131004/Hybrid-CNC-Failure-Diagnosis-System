"""
decision.py
Implements Bayes Decision Theory for maintenance recommendations.
"""

import pandas as pd


# ============================================================== 
# === Helper: compute expected cost and risk over causes
# ==============================================================

# src/decision.py

def _compute_expected_cause_costs(cause_probs: dict[str, float] | None,
                                  procedures_df: pd.DataFrame,
                                  top_cause: str | None):
    """
    Aggregates, over all causes, the expected:
      - spare parts cost
      - risk severity
    using the posterior probabilities of each cause.
    If cause_probs is None, falls back to a degenerate distribution on top_cause.
    """
    if cause_probs is None or len(cause_probs) == 0:
        cause_probs = {}
        if top_cause is not None:
            cause_probs[top_cause] = 1.0

    expected_proc_cost = 0.0
    expected_risk_severity = 0.0

    # Safeguard: check expected column names
    cost_col = "spare_parts_cost_eur"
    risk_col = "risk_rating"

    # --- CORRECTION START: Normalize Probabilities ---
    # We must normalize the probabilities to sum to 1.0 relative to the faults.
    # Otherwise, if we use low priors (e.g. 1%), the sum of probabilities is tiny (~0.04),
    # resulting in a negligible risk penalty (Cost * 0.04), leading to "Continue" erroneously.
    total_p = sum(cause_probs.values())
    
    if total_p > 1e-9: # Avoid division by zero
        # Use normalized weights to calculate the Weighted Average Risk/Cost
        norm_probs = {k: v / total_p for k, v in cause_probs.items()}
    else:
        norm_probs = cause_probs
    # --- CORRECTION END ---

    for cause, p_c in norm_probs.items():
        if p_c <= 0:
            continue

        row = procedures_df[procedures_df["mitigates_cause"] == cause]
        if row.empty:
            continue

        cost_c = float(row[cost_col].iloc[0])
        risk_c = float(row[risk_col].iloc[0])

        expected_proc_cost += p_c * cost_c
        expected_risk_severity += p_c * risk_c

    return expected_proc_cost, expected_risk_severity


# ============================================================== 
# === Expected cost computation (Bayes Decision Theory)
# ==============================================================

def expected_cost(action: str,
                  p_overheat: float,
                  expected_proc_cost: float,
                  expected_risk_severity: float,
                  cfg) -> float:
    """
    Computes expected cost of an action given:
      - P(Overheat = 1 | evidence)
      - distribution over causes mapped to:
          * expected_proc_cost (from procedures.csv)
          * expected_risk_severity (risk_rating, severity of failure)
    """
    rw = cfg["decision"]["risk_weight"]
    cost_params = cfg["decision"]["cost"]

    # Base penalty factor (interpretable no relatório)
    base_failure_penalty = 500.0

    # Expected failure penalty weighted by cause severity
    failure_penalty = base_failure_penalty * expected_risk_severity

    if action == "Continue":
        # No immediate cost, but full failure risk
        return cost_params.get("Continue", 0.0) + rw * p_overheat * failure_penalty

    if action == "SlowDown":
        # Operational cost + reduced failure risk (e.g. 50%)
        slowdown_factor = 0.5
        return (
            cost_params["SlowDown_per_hour"]
            + rw * p_overheat * failure_penalty * slowdown_factor
        )

    if action == "ScheduleMaintenance":
        # Immediate maintenance base cost + expected procedure cost
        # + residual small failure risk (e.g. 10%)
        residual_factor = 0.1
        return (
            cost_params["ScheduleMaintenance_base"]
            + expected_proc_cost
            + rw * p_overheat * failure_penalty * residual_factor
        )

    raise ValueError(f"Unknown action: {action}")


# ============================================================== 
# === Decision selection logic
# ==============================================================

def choose_action(p_overheat: float,
                  top_cause: str | None,
                  cfg,
                  procedures_csv: str,
                  cause_probs: dict[str, float] | None = None):
    """
    Chooses the optimal action minimizing expected cost:
      - Continue
      - SlowDown
      - ScheduleMaintenance

    Now explicitly depends on the posterior distribution over causes:
      P(cause | evidence) and their costs/risks from procedures.csv.
    """
    try:
        df = pd.read_csv(procedures_csv)
    except Exception:
        # If we can't read procedures, fall back to generic values
        df = pd.DataFrame(columns=["mitigates_cause", "spare_parts_cost_eur", "risk_rating"])

    # Compute expectation over causes: cost and risk
    expected_proc_cost, expected_risk_severity = _compute_expected_cause_costs(
        cause_probs=cause_probs,
        procedures_df=df,
        top_cause=top_cause,
    )

    # Compute expected cost for each possible action
    options = {
        "Continue": expected_cost("Continue", p_overheat, expected_proc_cost, expected_risk_severity, cfg),
        "SlowDown": expected_cost("SlowDown", p_overheat, expected_proc_cost, expected_risk_severity, cfg),
        "ScheduleMaintenance": expected_cost("ScheduleMaintenance", p_overheat, expected_proc_cost, expected_risk_severity, cfg),
    }

    # Choose action with minimum expected cost
    best_action = min(options, key=options.get)
    return best_action, options
