import pandas as pd
from src.bn_model import load_telemetry, discretize, learn_structure, fit_parameters, make_inferencer, TARGET
from src.kg_module import build_graph, procedures_for_cause
from src.decision import expected_cost_for_action, choose_action

def load_miss_cost(maintenance_csv: str) -> dict:
    """Lê custos de falha do ficheiro maintenance.csv."""
    df = pd.read_csv(maintenance_csv)
    if "cause" not in df.columns or "failure_cost" not in df.columns:
        raise ValueError("maintenance.csv deve conter colunas 'cause' e 'failure_cost'.")
    return dict(zip(df["cause"], df["failure_cost"]))

def top_causes(posterior: dict[str,float], k: int = 3) -> dict[str,float]:
    return dict(sorted(posterior.items(), key=lambda x: x[1], reverse=True)[:k])

def run_pipeline(paths: dict, continuous_cols: list[str], evidence: dict[str,int]):
    miss_cost = load_miss_cost(paths["maintenance"])

    # === Bayesian Network ===
    df = load_telemetry(paths["telemetry"], paths["labels"])
    dfd = discretize(df, continuous_cols)
    model = fit_parameters(learn_structure(dfd), dfd)
    inf = make_inferencer(model)

    causes = pd.read_csv(paths["causes"])["name"].tolist()
    posterior = {}
    for c in causes:
        try:
            q = inf.query(variables=[c], evidence=evidence)
            valmap = dict(enumerate(q.values))
            posterior[c] = valmap.get(1, max(valmap.values()))
        except Exception:
            continue
    p_top = top_causes(posterior)

    # === Knowledge Graph ===
    g = build_graph(paths["components"], paths["causes"], paths["symptoms"], paths["procedures"], paths["relations"])

    action_map = {c: procedures_for_cause(g, c) for c in p_top}

    # === Decision logic ===
    ec = expected_cost_for_action(p_top, action_map, miss_cost)
    act, ec_val = choose_action(ec)

    return {
        "posterior": p_top,
        "actions": action_map,
        "expected_costs": ec,
        "recommendation": {"action": act, "expected_cost": ec_val}
    }
