"""
integration.py
Pipeline connecting Bayesian Network (BN) inference with Knowledge Graph (KG)
and Decision Module for hybrid CNC fault diagnosis — with detailed debug prints.
"""

from .utils import load_cfg, path_for
from .bn_model import (
    build_manual_bn,
    add_manual_cpds,
    infer_overheat_prob,
    make_inferencer,
)
from .kg_module import CNCKG
from .decision import choose_action


def run_demo(evidence: dict):
    """
    Runs an integrated hybrid reasoning cycle:
    1. Bayesian inference → Overheat probability and top cause
    2. Knowledge Graph query → maintenance procedure(s)
    3. Decision computation → optimal recommended action
    """

    print("=== [STEP 1] LOADING CONFIGURATION ===")
    cfg = load_cfg()

    # === Step 1. Bayesian Network inference ===
    print("\n=== [STEP 2] INITIALIZING BAYESIAN NETWORK ===")
    model = build_manual_bn()
    model = add_manual_cpds(model)
    print("[BN] Structure defined manually with causal relations and fixed CPDs.")

    # Overheat inference
    print("\n=== [STEP 3] BAYESIAN INFERENCE ===")
    print(f"[Evidence] Provided sensor states: {evidence}")
    p_overheat = infer_overheat_prob(model, evidence)
    print(f"[Result] P(Overheat=1 | Evidence) = {p_overheat:.3f}")

    # Cause probabilities
    infer = make_inferencer(model)
    cause_nodes = ["BearingWear", "FanFault", "CloggedFilter", "LowCoolingEfficiency"]
    cause_probs = {}
    for c in cause_nodes:
        try:
            q = infer.query(variables=[c], evidence=evidence)
            cause_probs[c] = float(q.values[1])
            print(f"[Cause] P({c}=1 | Evidence) = {cause_probs[c]:.3f}")
        except Exception as e:
            print(f"[WARN] Could not infer {c}: {e}")

    top_cause = max(cause_probs, key=cause_probs.get) if cause_probs else None
    print(f"\n[BN] Top inferred root cause = {top_cause}")
    print("[Explanation] This is the node with the highest posterior probability given the evidence.")

    # === Step 2. Knowledge Graph reasoning ===
    print("\n=== [STEP 4] KNOWLEDGE GRAPH QUERIES ===")
    kg = CNCKG().load_from_cfg(cfg)
    if top_cause:
        procedures = kg.procedures_for_cause(top_cause)
        components = kg.components_for_cause(top_cause)
        symptoms = kg.symptoms_for_cause(top_cause)
    else:
        procedures, components, symptoms = [], [], []

    print(f"[KG] Found {len(procedures)} procedure(s), {len(components)} component(s), {len(symptoms)} symptom(s) for {top_cause}")
    print(f"[KG] Procedures: {procedures or 'none'}")
    print(f"[KG] Components: {components or 'none'}")
    print(f"[KG] Symptoms: {symptoms or 'none'}")

    # === Step 3. Decision analysis ===
    print("\n=== [STEP 5] DECISION ANALYSIS (BAYES DECISION THEORY) ===")
    print("[Formula] Expected cost = Action cost + RiskWeight × P(Overheat) × FailurePenalty")

    procedures_csv = path_for(cfg, "procedures")
    action, costs = choose_action(p_overheat, top_cause, cfg, procedures_csv)

    for act, val in costs.items():
        print(f"[Decision] Expected cost({act}) = {val:.2f}")

    print(f"\n[Decision] Recommended action = {action}")
    print("[Reason] The selected action minimizes the expected cost given the inferred failure probability.")

    # === Output summary ===
    result = {
        "p_overheat": round(p_overheat, 3),
        "top_cause": top_cause,
        "probabilities": cause_probs,
        "components": components,
        "symptoms": symptoms,
        "procedures": procedures,
        "recommended_action": action,
        "expected_costs": costs,
    }

    print("\n=== [SUMMARY] HYBRID REASONING RESULT ===")
    for k, v in result.items():
        print(f"{k}: {v}")

    print("\n[Done] End of hybrid reasoning cycle.\n")
    return result
