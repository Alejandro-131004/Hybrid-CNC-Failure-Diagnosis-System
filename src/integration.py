"""
integration.py
Pipeline connecting Bayesian Network (BN) inference with Knowledge Graph (KG)
and Decision Module for hybrid CNC fault diagnosis — with optional debug prints.
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
import pandas as pd
from .bn_model import integrate_latent_causes, fit_parameters_em


def run_demo(evidence: dict, debug=True, model_override=None):
    """
    Runs an integrated hybrid reasoning cycle:
    1. Bayesian inference → Overheat probability and top cause
    2. Knowledge Graph query → maintenance procedure(s)
    3. Decision computation → optimal recommended action
    """

    # Load configuration
    cfg = load_cfg()

    # Step 1: Initialize BN
    if model_override is None:
        model = build_manual_bn()
        model = add_manual_cpds(model)
    else:
        model = model_override

    if debug:
        print("=== [STEP 1] BAYESIAN NETWORK INITIALIZATION ===")
        print("[BN] Structure defined manually with causal relations and fixed CPDs.")

    # Step 2: BN inference
    if debug:
        print("\n=== [STEP 2] BAYESIAN INFERENCE ===")
        print(f"[Evidence] Provided sensor states: {evidence}")

    p_overheat = infer_overheat_prob(model, evidence)
    if debug:
        print(f"[Result] P(Overheat=1 | Evidence) = {p_overheat:.3f}")

    infer = make_inferencer(model)

    cause_nodes = ["BearingWearHigh", "FanFault", "CloggedFilter", "LowCoolingEfficiency"]
    cause_probs = {}

    for c in cause_nodes:
        try:
            q = infer.query(variables=[c], evidence=evidence)
            cause_probs[c] = float(q.values[1])
            if debug:
                print(f"[Cause] P({c}=1 | Evidence) = {cause_probs[c]:.3f}")
        except Exception:
            cause_probs[c] = 0.0

    top_cause = max(cause_probs, key=cause_probs.get) if cause_probs else None
    if debug:
        print(f"\n[BN] Top inferred root cause = {top_cause}")
        print("[Explanation] Node with highest posterior probability given the evidence.")

    # Step 3: Knowledge Graph reasoning
    if debug:
        print("\n=== [STEP 3] KNOWLEDGE GRAPH QUERIES ===")

    kg = CNCKG(debug=debug).load_from_cfg(cfg)
    if top_cause:
        procedures = kg.procedures_for_cause(top_cause)
        components = kg.components_for_cause(top_cause)
        symptoms = kg.symptoms_for_cause(top_cause)
    else:
        procedures, components, symptoms = [], [], []

    if debug:
        print(f"[KG] Found {len(procedures)} procedure(s), {len(components)} component(s), {len(symptoms)} symptom(s) for {top_cause}")
        print(f"[KG] Procedures: {procedures or 'none'}")
        print(f"[KG] Components: {components or 'none'}")
        print(f"[KG] Symptoms: {symptoms or 'none'}")

    # Step 4: Decision analysis
    if debug:
        print("\n=== [STEP 4] DECISION ANALYSIS (BAYES DECISION THEORY) ===")
        print("[Formula] Expected cost = Action cost + RiskWeight × P(Overheat) × FailurePenalty")

    procedures_csv = path_for(cfg, "procedures")
    action, costs = choose_action(p_overheat, top_cause, cfg, procedures_csv)

    if debug:
        for act, val in costs.items():
            print(f"[Decision] Expected cost({act}) = {val:.2f}")

        print(f"\n[Decision] Recommended action = {action}")
        print("[Reason] Action minimizing expected operational risk and cost.")

    # ===== Result Dictionary =====
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

    # ===== NORMAL MODE OUTPUT =====
    if not debug:
        cause = result["top_cause"]
        proc = result["procedures"][0] if result["procedures"] else "None"

        # obtain cost/time of the procedure
        dfp = pd.read_csv(path_for(cfg, "procedures"))
        row = dfp[dfp["name"] == proc]

        cost = float(row["spare_parts_cost_eur"].iloc[0]) if not row.empty else 0
        timeh = float(row["effort_h"].iloc[0]) if not row.empty else 0
        risk = int(row["risk_rating"].iloc[0]) if not row.empty else "unknown"

        print(
            f"\nBecause Vibration={evidence['Vibration']} and CoolantFlow={evidence['CoolantFlow']}, "
            f"the system estimates Overheat={result['p_overheat']:.2f}. "
            f"Likely cause is {cause}. Recommended action: {proc} ({timeh:.0f}h, {cost:.0f}€, risk={risk})."
        )

        return result

    # ===== DEBUG SUMMARY =====
    if debug:
        print("\n=== [SUMMARY] HYBRID REASONING RESULT ===")
        for k, v in result.items():
            print(f"{k}: {v}")

        print("\n[Done] End of hybrid reasoning cycle.\n")

    return result



def run_real(evidence: dict, debug=False):
    """
    Like run_demo, but CPDs are learned from telemetry CSV instead of manually defined.
    """
    from .bn_model import discretize, learn_structure, fit_parameters

    cfg = load_cfg()

    # Load real telemetry + labels
    tel = pd.read_csv(path_for(cfg, "telemetry"))
    lab = pd.read_csv(path_for(cfg, "labels"))
    df = tel.join(lab["spindle_overheat"])

    # Discretize sensors
    df_disc = discretize(df, cfg["bn"]["sensors"], n_bins=cfg["bn"]["discretize_bins"])

    
    # Learn BN structure based on real data
    model = learn_structure(df_disc)

    # Inject latent unobserved causes
    model = integrate_latent_causes(
        model,
        vib_node="vibration_rms",
        temp_node="spindle_temp",
        flow_node="coolant_flow",
    )

    # Fit parameters using EM because latent causes have no direct labels
    model = fit_parameters_em(model, df_disc, max_iter=50)

    # Reuse the same reasoning pipeline with learned model
    return run_demo(evidence, debug=debug, model_override=model)
