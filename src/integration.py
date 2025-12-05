"""
integration.py
Pipeline connecting Bayesian Network (BN) inference with Knowledge Graph (KG)
and Decision Module for hybrid CNC fault diagnosis — corrected version.
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

from .evaluation import evaluate_bn
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from .bn_model import fit_parameters_em, integrate_latent_causes, discretize

###############################################################
#  RUN DEMO (works for both manual and real models)
###############################################################

def run_demo(evidence: dict, debug=True, model_override=None):
    cfg = load_cfg()

    # ------------------------------------------------------------
    # MODEL SELECTION: MANUAL OR REAL
    # ------------------------------------------------------------
    if model_override is None:
        model = build_manual_bn()
        model = add_manual_cpds(model)
        if debug:
            print("[BN] Using manually defined CPDs.")
    else:
        model = model_override
        if debug:
            print("[BN] Using REAL trained model for inference.")

    # ------------------------------------------------------------
    # STEP 2 — BAYESIAN INFERENCE
    # ------------------------------------------------------------
    if debug:
        print("\n=== [STEP 2] BAYESIAN INFERENCE ===")
        # Note: 'evidence' here is the discrete evidence (0/1)
        print(f"[Evidence] Provided (Discrete): {evidence}") 

    # 1. Infer P(Overheat)
    p_overheat = infer_overheat_prob(model, evidence)

    if debug:
        print(f"→ P(Overheat=1 | evidence) = {p_overheat:.3f}")

    infer = make_inferencer(model)

    # 2. Infer latent causes
    cause_nodes = ["BearingWearHigh", "FanFault", "CloggedFilter", "LowCoolingEfficiency"]
    cause_sensor_map = {
        "BearingWearHigh": ["vibration_rms"],
        "FanFault": ["spindle_temp"],
        "CloggedFilter": ["coolant_flow"],
        "LowCoolingEfficiency": ["spindle_temp"],
    }
    cause_probs = {}
    top_cause = "UnknownCause"

    if debug:
        print("\n--- Inferring Latent Causes (Simplified Evidence) ---")

    for c in cause_nodes:
        # Create minimal evidence dictionary for the specific cause query
        minimal_evidence = {
            s: evidence[s]
            for s in cause_sensor_map.get(c, [])
            if s in evidence
        }

        try:
            q = infer.query(variables=[c], evidence=minimal_evidence)
            cause_probs[c] = float(q.values[1])
        except Exception as e:
            if debug:
                print(f"[ERROR] Inference failed for {c} with evidence {minimal_evidence}: {e}")
            cause_probs[c] = 0.0

        if debug:
            print(f"→ P({c}=1 | evidence) = {cause_probs[c]:.3f}")

    # Ensure top_cause is selected robustly
    if cause_probs and any(p > 0.0 for p in cause_probs.values()):
        top_cause = max(cause_probs, key=cause_probs.get)
    elif cause_nodes:
        # Fallback to the cause with the highest *prior* probability (BearingWearHigh is often chosen by default here)
        top_cause = max(cause_nodes, key=lambda k: model.get_cpds(k).values[1] if model.get_cpds(k) else 0)

    # ------------------------------------------------------------
    # STEP 3 — KNOWLEDGE GRAPH QUERIES
    # ------------------------------------------------------------
    if debug:
        print("\n=== [STEP 3] KNOWLEDGE GRAPH QUERIES ===")

    kg = CNCKG(debug=debug).load_from_cfg(cfg)

    symptoms = kg.symptoms_for_cause(top_cause)
    components = kg.components_for_cause(top_cause)
    procedures = kg.procedures_for_cause(top_cause)

    # ------------------------------------------------------------
    # STEP 4 — DECISION THEORY
    # ------------------------------------------------------------
    procedures_csv = path_for(cfg, "procedures")

    action, costs = choose_action(
        p_overheat=p_overheat,
        top_cause=top_cause,
        cfg=cfg,
        procedures_csv=procedures_csv,
        cause_probs=cause_probs
    )

    # ------------------------------------------------------------
    # RESULT
    # ------------------------------------------------------------
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

    # Pretty explanation when debug=False
    if not debug:
        vib = evidence.get("vibration_rms")
        flow = evidence.get("coolant_flow")

        vib_state = "high" if str(vib) == "1" else "normal"
        flow_state = "low" if str(flow) == "1" else "normal"

        proc = procedures[0] if procedures else "None"
        dfp = pd.read_csv(path_for(cfg, "procedures"))
        row = dfp[dfp["name"] == proc]

        cost = float(row["spare_parts_cost_eur"].iloc[0]) if not row.empty else 0
        timeh = float(row["effort_h"].iloc[0]) if not row.empty else 0
        risk = int(row["risk_rating"].iloc[0]) if not row.empty else 0

        print(
            f"\nBecause vibration is {vib_state} and coolant flow is {flow_state}, "
            f"the system estimates Overheat={result['p_overheat']:.2f}. "
            f"Likely cause is {top_cause}. "
            f"Recommended action: {proc} ({timeh:.0f}h, {cost:.0f}€, risk={risk})."
        )

    return result


###############################################################
#  REAL MODE TRAINING WITH FIXED CAUSAL STRUCTURE + EM
###############################################################

def run_real(evidence: dict, debug=False, force_retrain=False, return_test_data=False):
    from sklearn.model_selection import train_test_split
    import os, pickle

    cfg = load_cfg()
    model_cache_path = cfg["bn"].get("model_cache_path", "models/trained_model.pkl")

    # ------------------------------------------------------------
    # FAST PATH: Load already trained model
    # ------------------------------------------------------------
    if not force_retrain and os.path.exists(model_cache_path):
        model = pickle.load(open(model_cache_path, "rb"))
        if debug:
            print("[BN] Loaded cached REAL model.")

        # Load discretizer
        discretizer = None
        if os.path.exists("models/discretizer.pkl"):
            # Correct opening and assignment
            with open("models/discretizer.pkl", "rb") as f: 
                discretizer = pickle.load(f)

        # Return test set if requested
        if return_test_data:
            tel = pd.read_csv(path_for(cfg, "telemetry"))
            lab = pd.read_csv(path_for(cfg, "labels"))
            df = tel.join(lab["spindle_overheat"])

            df_disc = discretize(df, cfg["bn"]["sensors"], n_bins=cfg["bn"]["discretize_bins"])
            _, test_data = train_test_split(
                df_disc, test_size=cfg["bn"]["test_size"], random_state=42
            )
            return model, test_data

        return run_demo(evidence, debug=debug, model_override=model)

    # ------------------------------------------------------------
    # SLOW PATH: FULL TRAINING
    # ------------------------------------------------------------
    if debug:
        print("\n[BN] TRAINING REAL MODEL...\n")

    tel = pd.read_csv(path_for(cfg, "telemetry"))
    lab = pd.read_csv(path_for(cfg, "labels"))
    df = tel.join(lab["spindle_overheat"])

    # Discretize + save discretizer
    df_disc, discretizer = discretize(
        df, cfg["bn"]["sensors"], n_bins=cfg["bn"]["discretize_bins"], return_discretizer=True
    )

    os.makedirs("models", exist_ok=True)
    pickle.dump(discretizer, open("models/discretizer.pkl", "wb"))

    # Split
    df_train, df_test = train_test_split(
        df_disc, test_size=cfg["bn"]["test_size"], random_state=42
    )
    if debug:
        print(f"[Split] Train={len(df_train)}, Test={len(df_test)}")

    # ------------------------------------------------------------
    # FIXED, PHYSICALLY CORRECT STRUCTURE (matches DEMO + PDF)
    # ------------------------------------------------------------
    model = BayesianNetwork([
        ('BearingWearHigh', 'vibration_rms'),
        ('FanFault', 'spindle_temp'),
        ('CloggedFilter', 'coolant_flow'),
        ('LowCoolingEfficiency', 'spindle_temp'),
        ('vibration_rms', 'spindle_overheat'),
        ('spindle_temp', 'spindle_overheat'),
        ('coolant_flow', 'spindle_overheat'),
    ])

    # Add latent cause logic
    model = integrate_latent_causes(model)

    # ------------------------------------------------------------
    # INITIAL CPDs (neutral)
    # ------------------------------------------------------------
    initial_cpds = [
        TabularCPD("BearingWearHigh", 2, [[0.5], [0.5]]),
        TabularCPD("FanFault", 2, [[0.5], [0.5]]),
        TabularCPD("CloggedFilter", 2, [[0.5], [0.5]]),
        TabularCPD("LowCoolingEfficiency", 2, [[0.5], [0.5]]),
    ]
    for cpd in initial_cpds:
        try:
            model.add_cpds(cpd)
        except:
            pass

    # ------------------------------------------------------------
    # EM PARAMETER LEARNING
    # ------------------------------------------------------------
    if debug:
        print("[BN] Learning CPDs with EM...")

    model = fit_parameters_em(
        model,
        df_train,
        max_iter=cfg["bn"]["max_iter"],
        n_jobs=cfg["bn"]["n_jobs"],
    )

    model.check_model()

    # Save model
    pickle.dump(model, open(model_cache_path, "wb"))
    if debug:
        print("[BN] Model saved.")

    if return_test_data:
        return model, df_test

    # Otherwise perform reasoning
    return run_demo(evidence, debug=debug, model_override=model)
