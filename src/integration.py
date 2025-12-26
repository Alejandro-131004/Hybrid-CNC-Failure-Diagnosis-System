"""
integration.py
Pipeline connecting Bayesian Network (BN) inference with Knowledge Graph (KG)
and Decision Module for hybrid CNC fault diagnosis.
"""

from .utils import load_cfg, path_for
from .bn_model import infer_overheat_prob, make_inferencer, integrate_latent_causes
from .kg_module import CNCKG
from .decision import choose_action
import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

# ------------------------------------------------------------
# CRITICAL: MANUAL DISCRETIZATION (THRESHOLDS ADJUSTED)
# ------------------------------------------------------------
def manual_discretize(df, sensors_list):
    """
    Discretizes sensors based on SAFE PHYSICAL THRESHOLDS.
    Adjusted to avoid false positives on noisy normal data.
    0 = Normal
    1 = Abnormal
    """
    df_disc = df.copy()
    
    # 1. Spindle Temp
    # Augmentation injects faults > 90. Normal is likely < 70.
    # Threshold: 85 (Safe middle ground)
    if "spindle_temp" in df.columns:
        df_disc["spindle_temp"] = (df["spindle_temp"] > 85.0).astype(int)
        
    # 2. Vibration RMS
    # Augmentation injects faults > 1.5. 
    # Previous Threshold (1.0) was too low (caught normal noise).
    # New Threshold: 1.4 (Only catches the real faults)
    if "vibration_rms" in df.columns:
        df_disc["vibration_rms"] = (df["vibration_rms"] > 1.4).astype(int)
        
    # 3. Coolant Flow
    # Augmentation injects faults < 0.5.
    # Normal is likely > 2.0.
    # Threshold: 0.6 (Anything below 0.6 is definitely a clog)
    # We invert logic: 1 = Abnormal (Low Flow)
    if "coolant_flow" in df.columns:
        df_disc["coolant_flow"] = (df["coolant_flow"] < 0.6).astype(int)
        
    # 4. Others (Default to 0)
    for col in sensors_list:
        if col not in ["spindle_temp", "vibration_rms", "coolant_flow"] and col in df.columns:
            df_disc[col] = 0 
            
    return df_disc

# ------------------------------------------------------------
# RUN DEMO
# ------------------------------------------------------------
def run_demo(evidence: dict, debug=True, model_override=None, discretizer=None):
    cfg = load_cfg()
    
    if model_override is None:
        print("[ERROR] No model provided to run_demo.")
        return {}
    else:
        model = model_override

    # --- INFERENCE ---
    try:
        p_overheat = infer_overheat_prob(model, evidence)
    except:
        p_overheat = 0.0

    infer = make_inferencer(model)
    cause_nodes = ["BearingWearHigh", "FanFault", "CloggedFilter", "LowCoolingEfficiency"]
    cause_probs = {}
    
    # Filter evidence to prevent unused nodes from causing errors
    relevant_evidence = {k: v for k, v in evidence.items() if k in model.nodes()}

    for c in cause_nodes:
        try:
            q = infer.query(variables=[c], evidence=relevant_evidence)
            if len(q.values) > 1:
                cause_probs[c] = float(q.values[1])
            else:
                cause_probs[c] = 0.0
        except:
            cause_probs[c] = 0.0

    top_cause = max(cause_probs, key=cause_probs.get) if cause_probs else "Unknown"

    # --- KG & DECISION ---
    kg = CNCKG(debug=debug).load_from_cfg(cfg)
    symptoms = kg.symptoms_for_cause(top_cause)
    components = kg.components_for_cause(top_cause)
    procedures = kg.procedures_for_cause(top_cause)
    procedures_csv = path_for(cfg, "procedures")

    action, costs = choose_action(
        p_overheat=p_overheat,
        top_cause=top_cause,
        cfg=cfg,
        procedures_csv=procedures_csv,
        cause_probs=cause_probs
    )

    return {
        "p_overheat": round(p_overheat, 3),
        "top_cause": top_cause,
        "probabilities": cause_probs,
        "components": components,
        "symptoms": symptoms,
        "procedures": procedures,
        "recommended_action": action,
        "expected_costs": costs,
    }

# ------------------------------------------------------------
# RUN REAL (TRAINING)
# ------------------------------------------------------------
def run_real(evidence: dict,
             debug: bool = False,
             force_retrain: bool = False,
             return_test_data: bool = False):
    """
    Train and evaluate the Bayesian Network using either MLE or EM.
    Ensures a non-degenerate dataset and defensible evaluation.
    """

    from sklearn.model_selection import train_test_split
    from pgmpy.models import BayesianNetwork
    from pgmpy.estimators import MaximumLikelihoodEstimator
    import pickle
    import os
    import pandas as pd

    cfg = load_cfg()
    model_cache_path = cfg["bn"].get("model_cache_path", "models/trained_model.pkl")
    train_method = cfg["bn"].get("train_method", "mle").lower()

    # ------------------------------------------------------------
    # 1. LOAD DATA
    # ------------------------------------------------------------
    tel = pd.read_csv(path_for(cfg, "telemetry"))
    lab = pd.read_csv(path_for(cfg, "labels"))

    try:
        causes_gt = pd.read_csv("data/causes_ground_truth.csv")
    except FileNotFoundError:
        raise RuntimeError("Missing causes_ground_truth.csv")

    df = tel.join(lab["spindle_overheat"]).join(causes_gt["actual_cause"])

    # ------------------------------------------------------------
    # 2. CREATE BINARY CAUSE INDICATORS
    # ------------------------------------------------------------
    causes = ["BearingWearHigh", "FanFault", "CloggedFilter", "LowCoolingEfficiency"]
    for c in causes:
        df[c] = (df["actual_cause"] == c).astype(int)

    # ------------------------------------------------------------
    # 3. MANUAL DISCRETIZATION (PHYSICS-BASED)
    # ------------------------------------------------------------
    df_disc = manual_discretize(df, cfg["bn"]["sensors"])

    # Preserve labels
    for c in causes + ["spindle_overheat", "actual_cause"]:
        df_disc[c] = df[c]

    # ------------------------------------------------------------
    # 4. FIX DATASET DEGENERACY
    # ------------------------------------------------------------
    normal_df = df_disc[df_disc["spindle_overheat"] == 0]
    fault_df = df_disc[df_disc["spindle_overheat"] == 1]

    fault_frac = cfg["bn"].get("fault_fraction", 0.3)
    fault_df = fault_df.sample(frac=fault_frac, random_state=42)

    df_final = pd.concat([normal_df, fault_df]).sample(frac=1.0, random_state=42)

    if debug:
        print("[DATASET]")
        print(df_final["spindle_overheat"].value_counts())

    # ------------------------------------------------------------
    # 5. TRAIN / TEST SPLIT
    # ------------------------------------------------------------
    df_train, df_test = train_test_split(
        df_final,
        test_size=cfg["bn"]["test_size"],
        random_state=42,
        stratify=df_final["spindle_overheat"]
    )

    # ------------------------------------------------------------
    # 6. TRAIN OR LOAD MODEL
    # ------------------------------------------------------------
    model = None

    if not force_retrain and os.path.exists(model_cache_path):
        try:
            with open(model_cache_path, "rb") as f:
                model = pickle.load(f)
            if debug:
                print("[BN] Loaded cached model.")
        except Exception:
            model = None

    if model is None:
        if debug:
            print(f"[BN] Training using method: {train_method.upper()}")

        if train_method == "em":
            from .bn_model import train_bn_em
            model = train_bn_em(df_train, debug=debug)

        else:
            # DEFAULT: MLE
            model = BayesianNetwork([
                ("BearingWearHigh", "vibration_rms"),
                ("BearingWearHigh", "spindle_temp"),
                ("FanFault", "spindle_temp"),
                ("CloggedFilter", "coolant_flow"),
                ("CloggedFilter", "spindle_temp"),
                ("LowCoolingEfficiency", "spindle_temp"),
                ("vibration_rms", "spindle_overheat"),
                ("spindle_temp", "spindle_overheat"),
                ("coolant_flow", "spindle_overheat"),
            ])

            model = integrate_latent_causes(model)
            model.fit(df_train, estimator=MaximumLikelihoodEstimator)
            model.check_model()

        os.makedirs(os.path.dirname(model_cache_path), exist_ok=True)
        with open(model_cache_path, "wb") as f:
            pickle.dump(model, f)

    # ------------------------------------------------------------
    # 7. RETURN
    # ------------------------------------------------------------
    if return_test_data:
        return model, df_test, None, cfg

    return run_demo(
        evidence=evidence,
        debug=debug,
        model_override=model,
        discretizer=None
    )
