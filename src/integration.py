"""
integration.py
Pipeline connecting BN inference with KG and Decision Module.
"""
from .utils import load_cfg, path_for
# ADDED: define_bn_structure
from .bn_model import (
    infer_overheat_prob,
    make_inferencer,
    integrate_latent_causes,
    align_em_states,
    train_bn_em,
    define_bn_structure,
)
from .kg_module import CNCKG
from .decision import choose_action
import pandas as pd
import numpy as np
import os
import pickle
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.discrete import TabularCPD


def manual_discretize(df, sensors_list):
    """Discretizes sensors using PROFESSOR'S EXACT THRESHOLDS."""
    df_disc = df.copy()

    if "spindle_temp" in df.columns:
        vals = pd.to_numeric(df["spindle_temp"], errors="coerce").fillna(0)
        df_disc["spindle_temp"] = (vals > 87.0).astype(int)

    if "vibration_rms" in df.columns:
        vals = pd.to_numeric(df["vibration_rms"], errors="coerce").fillna(0)
        df_disc["vibration_rms"] = (vals > 1.5).astype(int)

    if "coolant_flow" in df.columns:
        vals = pd.to_numeric(df["coolant_flow"], errors="coerce").fillna(0)
        df_disc["coolant_flow"] = (vals < 0.6).astype(int)

    for col in sensors_list:
        if col not in ["spindle_temp", "vibration_rms", "coolant_flow"] and col in df.columns:
            df_disc[col] = 0

    return df_disc


def run_demo(evidence: dict, debug=True, model_override=None, discretizer=None):
    cfg = load_cfg()
    if model_override is None:
        return {}

    model = model_override

    # Discretize on the fly
    temp_df = pd.DataFrame([evidence])
    for col in ["spindle_temp", "vibration_rms", "coolant_flow"]:
        if col in temp_df.columns:
            temp_df[col] = pd.to_numeric(temp_df[col], errors="coerce").fillna(0)

    disc_df = manual_discretize(temp_df, list(evidence.keys()))
    clean_evidence = {
        k: v
        for k, v in disc_df.iloc[0].to_dict().items()
        if k in model.nodes()
    }

    # Inference for Overheat
    try:
        p_overheat = infer_overheat_prob(model, clean_evidence)
    except Exception:
        p_overheat = 0.0

    infer = make_inferencer(model)
    cause_nodes = [
        "BearingWearHigh",
        "FanFault",
        "CloggedFilter",
        "LowCoolingEfficiency",
    ]

    cause_probs = {}

    # FIX: strictly filter evidence to existing nodes
    relevant_evidence = {
        k: int(v) for k, v in clean_evidence.items() if k in model.nodes()
    }

    for c in cause_nodes:
        if c not in model.nodes():
            continue

        # If the cause appears in evidence (wrong manual input), remove it to avoid bias
        ev_for_cause = relevant_evidence.copy()
        if c in ev_for_cause:
            del ev_for_cause[c]

        try:
            q = infer.query(variables=[c], evidence=ev_for_cause)
            cause_probs[c] = float(q.values[1]) if len(q.values) > 1 else 0.0
        except Exception:
            cause_probs[c] = 0.0

    top_cause = (
        max(cause_probs, key=cause_probs.get) if cause_probs else "Unknown"
    )

    # KG & Decision
    kg = CNCKG(debug=debug).load_from_cfg(cfg)
    symptoms = kg.symptoms_for_cause(top_cause)
    components = kg.components_for_cause(top_cause)
    procedures = kg.procedures_for_cause(top_cause)

    try:
        procedures_csv = path_for(cfg, "procedures")
    except Exception:
        procedures_csv = "data/procedures.csv"

    action, costs = choose_action(
        p_overheat, top_cause, cfg, procedures_csv, cause_probs
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


def run_real(
    evidence: dict,
    debug=False,
    force_retrain=False,
    return_test_data=False,
    train_method_override=None,
):
    from sklearn.model_selection import train_test_split

    cfg = load_cfg()
    train_method = (
        train_method_override or cfg["bn"].get("train_method", "mle")
    ).lower()

    base_path = cfg["bn"].get("model_cache_path", "models/trained_model.pkl")
    model_cache_path = base_path.replace(".pkl", f"_{train_method}.pkl")

    if debug:
        print(
            f"[BN] Method: {train_method.upper()} | Cache: {model_cache_path}"
        )

    # 1. Load data
    tel = pd.read_csv(path_for(cfg, "telemetry"))
    lab = pd.read_csv(path_for(cfg, "labels"))

    gt_path = path_for(cfg, "telemetry").replace(
        "telemetry.csv", "causes_ground_truth.csv"
    )
    if not os.path.exists(gt_path):
        gt_path = "data/causes_ground_truth.csv"

    try:
        causes_gt = pd.read_csv(gt_path)
    except Exception:
        raise RuntimeError("Missing causes_ground_truth.csv")

    df = tel.join(lab["spindle_overheat"]).join(
        causes_gt["actual_cause"]
    )

    causes = [
        "BearingWearHigh",
        "FanFault",
        "CloggedFilter",
        "LowCoolingEfficiency",
    ]

    for c in causes:
        df[c] = (df["actual_cause"] == c).astype(int)

    # 2. Discretization
    df_disc = manual_discretize(df, cfg["bn"]["sensors"])
    for c in causes + ["spindle_overheat", "actual_cause"]:
        df_disc[c] = df[c]

    # 3. Data balancing
    normal_df = df_disc[df_disc["spindle_overheat"] == 0]
    fault_df = df_disc[df_disc["spindle_overheat"] == 1]

    if not fault_df.empty:
        fault_df = fault_df.sample(
            frac=cfg["bn"].get("fault_fraction", 1.0),
            random_state=42,
        )

    df_final = pd.concat([normal_df, fault_df]).sample(
        frac=1.0, random_state=42
    )

    df_train, df_test = train_test_split(
        df_final,
        test_size=cfg["bn"]["test_size"],
        random_state=42,
    )

    # 4. Train or load model
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
            print(f"[BN] Training new model ({train_method})...")

        # === A. EXPECTATION-MAXIMIZATION (EM) ===
        if train_method == "em":
            model = train_bn_em(
                df_train,
                debug=debug,
                max_iter=cfg["bn"].get("em_max_iter", 50),
            )

            causes_map = {
                "FanFault": "spindle_temp",
                "BearingWearHigh": "vibration_rms",
                "CloggedFilter": "coolant_flow",
                "LowCoolingEfficiency": "spindle_temp",
            }

            model = align_em_states(model, causes_map)

        # === B. MAXIMUM LIKELIHOOD ESTIMATION (MLE) ===
        else:
            # FIX: use centralized structure definition
            model = define_bn_structure()
            model = integrate_latent_causes(model)

            # Force binary cardinality
            state_names = {node: [0, 1] for node in model.nodes()}
            model.fit(
                df_train,
                estimator=MaximumLikelihoodEstimator,
                state_names=state_names,
            )

            # Adjust priors to realistic fault probabilities
            if debug:
                print("[INFO] Adjusting priors to realistic values...")

            model.add_cpds(
                TabularCPD("BearingWearHigh", 2, [[0.99], [0.01]]),
                TabularCPD("FanFault", 2, [[0.99], [0.01]]),
                TabularCPD("CloggedFilter", 2, [[0.99], [0.01]]),
                TabularCPD("LowCoolingEfficiency", 2, [[0.99], [0.01]]),
            )

        os.makedirs(os.path.dirname(model_cache_path), exist_ok=True)
        with open(model_cache_path, "wb") as f:
            pickle.dump(model, f)

    if return_test_data:
        return model, df_test, None, cfg

    return run_demo(evidence, debug=debug, model_override=model)
