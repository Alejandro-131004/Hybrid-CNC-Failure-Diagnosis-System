"""
bn_model.py
Hybrid Bayesian Network for CNC Diagnosis
"""

import pandas as pd
import pickle
import numpy as np
import os
from sklearn.preprocessing import KBinsDiscretizer
from pgmpy.estimators import (
    HillClimbSearch,
    BicScore,
    MaximumLikelihoodEstimator,
    ExpectationMaximization,
)
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

TARGET = "spindle_overheat"

# ==============================================================
# === 1. CENTRAL STRUCTURE DEFINITION (SOT - Source of Truth)
# ==============================================================


def define_bn_structure() -> BayesianNetwork:
    """
    Defines the network structure in a single place.
    If you change an edge here, it changes across the entire project
    (MLE, EM, Demo).
    """
    model = BayesianNetwork(
        [
            # Causes -> Symptoms
            ("BearingWearHigh", "vibration_rms"),
            ("BearingWearHigh", "spindle_temp"),
            ("FanFault", "spindle_temp"),
            ("CloggedFilter", "coolant_flow"),
            ("CloggedFilter", "spindle_temp"),
            ("LowCoolingEfficiency", "spindle_temp"),
            # Symptoms -> Overheat (Target)
            ("vibration_rms", "spindle_overheat"),
            ("spindle_temp", "spindle_overheat"),
            ("coolant_flow", "spindle_overheat"),
        ]
    )

    # Explicit definition of latent variables (causes)
    model.latents = {
        "BearingWearHigh",
        "FanFault",
        "CloggedFilter",
        "LowCoolingEfficiency",
    }

    return model


def integrate_latent_causes(model: BayesianNetwork) -> BayesianNetwork:
    """
    Ensures that latent nodes exist in the model.
    (Now redundant because define_bn_structure already includes them,
    but kept for safety.)
    """
    if not hasattr(model, "latents"):
        model.latents = {
            "BearingWearHigh",
            "FanFault",
            "CloggedFilter",
            "LowCoolingEfficiency",
        }

    for node in model.latents:
        if node not in model.nodes():
            model.add_node(node)

    return model


# ==============================================================
# === 2. DATA HANDLING
# ==============================================================


def load_telemetry(telemetry_csv: str, labels_csv: str) -> pd.DataFrame:
    """Merge telemetry and label datasets."""
    X = pd.read_csv(telemetry_csv)
    y = pd.read_csv(labels_csv)
    return X.join(y[TARGET])


def discretize(
    df: pd.DataFrame,
    continuous_cols: list[str],
    n_bins: int = 4,
    return_discretizer=False,
):
    """
    Discretizes continuous sensor columns using KBinsDiscretizer
    with KMeans strategy.
    """
    dfx = df.copy()

    for col in continuous_cols:
        dfx[col] = pd.to_numeric(dfx[col], errors="coerce").fillna(0)

    enc = KBinsDiscretizer(
        n_bins=n_bins, encode="ordinal", strategy="kmeans"
    )

    valid_cols = [c for c in continuous_cols if c in dfx.columns]
    if valid_cols:
        dfx[valid_cols] = enc.fit_transform(dfx[valid_cols])

    if return_discretizer:
        return dfx, enc

    return dfx


# ==============================================================
# === 3. TRAINING (LEARNING)
# ==============================================================


def train_bn_em(df_train, debug=False, max_iter=50):
    """
    Train using EM with Structural Simplification AND Hard Initialization.

    We force EM to respect the physical separation of symptoms by initializing
    CPDs manually and running EM carefully to avoid immediate collapse.
    """
    from pgmpy.models import BayesianNetwork
    from pgmpy.estimators import ExpectationMaximization
    from pgmpy.factors.discrete import TabularCPD

    # 1. Simplified structure (3 causes)
    base_model = define_bn_structure()
    edges = list(base_model.edges())

    # Remove LowCoolingEfficiency to avoid redundancy
    em_edges = [
        (u, v)
        for u, v in edges
        if u != "LowCoolingEfficiency" and v != "LowCoolingEfficiency"
    ]
    model = BayesianNetwork(em_edges)

    # Explicit latent causes
    model.latents = {"BearingWearHigh", "FanFault", "CloggedFilter"}
    for node in model.latents:
        if node not in model.nodes():
            model.add_node(node)

    bn_nodes = list(model.nodes())

    # 2. Data (observable variables only)
    observed_nodes = [n for n in bn_nodes if n not in model.latents]
    df_em = df_train[observed_nodes].copy()
    for col in df_em.columns:
        df_em[col] = df_em[col].astype("Int64").fillna(0).astype(int)

    if debug:
        print("[BN-EM] Training EM with Simplified Structure...")

    # 3. CRITICAL: Hard warm start (aggressive manual initialization)
    # pgmpy requires CPDs for all nodes (or will complain during checks).
    # We'll inject strong CPDs to separate coolant_flow and vibration_rms.

    # Priors for latent causes (start uniform)
    cpd_priors = [
        TabularCPD(variable=c, variable_card=2, values=[[0.5], [0.5]])
        for c in ["BearingWearHigh", "FanFault", "CloggedFilter"]
    ]

    # Strong links (core trick):
    # Assumption: 0=Normal, 1=Fault/Low/High (binary discretization)
    cpd_filter_flow = TabularCPD(
        variable="coolant_flow",
        variable_card=2,
        values=[[0.9, 0.1], [0.1, 0.9]],
        evidence=["CloggedFilter"],
        evidence_card=[2],
    )

    cpd_bearing_vib = TabularCPD(
        variable="vibration_rms",
        variable_card=2,
        values=[[0.9, 0.1], [0.1, 0.9]],
        evidence=["BearingWearHigh"],
        evidence_card=[2],
    )

    # NOTE: spindle_temp has 3 parents in this simplified structure:
    # BearingWearHigh, FanFault, CloggedFilter
    # A full CPD would be 2 x (2*2*2)=2x8 values. We let EM learn it.

    try:
        # If add_manual_cpds exists and is compatible, use it to initialize CPDs
        model = add_manual_cpds(model)

        # Overwrite / reinforce critical separations
        model.add_cpds(cpd_filter_flow, cpd_bearing_vib)

        if debug:
            print("[BN-EM] Hard Warm Start applied successfully.")
    except Exception as e:
        print(f"[BN-EM] Warning: could not fully apply Hard Warm Start ({e}).")
        # At minimum, add priors so EM can start
        model.add_cpds(*cpd_priors)

    # 4. Run EM (uses current CPDs as initialization if present)
    em = ExpectationMaximization(model, df_em)

    try:
        cpds = em.get_parameters(max_iter=max_iter)
        model.add_cpds(*cpds)
    except Exception as e:
        print(f"[BN-EM] EM loop failed ({e}). Reverting to initial CPDs.")

    model.check_model()
    return model


def fit_parameters(
    model: BayesianNetwork, df_disc: pd.DataFrame
) -> BayesianNetwork:
    """Estimate CPDs using Maximum Likelihood Estimation (MLE)."""
    model.fit(df_disc, estimator=MaximumLikelihoodEstimator)
    return model


def learn_structure(df_disc: pd.DataFrame) -> BayesianNetwork:
    """
    Learn network structure purely from data using Hill-Climbing.
    Used only for comparison, not in the hybrid approach.
    """
    hc = HillClimbSearch(df_disc)
    dag = hc.estimate(scoring_method=BicScore(df_disc))
    model = BayesianNetwork(dag.edges())
    return model


# ==============================================================
# === 4. INFERENCE
# ==============================================================


def make_inferencer(model: BayesianNetwork) -> VariableElimination:
    return VariableElimination(model)


def infer_overheat_prob(model, evidence: dict):
    """
    Perform inference:
    P(spindle_overheat = 1 | evidence)
    """
    infer = make_inferencer(model)

    ev = {}
    for k, v in (evidence or {}).items():
        if k in model.nodes():
            try:
                ev[k] = int(v)
            except Exception:
                continue

    try:
        q = infer.query(
            variables=["spindle_overheat"],
            evidence=ev if ev else None,
        )
        # State 1 is assumed to represent "overheat = true"
        return float(q.values[1])
    except Exception:
        return 0.0


# ==============================================================
# === 5. UTILITIES AND PERSISTENCE
# ==============================================================


def save_model(model: BayesianNetwork, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[Model] Saved to {path}")


def load_model(path: str) -> BayesianNetwork:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"[Model] Loaded from {path}")
    return model


def print_structure(model: BayesianNetwork):
    print("\n=== [BN STRUCTURE] ===")
    print(f"Nodes: {len(model.nodes())}")
    print(f"Edges: {len(model.edges())}")

    if hasattr(model, "latents"):
        print(f"Latent Variables: {model.latents}")

    print("\nEdges:")
    for u, v in model.edges():
        print(f"  {u} -> {v}")


def print_cpds(model: BayesianNetwork):
    print("\n=== [BN PARAMETERS (CPDs)] ===")
    for cpd in model.get_cpds():
        print(f"\nCPD for {cpd.variable}:")
        print(cpd)


def align_em_states(model: BayesianNetwork, causes_map: dict):
    """
    Helper to warn the user about EM latent-state semantic alignment
    (state 0 vs state 1).
    """
    print("\n[EM Alignment] Verifying semantic alignment of latent states...")
    for cause, symptom in causes_map.items():
        if (
            cause not in model.nodes()
            or symptom not in model.nodes()
        ):
            continue
        print(
            f" -> Please manually inspect CPD for '{symptom}' given '{cause}'."
        )
    return model


# ==============================================================
# === 6. LEGACY / MANUAL SETUP (REFERENCE ONLY)
# ==============================================================


def build_manual_bn() -> BayesianNetwork:
    """Legacy wrapper kept for compatibility."""
    return define_bn_structure()


def add_manual_cpds(model: BayesianNetwork) -> BayesianNetwork:
    """
    Add predefined CPDs.
    Kept only to allow testing of a fully manual network.
    """
    cpd_bearing = TabularCPD("BearingWearHigh", 2, [[0.9], [0.1]])
    cpd_fan = TabularCPD("FanFault", 2, [[0.95], [0.05]])
    cpd_filter = TabularCPD("CloggedFilter", 2, [[0.9], [0.1]])
    cpd_cooling = TabularCPD(
        "LowCoolingEfficiency", 2, [[0.85], [0.15]]
    )

    model.add_cpds(
        cpd_bearing,
        cpd_fan,
        cpd_filter,
        cpd_cooling,
    )

    return model
