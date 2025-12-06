"""
bn_model.py
Hybrid Bayesian Network for CNC Diagnosis
"""

import pandas as pd
import pickle
import numpy as np
import os
from sklearn.preprocessing import KBinsDiscretizer
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator, ExpectationMaximization
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

TARGET = "spindle_overheat"  # from labels.csv

# ==============================================================
# === Data handling and discretization
# ==============================================================

def load_telemetry(telemetry_csv: str, labels_csv: str) -> pd.DataFrame:
    """Merge telemetry and label datasets."""
    X = pd.read_csv(telemetry_csv)
    y = pd.read_csv(labels_csv)
    return X.join(y[TARGET])

# src/bn_model.py

def discretize(df: pd.DataFrame, continuous_cols: list[str], n_bins: int = 4, return_discretizer=False):
    """
    Discretizes continuous sensor columns using KBinsDiscretizer.
    Strategy: 'kmeans' (Best for separating distinct fault clusters from normal data).
    """
    dfx = df.copy()
    
    # Ensure columns are numeric
    for col in continuous_cols:
        dfx[col] = pd.to_numeric(dfx[col], errors='coerce').fillna(0)

    # USAR KMEANS: Melhora a separação de classes (Topicos 1 e 2)
    enc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="kmeans")
    
    dfx[continuous_cols] = enc.fit_transform(dfx[continuous_cols])

    if return_discretizer:
        return dfx, enc

    return dfx


# ==============================================================
# === Data-driven learning (Hill-Climb)
# ==============================================================

def learn_structure(df_disc: pd.DataFrame) -> BayesianNetwork:
    """Learn BN structure using Hill-Climbing and return a BayesianNetwork."""
    hc = HillClimbSearch(df_disc)
    dag = hc.estimate(scoring_method=BicScore(df_disc))
    # Convert edges into a BayesianNetwork object
    model = BayesianNetwork(dag.edges())
    return model



def fit_parameters(model: BayesianNetwork, df_disc: pd.DataFrame) -> BayesianNetwork:
    """Estimate CPDs with Maximum Likelihood Estimation."""
    model.fit(df_disc, estimator=MaximumLikelihoodEstimator)
    return model


def make_inferencer(model: BayesianNetwork) -> VariableElimination:
    """Create inference engine."""
    return VariableElimination(model)


# ==============================================================
# === Knowledge-driven structure (manual causal model)
# ==============================================================

def build_manual_bn() -> BayesianNetwork:
    """Define the BN structure based on known physical and causal relations."""
    model = BayesianNetwork([
        # USING CSV NAMES (snake_case)
        ('BearingWearHigh', 'vibration_rms'),
        ('FanFault', 'spindle_temp'),
        ('CloggedFilter', 'coolant_flow'),
        ('LowCoolingEfficiency', 'spindle_temp'),
        ('vibration_rms', 'spindle_overheat'),
        ('spindle_temp', 'spindle_overheat'),
        ('coolant_flow', 'spindle_overheat')
    ])
    return model


def add_manual_cpds(model: BayesianNetwork) -> BayesianNetwork:
    """Add pre-defined CPDs (domain-based), using CSV names for nodes."""
    cpd_bearing = TabularCPD('BearingWearHigh', 2, [[0.9], [0.1]]) # Prior: 90% normal bearing, 10% worn (root node)
    cpd_fan = TabularCPD('FanFault', 2, [[0.95], [0.05]])
    cpd_filter = TabularCPD('CloggedFilter', 2, [[0.9], [0.1]])
    cpd_cooling = TabularCPD('LowCoolingEfficiency', 2, [[0.85], [0.15]])

    cpd_vibration = TabularCPD(
        'vibration_rms', 2, [[0.95, 0.4], [0.05, 0.6]],
        evidence=['BearingWearHigh'], evidence_card=[2]
    )

    cpd_temp = TabularCPD(
        'spindle_temp', 2,
        [[0.96, 0.8, 0.7, 0.3],
         [0.04, 0.2, 0.3, 0.7]],
        evidence=['FanFault', 'LowCoolingEfficiency'],
        evidence_card=[2, 2]
    )

    cpd_flow = TabularCPD(
        'coolant_flow', 2, [[0.9, 0.4], [0.1, 0.6]],
        evidence=['CloggedFilter'], evidence_card=[2]
    )

    cpd_overheat = TabularCPD(
        'spindle_overheat', 2,
        [[0.99, 0.9, 0.85, 0.3, 0.6, 0.2, 0.1, 0.01],
         [0.01, 0.1, 0.15, 0.7, 0.4, 0.8, 0.9, 0.99]],
        evidence=['vibration_rms', 'spindle_temp', 'coolant_flow'],
        evidence_card=[2, 2, 2]
    )

    model.add_cpds(cpd_bearing, cpd_fan, cpd_filter, cpd_cooling,
                   cpd_vibration, cpd_temp, cpd_flow, cpd_overheat)
    model.check_model()
    return model


def example_inference(model: BayesianNetwork):
    """Run a simple test inference."""
    infer = VariableElimination(model)
    q = infer.query(variables=['spindle_overheat'], evidence={'Vibration': 1, 'CoolantFlow': 1})
    print(q)
    return q


def learn_from_cfg(cfg, telemetry_key="telemetry", labels_key="labels"):
    from .utils import load_csv
    df = load_csv(cfg, telemetry_key, parse_dates=["timestamp"])
    y = load_csv(cfg, labels_key, parse_dates=["timestamp"])
    df = df.join(y["spindle_overheat"])
    df_disc = discretize(df, cfg["bn"]["sensors"], n_bins=cfg["bn"]["discretize_bins"])
    model = learn_structure(df_disc)
    model = fit_parameters(model, df_disc)
    return model

def infer_overheat_prob(model, evidence: dict):
    """
    Perform inference P(spindle_overheat = 1 | evidence).

    Assumes that:
      - The model was trained on discretized data (KBinsDiscretizer).
      - The incoming evidence already uses the same discrete states
        (i.e., integers 0, 1, 2, ... as in df_disc / test_data).
    """
    infer = make_inferencer(model)

    ev = {}
    for k, v in (evidence or {}).items():
        if k in model.nodes():
            # evidence already in correct discrete space
            try:
                ev[k] = int(v)
            except Exception:
                continue

    q = infer.query(variables=["spindle_overheat"], evidence=ev if ev else None)
    # state index 1 = "overheat = 1"
    return float(q.values[1])



def fit_parameters_em(model: BayesianNetwork, df_disc: pd.DataFrame, max_iter: int = 50, n_jobs: int = 1):
    """
    Fit CPDs using the EM algorithm (supports latent variables and missing data).
    """
    print(">> EM max_iter =", max_iter)
    em = ExpectationMaximization(model, df_disc)
    cpds = em.get_parameters(max_iter=max_iter, n_jobs=n_jobs)
    model.add_cpds(*cpds)
    return model


def integrate_latent_causes(model: BayesianNetwork) -> BayesianNetwork:
    """
    Add latent cause nodes to the BN and connect them causally to sensors.
    This enforces a root-cause structure:
        BearingWearHigh      → vibration_rms
        FanFault             → spindle_temp
        CloggedFilter        → coolant_flow
        LowCoolingEfficiency → spindle_temp
    """
    latent = ["BearingWearHigh", "FanFault", "CloggedFilter", "LowCoolingEfficiency"]

    # 1) Ensuring that latent nodes exist
    for c in latent:
        if c not in model.nodes():
            model.add_node(c)

    # 2) Add latent edges → sensor (if the sensor exists in the graph)
    causal_edges = [
        ("BearingWearHigh", "vibration_rms"),
        ("FanFault", "spindle_temp"),
        ("CloggedFilter", "coolant_flow"),
        ("LowCoolingEfficiency", "spindle_temp"),
    ]

    for u, v in causal_edges:
        if v in model.nodes() and (u, v) not in model.edges():
            model.add_edge(u, v)

    # Mark latents
    model.latents = set(latent)
    return model




# ==============================================================
# === Model persistence
# ==============================================================

def save_model(model: BayesianNetwork, path: str):
    """Save a trained Bayesian Network model to disk using pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[Model] Saved to {path}")


def load_model(path: str) -> BayesianNetwork:
    """Load a trained Bayesian Network model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print(f"[Model] Loaded from {path}")
    return model


# ==============================================================
# === Model Analysis
# ==============================================================

def print_structure(model: BayesianNetwork):
    """Print the structure of the Bayesian Network."""
    print("\n=== [BN STRUCTURE] ===")
    print(f"Nodes: {len(model.nodes())}")
    print(f"Edges: {len(model.edges())}")
    
    if hasattr(model, "latents"):
        print(f"Latent Variables: {model.latents}")
    
    print("\nEdges:")
    for u, v in model.edges():
        print(f"  {u} -> {v}")


def print_cpds(model: BayesianNetwork):
    """Print the CPDs of the Bayesian Network."""
    print("\n=== [BN PARAMETERS (CPDs)] ===")
    for cpd in model.get_cpds():
        print(f"\nCPD for {cpd.variable}:")
        print(cpd)
