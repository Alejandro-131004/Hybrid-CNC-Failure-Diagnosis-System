"""
bn_model.py
Hybrid Bayesian Network for CNC Diagnosis
"""

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator, ExpectationMaximization
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD


TARGET = "spindle_overheat"  # from labels.csv


# ==============================================================
# === Data handling and discretization
# ==============================================================

def load_telemetry(telemetry_csv: str, labels_csv: str) -> pd.DataFrame:
    """Merge telemetry and label datasets."""
    X = pd.read_csv(telemetry_csv)
    y = pd.read_csv(labels_csv)
    return X.join(y[TARGET])


def discretize(df: pd.DataFrame, continuous_cols: list[str], n_bins: int = 4) -> pd.DataFrame:
    """Discretize continuous variables into quantile-based bins."""
    dfx = df.copy()
    enc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    dfx[continuous_cols] = enc.fit_transform(dfx[continuous_cols])
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
        ('BearingWear', 'Vibration'),
        ('FanFault', 'SpindleTemp'),
        ('CloggedFilter', 'CoolantFlow'),
        ('LowCoolingEfficiency', 'SpindleTemp'),
        ('Vibration', 'Overheat'),
        ('SpindleTemp', 'Overheat'),
        ('CoolantFlow', 'Overheat')
    ])
    return model


def add_manual_cpds(model: BayesianNetwork) -> BayesianNetwork:
    """Add pre-defined CPDs (domain-based)."""
    cpd_bearing = TabularCPD('BearingWear', 2, [[0.9], [0.1]]) # Prior: 90% normal bearing, 10% worn (root node)
    cpd_fan = TabularCPD('FanFault', 2, [[0.95], [0.05]])
    cpd_filter = TabularCPD('CloggedFilter', 2, [[0.9], [0.1]])
    cpd_cooling = TabularCPD('LowCoolingEfficiency', 2, [[0.85], [0.15]])

    cpd_vibration = TabularCPD(
        'Vibration', 2, [[0.95, 0.4], [0.05, 0.6]],
        evidence=['BearingWear'], evidence_card=[2]
    )

    cpd_temp = TabularCPD(
        'SpindleTemp', 2,
        [[0.96, 0.8, 0.7, 0.3],
         [0.04, 0.2, 0.3, 0.7]],
        evidence=['FanFault', 'LowCoolingEfficiency'],
        evidence_card=[2, 2]
    )

    cpd_flow = TabularCPD(
        'CoolantFlow', 2, [[0.9, 0.4], [0.1, 0.6]],
        evidence=['CloggedFilter'], evidence_card=[2]
    )

    cpd_overheat = TabularCPD(
        'Overheat', 2,
        [[0.99, 0.9, 0.85, 0.3, 0.6, 0.2, 0.1, 0.01],
         [0.01, 0.1, 0.15, 0.7, 0.4, 0.8, 0.9, 0.99]],
        evidence=['Vibration', 'SpindleTemp', 'CoolantFlow'],
        evidence_card=[2, 2, 2]
    )

    model.add_cpds(cpd_bearing, cpd_fan, cpd_filter, cpd_cooling,
                   cpd_vibration, cpd_temp, cpd_flow, cpd_overheat)
    model.check_model()
    return model


def example_inference(model: BayesianNetwork):
    """Run a simple test inference."""
    infer = VariableElimination(model)
    q = infer.query(variables=['Overheat'], evidence={'Vibration': 1, 'CoolantFlow': 1})
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
    infer = make_inferencer(model)
    q = infer.query(variables=["Overheat"], evidence=evidence)
    return float(q.values[1])  # P(Overheat=1)


def fit_parameters_em(model: BayesianNetwork, df_disc: pd.DataFrame, max_iter: int = 100):
    """
    Estimate CPDs with EM (handles hidden/latent variables and missing data).
    Assumes df_disc contém nós observáveis (sensores, Overheat) e
    não contém as causas latentes (ficam como hidden).
    """
    em = ExpectationMaximization(model)
    em.fit(df_disc, max_iter=max_iter)
    return model

def fit_parameters_em(model: BayesianNetwork, df_disc: pd.DataFrame, max_iter: int = 50):
    """
    Fit CPDs using the EM algorithm (supports latent variables and missing data).
    """
    em = ExpectationMaximization(model)
    em.fit(df_disc, max_iter=max_iter)
    return model


def integrate_latent_causes(model: BayesianNetwork) -> BayesianNetwork:
    """
    Integrates latent cause variables into the learned BN structure and adds
    domain-driven causal edges between unobserved causes and observable sensors.
    """
    latent = ["BearingWearHigh", "FanFault", "CloggedFilter", "LowCoolingEfficiency"]
    for c in latent:
        if c not in model.nodes():
            model.add_node(c)

    edges = [
        ("BearingWearHigh", "Vibration"),
        ("FanFault", "SpindleTemp"),
        ("CloggedFilter", "CoolantFlow"),
        ("LowCoolingEfficiency", "SpindleTemp"),
    ]
    for u, v in edges:
        if (u, v) not in model.edges():
            model.add_edge(u, v)

    return model
