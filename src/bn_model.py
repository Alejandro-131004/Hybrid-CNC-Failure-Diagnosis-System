import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

TARGET = "spindle_overheat"  # coluna binária em labels.csv

def load_telemetry(telemetry_csv: str, labels_csv: str) -> pd.DataFrame:
    """Junta sensores (telemetry.csv) e labels (labels.csv)."""
    X = pd.read_csv(telemetry_csv)
    y = pd.read_csv(labels_csv)
    return X.join(y[TARGET])

def discretize(df: pd.DataFrame, continuous_cols: list[str], n_bins: int = 4) -> pd.DataFrame:
    """Discretiza variáveis contínuas em bins quantílicos."""
    dfx = df.copy()
    enc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    dfx[continuous_cols] = enc.fit_transform(dfx[continuous_cols])
    return dfx

def learn_structure(df_disc: pd.DataFrame):
    """Learn BN structure using Hill-Climb with BIC scoring (updated for pgmpy>=0.1.25)."""
    hc = HillClimbSearch(df_disc)
    best_model = hc.estimate(scoring_method=BicScore(df_disc))
    return best_model

def fit_parameters(model: BayesianNetwork, df_disc: pd.DataFrame) -> BayesianNetwork:
    """Ajusta CPDs (Maximum Likelihood)."""
    model.fit(df_disc, estimator=MaximumLikelihoodEstimator)
    return model

def make_inferencer(model: BayesianNetwork) -> VariableElimination:
    """Cria objeto para inferência de variáveis."""
    return VariableElimination(model)
