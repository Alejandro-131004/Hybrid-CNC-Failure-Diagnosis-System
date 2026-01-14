"""
integration.py
Pipeline connecting BN inference with KG and Decision Module.
"""
from .utils import load_cfg, path_for
from .bn_model import infer_overheat_prob, make_inferencer, integrate_latent_causes, align_em_states, train_bn_em
from .kg_module import CNCKG
from .decision import choose_action
import pandas as pd
import numpy as np
import os
import pickle
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

def manual_discretize(df, sensors_list):
    """
    Discretizes sensors using PROFESSOR'S EXACT THRESHOLDS.
    """
    df_disc = df.copy()
    
    # Regra da Prof: Temp > 87
    if "spindle_temp" in df.columns:
        vals = pd.to_numeric(df["spindle_temp"], errors='coerce').fillna(0)
        df_disc["spindle_temp"] = (vals > 87.0).astype(int)
        
    # Regra da Prof: Vib > 1.5
    if "vibration_rms" in df.columns:
        vals = pd.to_numeric(df["vibration_rms"], errors='coerce').fillna(0)
        df_disc["vibration_rms"] = (vals > 1.5).astype(int)
        
    # Regra da Prof: Flow < 0.6
    # Nota: Aqui invertemos a lógica (1 = Bad/Low), por isso usamos <
    if "coolant_flow" in df.columns:
        vals = pd.to_numeric(df["coolant_flow"], errors='coerce').fillna(0)
        df_disc["coolant_flow"] = (vals < 0.6).astype(int)
        
    for col in sensors_list:
        if col not in ["spindle_temp", "vibration_rms", "coolant_flow"] and col in df.columns:
            df_disc[col] = 0 
            
    return df_disc

def run_demo(evidence: dict, debug=True, model_override=None, discretizer=None):
    cfg = load_cfg()
    if model_override is None: return {}
    model = model_override

    # Discretize on the fly
    temp_df = pd.DataFrame([evidence])
    for col in ["spindle_temp", "vibration_rms", "coolant_flow"]:
        if col in temp_df.columns: temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').fillna(0)
    
    disc_df = manual_discretize(temp_df, list(evidence.keys()))
    clean_evidence = {k: v for k, v in disc_df.iloc[0].to_dict().items() if k in model.nodes()}

    # Inference
    try: p_overheat = infer_overheat_prob(model, clean_evidence)
    except: p_overheat = 0.0

    infer = make_inferencer(model)
    cause_nodes = ["BearingWearHigh", "FanFault", "CloggedFilter", "LowCoolingEfficiency"]
    cause_probs = {}
    
    relevant_evidence = {k: v for k, v in clean_evidence.items() if k in model.nodes()}
    for c in cause_nodes:
        if c not in model.nodes(): continue
        try:
            q = infer.query(variables=[c], evidence=relevant_evidence)
            cause_probs[c] = float(q.values[1]) if len(q.values) > 1 else 0.0
        except: cause_probs[c] = 0.0

    top_cause = max(cause_probs, key=cause_probs.get) if cause_probs else "Unknown"

    # KG & Decision
    kg = CNCKG(debug=debug).load_from_cfg(cfg)
    # Correção: passar o top_cause limpo
    symptoms = kg.symptoms_for_cause(top_cause)
    components = kg.components_for_cause(top_cause)
    procedures = kg.procedures_for_cause(top_cause)
    
    try: procedures_csv = path_for(cfg, "procedures")
    except: procedures_csv = "data/procedures.csv"

    action, costs = choose_action(p_overheat, top_cause, cfg, procedures_csv, cause_probs)

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

def run_real(evidence: dict, debug=False, force_retrain=False, return_test_data=False, train_method_override=None):
    from sklearn.model_selection import train_test_split
    # Note: train_bn_em is imported at top of file or here locally
    from pgmpy.models import BayesianNetwork
    from pgmpy.estimators import MaximumLikelihoodEstimator
    from pgmpy.factors.discrete import TabularCPD
    
    cfg = load_cfg()
    train_method = (train_method_override or cfg["bn"].get("train_method", "mle")).lower()
    
    # Cache separated for MLE and EM
    base_path = cfg["bn"].get("model_cache_path", "models/trained_model.pkl")
    model_cache_path = base_path.replace(".pkl", f"_{train_method}.pkl")

    if debug: print(f"[BN] Method: {train_method.upper()} | Cache: {model_cache_path}")

    # 1. Load Data
    tel = pd.read_csv(path_for(cfg, "telemetry"))
    lab = pd.read_csv(path_for(cfg, "labels"))
    
    # 2. Ground Truth Logic
    gt_path = path_for(cfg, "telemetry").replace("telemetry.csv", "causes_ground_truth.csv")
    if not os.path.exists(gt_path): gt_path = "data/causes_ground_truth.csv"
    if not os.path.exists(gt_path): gt_path = "../data/causes_ground_truth.csv"
    
    try: causes_gt = pd.read_csv(gt_path)
    except: raise RuntimeError("Missing causes_ground_truth.csv")

    df = tel.join(lab["spindle_overheat"]).join(causes_gt["actual_cause"])
    causes = ["BearingWearHigh", "FanFault", "CloggedFilter", "LowCoolingEfficiency"]
    for c in causes: df[c] = (df["actual_cause"] == c).astype(int)

    # 3. Discretize (Using Manual Rules to match Data Augmentation)
    df_disc = manual_discretize(df, cfg["bn"]["sensors"])
    for c in causes + ["spindle_overheat", "actual_cause"]: df_disc[c] = df[c]

    # 4. Balance Data (Crucial for learning conditional probs, bad for priors)
    normal_df = df_disc[df_disc["spindle_overheat"] == 0]
    fault_df = df_disc[df_disc["spindle_overheat"] == 1]
    
    if not fault_df.empty: 
        fault_df = fault_df.sample(frac=cfg["bn"].get("fault_fraction", 1.0), random_state=42)
    
    # Create balanced dataset for training
    df_final = pd.concat([normal_df, fault_df]).sample(frac=1.0, random_state=42)
    df_train, df_test = train_test_split(df_final, test_size=cfg["bn"]["test_size"], random_state=42)

    # 5. Train or Load Model
    model = None
    if not force_retrain and os.path.exists(model_cache_path):
        try:
            with open(model_cache_path, "rb") as f: model = pickle.load(f)
            if debug: print("[BN] Loaded cached model.")
        except: model = None

    if model is None:
        if debug: print(f"[BN] Training new model ({train_method})...")
        
        # === A. EXPECTATION-MAXIMIZATION (EM) ===
        if train_method == "em":
            # Train using the EM wrapper in bn_model.py
            model = train_bn_em(df_train, debug=debug, max_iter=cfg["bn"].get("em_max_iter", 50))
            
            # CRITICAL: Verify if EM learned inverted labels (State 0 = Fault vs State 1 = Fault)
            causes_map = {
                "FanFault": "spindle_temp",
                "BearingWearHigh": "vibration_rms",
                "CloggedFilter": "coolant_flow", 
                "LowCoolingEfficiency": "spindle_temp"
            }
            model = align_em_states(model, causes_map)

        # === B. MAXIMUM LIKELIHOOD ESTIMATION (MLE) ===
        else:
            # Define Structure explicitly
            model = BayesianNetwork([
                ("BearingWearHigh", "vibration_rms"), ("BearingWearHigh", "spindle_temp"),
                ("FanFault", "spindle_temp"),
                ("CloggedFilter", "coolant_flow"), ("CloggedFilter", "spindle_temp"),
                ("LowCoolingEfficiency", "spindle_temp"),
                ("vibration_rms", "spindle_overheat"), ("spindle_temp", "spindle_overheat"),
                ("coolant_flow", "spindle_overheat"),
            ])
            model = integrate_latent_causes(model)
            
            # --- CORRECTION FIX: Force cardinality=2 ---
            # Define explicitamente que todos os nós têm estados [0, 1].
            # Isto impede que o pgmpy assuma cardinalidade=1 se faltarem dados de falha no treino.
            state_names = {node: [0, 1] for node in model.nodes()}
            
            model.fit(df_train, estimator=MaximumLikelihoodEstimator, state_names=state_names)
            # -------------------------------------------

            # --- CORRECTION: Adjust Priors ---

            # --- CORRECTION: Adjust Priors ---
            # Since we trained on balanced data (approx 50/50 faults), the model thinks
            # faults are very common. We must manually reset priors to realistic values (e.g., 1%)
            # so the Decision Module doesn't overestimate risk/cost.
            if debug: print("[INFO] Adjusting priors to realistic values for Decision Making...")
            
            model.add_cpds(
                TabularCPD('BearingWearHigh', 2, [[0.99], [0.01]]),
                TabularCPD('FanFault', 2, [[0.99], [0.01]]),
                TabularCPD('CloggedFilter', 2, [[0.99], [0.01]]),
                TabularCPD('LowCoolingEfficiency', 2, [[0.99], [0.01]])
            )

        # Save the trained model
        os.makedirs(os.path.dirname(model_cache_path), exist_ok=True)
        with open(model_cache_path, "wb") as f: pickle.dump(model, f)

    if return_test_data: return model, df_test, None, cfg
    return run_demo(evidence, debug=debug, model_override=model)