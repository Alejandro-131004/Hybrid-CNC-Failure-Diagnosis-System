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
    # ===== NORMAL MODE OUTPUT =====
    if not debug:
        cause = result["top_cause"]
        proc = result["procedures"][0] if result["procedures"] else "None"

        # Obter custos e tempos do procedimento
        try:
            dfp = pd.read_csv(path_for(cfg, "procedures"))
            row = dfp[dfp["name"] == proc]
            cost = float(row["spare_parts_cost_eur"].iloc[0]) if not row.empty else 0
            timeh = float(row["effort_h"].iloc[0]) if not row.empty else 0
            risk = int(row["risk_rating"].iloc[0]) if not row.empty else "unknown"
        except Exception:
            cost, timeh, risk = 0, 0, "unknown"

        # --- CORREÇÃO DO ERRO KEYERROR ---
        # Tenta ler o nome 'Demo' (Vibration), se falhar lê o nome 'Real' (vibration_rms)
        # Isto resolve o crash que estava a ter.
        vib_val = evidence.get("Vibration", evidence.get("vibration_rms", "N/A"))
        flow_val = evidence.get("CoolantFlow", evidence.get("coolant_flow", "N/A"))

        # Map binary values to readable states
        vib_state = "High" if int(evidence.get("Vibration", evidence.get("vibration_rms", 0))) == 1 else "Normal"
        flow_state = "Low" if int(evidence.get("CoolantFlow", evidence.get("coolant_flow", 0))) == 1 else "Normal"
        
        print(
            f"\nBecause Vibration={vib_state} and CoolantFlow={flow_state}, "
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



def run_real(evidence: dict, debug=False, force_retrain=False, return_test_data=False):
    """
    Like run_demo, but CPDs are learned from telemetry CSV instead of manually defined.
    """
    from .bn_model import discretize, learn_structure, fit_parameters, save_model, load_model, print_structure
    from sklearn.model_selection import train_test_split
    import os

    cfg = load_cfg()
    
    test_data = None  # Will store test data if needed
    
    # Check for cached model
    model_cache_path = cfg["bn"].get("model_cache_path", "models/trained_model.pkl")
    if not force_retrain and os.path.exists(model_cache_path):
        if debug:
            print(f"[Cache] Loading cached model from {model_cache_path}")
        model = load_model(model_cache_path)
        
        if debug:
            print_structure(model)
        
        # If we need test data but model was cached, load and split data
        if return_test_data:
            tel = pd.read_csv(path_for(cfg, "telemetry"))
            lab = pd.read_csv(path_for(cfg, "labels"))
            df = tel.join(lab["spindle_overheat"])
            df_disc = discretize(df, cfg["bn"]["sensors"], n_bins=cfg["bn"]["discretize_bins"])
            test_size = cfg["bn"].get("test_size", 0.2)
            _, test_data = train_test_split(df_disc, test_size=test_size, random_state=42)
    else:
        # Load real telemetry + labels
        tel = pd.read_csv(path_for(cfg, "telemetry"))
        lab = pd.read_csv(path_for(cfg, "labels"))
        df = tel.join(lab["spindle_overheat"])

        # Discretize sensors
        df_disc = discretize(df, cfg["bn"]["sensors"], n_bins=cfg["bn"]["discretize_bins"])

        # Train-test split
        test_size = cfg["bn"].get("test_size", 0.2)
        df_train, test_data = train_test_split(df_disc, test_size=test_size, random_state=42)
        
        if debug:
            print(f"[Split] Training set: {len(df_train)} samples, Test set: {len(test_data)} samples")
        
        # Downsample if configured
        sample_size = cfg["bn"].get("sample_size", None)
        if sample_size and len(df_train) > sample_size:
            if debug:
                print(f"[Optimization] Downsampling training data from {len(df_train)} to {sample_size} rows.")
            df_train = df_train.sample(n=sample_size, random_state=42)

        # Learn BN structure based on real data
        model = learn_structure(df_train)

        # Inject latent unobserved causes
        model = integrate_latent_causes(
            model,
            vib_node="vibration_rms",
            temp_node="spindle_temp",
            flow_node="coolant_flow",
        )

        # Fit parameters using EM because latent causes have no direct labels
        max_iter = cfg["bn"].get("max_iter", 50)
        n_jobs = cfg["bn"].get("n_jobs", 1)
        model = fit_parameters_em(model, df_train, max_iter=max_iter, n_jobs=n_jobs)
        
        # Save the trained model
        save_model(model, model_cache_path)

    # Return based on what's requested
    if return_test_data:
        return model, test_data
    else:
        # Reuse the same reasoning pipeline with learned model
        return run_demo(evidence, debug=debug, model_override=model)


def evaluate_on_test_set(model, test_data, debug=False):
    """
    Evaluate the model on all test samples and compute accuracy metrics.
    """
    from .bn_model import infer_overheat_prob
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
    
    cfg = load_cfg()
    sensor_cols = cfg["bn"]["sensors"]
    
    predictions = []
    true_labels = []
    
    print(f"\n[Evaluation] Testing on {len(test_data)} samples...")
    
    for idx, row in test_data.iterrows():
        # Extract evidence (only sensor columns)
        evidence = {col: int(row[col]) for col in sensor_cols if col in test_data.columns}
        
        # Get true label
        true_label = int(row["spindle_overheat"])
        true_labels.append(true_label)
        
        # Make prediction
        prob_overheat = infer_overheat_prob(model, evidence)
        predicted_label = 1 if prob_overheat >= 0.5 else 0
        predictions.append(predicted_label)
    
    # Compute metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    cm = confusion_matrix(true_labels, predictions)
    report = classification_report(true_labels, predictions, zero_division=0)
    
    print(f"\n=== [TEST SET EVALUATION RESULTS] ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                0      1")
    print(f"Actual 0   {cm[0][0]:6d} {cm[0][1]:6d}")
    print(f"Actual 1   {cm[1][0]:6d} {cm[1][1]:6d}")
    
    print("\nClassification Report:")
    print(report)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "predictions": predictions,
        "true_labels": true_labels
    }
