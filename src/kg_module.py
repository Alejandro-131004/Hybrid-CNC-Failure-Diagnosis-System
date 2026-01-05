"""
kg_module.py
Hybrid CNC Knowledge Graph Builder and Query Module
"""

from rdflib import Graph, Namespace, Literal, RDF, RDFS, URIRef
import pandas as pd
from .utils import load_csv  # Agora importa do utils unificado

EX = Namespace("http://hybridcnc.org/ontology#")

class CNCKG:
    def __init__(self, debug=True):
        self.g = Graph()
        self.g.bind("ex", EX)
        self.debug = debug

    def _uri(self, s: str) -> URIRef:
        safe_str = str(s).strip().replace(" ", "_")
        return EX[safe_str]

    def load_from_cfg(self, cfg, ontology_path="ontology.ttl"):
        # 1. Load Ontology
        try:
            # Tenta caminho local ou relativo
            if not os.path.exists(ontology_path) and os.path.exists(f"../{ontology_path}"):
                ontology_path = f"../{ontology_path}"
            
            self.g.parse(ontology_path, format="turtle")
            if self.debug: print(f"Ontology loaded: {ontology_path}")
        except Exception as e:
            print(f"Warning: could not parse ontology ({e})")

        # 2. Add Entities
        for key, cls in [("components", "Component"), ("causes", "Cause"), 
                         ("symptoms", "Symptom"), ("procedures", "Procedure")]:
            try:
                df = load_csv(cfg, key)
                name_col = "name" if "name" in df.columns else df.columns[0]
                for v in df[name_col].unique():
                    self.g.add((self._uri(v), RDF.type, EX[cls]))
                    self.g.add((self._uri(v), RDFS.label, Literal(v)))
            except Exception as e:
                print(f"Warning skipping {key}: {e}")

        # 3. Add Relations (relations.csv)
        try:
            rel = load_csv(cfg, "relations")
            mapping = {"causesSymptom": "hasSymptom", "affectsComponent": "affects"}
            for _, r in rel.iterrows():
                subj, raw_pred, obj = self._uri(r["subj"]), r["pred"], self._uri(r["obj"])
                pred = self._uri(mapping.get(raw_pred, raw_pred))
                self.g.add((subj, pred, obj))
        except: pass

        # 4. CRITICAL FIX: Add relations from procedures.csv (Mitigates Cause)
        try:
            proc_df = load_csv(cfg, "procedures")
            if "mitigates_cause" in proc_df.columns:
                for _, row in proc_df.iterrows():
                    self.g.add((self._uri(row["name"]), EX["mitigates_cause"], self._uri(row["mitigates_cause"])))
        except: pass

        if self.debug: print(f"KG populated with {len(self.g)} triples.")
        return self

    def procedures_for_cause(self, cause_str: str):
        safe_cause = cause_str.replace(' ', '_')
        q = f"PREFIX ex: <{EX}> SELECT ?proc WHERE {{ ?proc ex:mitigates_cause ex:{safe_cause} . }}"
        return [str(row.proc).split("#")[-1] for row in self.g.query(q)]

    def components_for_cause(self, cause_str: str):
        safe_cause = cause_str.replace(' ', '_')
        q = f"PREFIX ex: <{EX}> SELECT ?comp WHERE {{ ex:{safe_cause} ex:affects ?comp }}"
        return [str(row.comp).split("#")[-1] for row in self.g.query(q)]

    def symptoms_for_cause(self, cause_str: str):
        safe_cause = cause_str.replace(' ', '_')
        q = f"PREFIX ex: <{EX}> SELECT ?sym WHERE {{ ex:{safe_cause} ex:hasSymptom ?sym }}"
        return [str(row.sym).split("#")[-1] for row in self.g.query(q)]

import os