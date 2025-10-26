"""
kg_module.py
Hybrid CNC Knowledge Graph Builder and Query Module
"""

from rdflib import Graph, Namespace, Literal, RDF, RDFS, URIRef
import pandas as pd
from .utils import load_csv


# ==============================================================
# === RDF Namespace Definition
# ==============================================================

EX = Namespace("http://hybridcnc.org/ontology#")

# The EX object defines an RDF namespace (a unique identifier prefix) for all entities
# and relations in the Knowledge Graph.
#
# In RDF and SPARQL, every concept — such as a cause, component, symptom, or procedure —
# must be represented by a globally unique IRI (Internationalized Resource Identifier).
# 
# By creating a Namespace like "http://hybridcnc.org/ontology#", we ensure that:
#   - Each entity (e.g., EX["BearingWearHigh"]) becomes a valid URIRef:
#       → <http://hybridcnc.org/ontology#BearingWearHigh>
#   - Each relation (e.g., EX["affects"]) is a named predicate:
#       → <http://hybridcnc.org/ontology#affects>
#   - All triples (subject–predicate–object) in the graph belong to the same domain context.
#
# Example:
#     self.g.add((EX["CloggedFilter"], EX["fixedBy"], EX["CleanFilter"]))
# expands to the RDF triple:
#     <http://hybridcnc.org/ontology#CloggedFilter>
#         <http://hybridcnc.org/ontology#fixedBy>
#         <http://hybridcnc.org/ontology#CleanFilter>
#
# This structure guarantees that the Knowledge Graph is semantically valid,
# SPARQL-compliant, and compatible with external ontologies or visualization tools.


# ==============================================================
# === CNC Knowledge Graph Class
# ==============================================================

class CNCKG:
    """
    Builds and queries a Knowledge Graph for CNC failure diagnosis.
    Integrates ontology classes and properties with instances from CSV data.
    """

    def __init__(self):
        self.g = Graph()
        self.g.bind("ex", EX)

    # ----------------------------------------------------------
    # Utility: create URIRef safely
    # ----------------------------------------------------------
    def _uri(self, s: str) -> URIRef:
        """Converts a string into a valid URIRef within the ontology namespace."""
        return EX[str(s).strip().replace(" ", "_")]

    # ----------------------------------------------------------
    # Load base ontology and add CSV-based instances
    # ----------------------------------------------------------
    def load_from_cfg(self, cfg, ontology_path: str = "ontology.ttl"):
        """
        Loads ontology.ttl and populates the Knowledge Graph with instances
        from CSV files (components, causes, symptoms, procedures, relations).
        """
        # 1. Parse the base ontology (classes + properties)
        try:
            self.g.parse(ontology_path, format="turtle")
            print(f"Ontology loaded: {ontology_path}")
        except Exception as e:
            print(f"Warning: could not parse ontology ({e})")

        # 2. Add entities from CSVs
        for key, cls in [
            ("components", "Component"),
            ("causes", "Cause"),
            ("symptoms", "Symptom"),
            ("procedures", "Procedure")
        ]:
            df = load_csv(cfg, key)
            name_col = "name" if "name" in df.columns else df.columns[0]
            for v in df[name_col].unique():
                self.g.add((self._uri(v), RDF.type, EX[cls]))
                self.g.add((self._uri(v), RDFS.label, Literal(v)))

        # 3. Add relations from relations.csv
        rel = load_csv(cfg, "relations")
        for _, r in rel.iterrows():
            subj = self._uri(r["subj"])
            pred = self._uri(r["pred"])
            obj = self._uri(r["obj"])
            self.g.add((subj, pred, obj))

        print(f"Knowledge Graph populated with {len(self.g)} triples.")
        return self

    # ----------------------------------------------------------
    # Query 1: Get procedure(s) that fix a cause
    # ----------------------------------------------------------
    def procedures_for_cause(self, cause_str: str):
        """Returns list of procedures that fix a given cause."""
        q = f"""
        PREFIX ex: <{EX}>
        SELECT ?proc WHERE {{
          ex:{cause_str.replace(' ', '_')} ex:fixedBy ?proc .
        }}
        """
        return [str(row.proc).split("#")[-1] for row in self.g.query(q)]

    # ----------------------------------------------------------
    # Query 2: Get component(s) affected by a cause
    # ----------------------------------------------------------
    def components_for_cause(self, cause_str: str):
        """Returns components affected by a given cause."""
        q = f"""
        PREFIX ex: <{EX}>
        SELECT ?comp WHERE {{
          ex:{cause_str.replace(' ', '_')} ex:affects ?comp .
        }}
        """
        return [str(row.comp).split("#")[-1] for row in self.g.query(q)]

    # ----------------------------------------------------------
    # Query 3: Get symptom(s) caused by a cause
    # ----------------------------------------------------------
    def symptoms_for_cause(self, cause_str: str):
        """Returns symptoms linked to a given cause."""
        q = f"""
        PREFIX ex: <{EX}>
        SELECT ?sym WHERE {{
          ex:{cause_str.replace(' ', '_')} ex:hasSymptom ?sym .
        }}
        """
        return [str(row.sym).split("#")[-1] for row in self.g.query(q)]

    # ----------------------------------------------------------
    # Query 4: Show all triples (optional)
    # ----------------------------------------------------------
    def show_triples(self, limit: int = 10):
        """Displays a subset of triples for inspection."""
        for i, triple in enumerate(self.g):
            print(triple)
            if i >= limit - 1:
                break
