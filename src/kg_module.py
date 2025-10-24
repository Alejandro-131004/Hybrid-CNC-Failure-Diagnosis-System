from rdflib import Graph, Namespace, RDF, RDFS, Literal
import pandas as pd

EX = Namespace("http://example.org/cnc#")

def build_graph(components_csv, causes_csv, symptoms_csv, procedures_csv, relations_csv) -> Graph:
    """Constrói grafo RDF completo a partir dos CSVs."""
    g = Graph()
    g.bind("ex", EX)

    comp = pd.read_csv(components_csv)
    cau  = pd.read_csv(causes_csv)
    sym  = pd.read_csv(symptoms_csv)
    proc = pd.read_csv(procedures_csv)
    rel  = pd.read_csv(relations_csv)

    def uri(kind, id_): return EX[f"{kind}_{id_}"]

    # Adiciona classes e instâncias
    for _, r in comp.iterrows():
        g.add((uri("Component", r["id"]), RDF.type, EX.Component))
        g.add((uri("Component", r["id"]), RDFS.label, Literal(r["name"])))
    for _, r in cau.iterrows():
        g.add((uri("Cause", r["id"]), RDF.type, EX.Cause))
        g.add((uri("Cause", r["id"]), RDFS.label, Literal(r["name"])))
    for _, r in sym.iterrows():
        g.add((uri("Symptom", r["id"]), RDF.type, EX.Symptom))
        g.add((uri("Symptom", r["id"]), RDFS.label, Literal(r["name"])))
    for _, r in proc.iterrows():
        u = uri("Procedure", r["id"])
        g.add((u, RDF.type, EX.Procedure))
        g.add((u, RDFS.label, Literal(r["name"])))
        if "cost_eur" in r: g.add((u, EX.cost, Literal(float(r["cost_eur"]))))
        if "time_h"   in r: g.add((u, EX.timeH, Literal(float(r["time_h"]))))
        if "risk"     in r: g.add((u, EX.risk, Literal(str(r["risk"]))))

    for _, r in rel.iterrows():
        pred = getattr(EX, r["relation"])
        g.add((uri(r["head_type"], r["head_id"]), pred, uri(r["tail_type"], r["tail_id"])))
    return g

def procedures_for_cause(g: Graph, cause_label: str) -> list[dict]:
    """Consulta SPARQL: devolve procedimentos que resolvem uma causa."""
    q = """
    PREFIX ex: <http://example.org/cnc#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?proc ?name ?cost ?time ?risk WHERE {
      ?c a ex:Cause ; rdfs:label ?clab .
      FILTER(LCASE(STR(?clab)) = LCASE(?cause))
      ?c ex:fixedBy ?proc .
      OPTIONAL { ?proc rdfs:label ?name }
      OPTIONAL { ?proc ex:cost ?cost }
      OPTIONAL { ?proc ex:timeH ?time }
      OPTIONAL { ?proc ex:risk ?risk }
    }
    """
    res = g.query(q, initBindings={"cause": Literal(cause_label)})
    return [
        {"uri": str(r.proc), "name": str(r.name), "cost": float(r.cost) if r.cost else None,
         "time_h": float(r.time) if r.time else None, "risk": str(r.risk) if r.risk else None}
        for r in res
    ]
