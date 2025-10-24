RISK_WEIGHTS = {"low": 1.0, "medium": 1.2, "high": 1.5}

def expected_cost_for_action(p_cause, action_map, miss_cost):
    """
    Calcula custo esperado de cada ação com base em probabilidades e custos.
    """
    ec = {}
    # não agir
    ec["none"] = sum(p_cause[c]*miss_cost.get(c, 0) for c in p_cause)
    # ações específicas
    for c, p in p_cause.items():
        if c not in action_map: continue
        for a in action_map[c]:
            base = a.get("cost", 0) or 0
            risk_w = RISK_WEIGHTS.get(a.get("risk", "low"), 1.0)
            do_cost = base * risk_w
            miss_others = sum(p_cause[c2]*miss_cost.get(c2,0) for c2 in p_cause if c2 != c)
            ec[a["name"]] = p*do_cost + miss_others
    return ec

def choose_action(ec):
    """Escolhe a ação com custo esperado mínimo."""
    best = min(ec, key=ec.get)
    return best, ec[best]
