import yaml
from src.integration import run_pipeline

with open("configs/settings.yaml") as f:
    cfg = yaml.safe_load(f)

out = run_pipeline(
    paths=cfg["paths"],
    continuous_cols=cfg["continuous_cols"],
    evidence=cfg["evidence"]
)

print("\n=== Posterior Probabilities ===")
for c, p in out["posterior"].items():
    print(f"{c}: {p:.2f}")

print("\n=== Expected Costs ===")
for a, c in out["expected_costs"].items():
    print(f"{a}: {c:.2f}")

print("\n=== Recommended Action ===")
print(out["recommendation"])
