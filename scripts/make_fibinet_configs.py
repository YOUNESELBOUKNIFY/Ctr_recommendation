import os
import copy
import yaml

BASE_YAML = "../config/fibinet_config.yaml"   # adapte si besoin
OUT_DIR = "../configs_ensemble"

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d

def save_yaml(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)

os.makedirs(OUT_DIR, exist_ok=True)

with open(BASE_YAML, "r", encoding="utf-8") as f:
    base = yaml.safe_load(f)

variants = [
    {"name": "s2025", "override": {"base_config": {"seed": 2025}}},
    {"name": "s2026", "override": {"base_config": {"seed": 2026}}},
    {"name": "s2027", "override": {"base_config": {"seed": 2027}}},
    {"name": "do020", "override": {"MM_FiBiNET_Run": {"net_dropout": 0.20}}},
    {"name": "do030", "override": {"MM_FiBiNET_Run": {"net_dropout": 0.30}}},
    {"name": "sr3",   "override": {"MM_FiBiNET_Run": {"senet_reduction": 3}}},
    {"name": "sr4",   "override": {"MM_FiBiNET_Run": {"senet_reduction": 4}}},
    {"name": "bil_all", "override": {"MM_FiBiNET_Run": {"bilinear_type": "all"}}},
    {"name": "bil_each", "override": {"MM_FiBiNET_Run": {"bilinear_type": "each"}}},
    {"name": "lr0008", "override": {"MM_FiBiNET_Run": {"learning_rate": 0.0008}}},
    {"name": "wd3e5",  "override": {"MM_FiBiNET_Run": {"weight_decay": 3e-5}}},
]

for v in variants:
    cfg = copy.deepcopy(base)
    expid = f"{cfg['base_expid']}_{v['name']}"
    cfg["base_expid"] = expid
    deep_update(cfg, v["override"])

    out_path = os.path.join(OUT_DIR, f"{expid}.yaml")
    save_yaml(out_path, cfg)

print("✅ configs générées dans:", OUT_DIR)
