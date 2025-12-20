import os
import glob
import subprocess
import yaml

CONFIG_DIR = "../configs_ensemble"
CKPT_DIR = "../checkpoints"
PRED_DIR = "../preds"

TRAIN_PY = "../src2/train_fibinet.py"
PRED_PY  = "../src2/predict_fibinet.py"

configs = sorted(glob.glob(os.path.join(CONFIG_DIR, "*.yaml")))
assert len(configs) > 0, f"Aucune config trouvée dans {CONFIG_DIR}/"

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

for cfg_path in configs:
    print("\n==============================")
    print("CONFIG:", cfg_path)
    print("==============================")

    # TRAIN
    subprocess.check_call(["python", TRAIN_PY, "--config", cfg_path, "--ckpt_dir", CKPT_DIR])

    # lire expid
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    expid = cfg["base_expid"]
    ckpt_path = os.path.join(CKPT_DIR, f"{expid}_best.pth")

    # PRED TEST
    subprocess.check_call([
        "python", PRED_PY,
        "--config", cfg_path,
        "--checkpoint", ckpt_path,
        "--split", "test",
        "--out_dir", PRED_DIR
    ])

print("\n✅ Tous les runs train + predict terminés.")
