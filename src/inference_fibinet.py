import torch
import yaml
import pandas as pd
import numpy as np
import os
import sys
import zipfile
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloader import ParquetDataset
from model_fibinet import build_model
from utils import set_seed

# ========================
# Inference Collator
# ========================
class InferenceCollator:
    def __init__(self, max_len, column_index, item_info_path):
        self.max_len = max_len
        self.column_index = column_index
        print("Chargement item_info...")
        self.item_info = pd.read_parquet(item_info_path).set_index("item_id")
        
    def __call__(self, batch):
        batch_tensor = default_collate(batch)
        batch_dict = {}
        for col, idx in self.column_index.items():
            if isinstance(idx, list):
                batch_dict[col] = batch_tensor[:, idx]
            else:
                batch_dict[col] = batch_tensor[:, idx].squeeze(-1)

        item_ids = batch_dict["item_id"].numpy()
        try:
            batch_item_info = self.item_info.reindex(item_ids).fillna(0)
            emb_vals = np.stack(batch_item_info["item_emb_d128"].values)
        except Exception:
            emb_vals = np.zeros((len(item_ids), 128))

        batch_dict["item_emb_d128"] = torch.tensor(emb_vals, dtype=torch.float32)

        if "item_seq" in batch_dict:
            seq = batch_dict["item_seq"]
            if seq.shape[1] > self.max_len:
                seq = seq[:, -self.max_len:]
            batch_dict["item_seq"] = seq.long()

        return batch_dict

# ========================
# Config & Load
# ========================
config_path = "../config/fibinet_config.yaml"
if not os.path.exists(config_path): config_path = "config/fibinet_config.yaml"

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

dataset_id = cfg["dataset_id"]
dataset_cfg = cfg["dataset_config"][dataset_id]
model_cfg = cfg[cfg["base_expid"]]

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Construction modèle FiBiNET...")
model = build_model(None, model_cfg)

checkpoint_path = "../checkpoints/FiBiNET_best.pth"
if not os.path.exists(checkpoint_path): checkpoint_path = "checkpoints/FiBiNET_best.pth"

print(f"Chargement poids : {checkpoint_path}")
state_dict = torch.load(checkpoint_path, map_location=device)
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)

model.to(device)
model.eval()

# ========================
# Data Test
# ========================
test_path = dataset_cfg["test_data"]
test_dataset = ParquetDataset(test_path)

collator = InferenceCollator(
    max_len=int(model_cfg.get("max_len", 20)),
    column_index=test_dataset.column_index,
    item_info_path=dataset_cfg["item_info"]
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=8192,
    shuffle=False,
    num_workers=4,
    collate_fn=collator
)

# ========================
# Predict
# ========================
print("Prédiction en cours...")
all_preds = []
with torch.no_grad():
    for batch_dict in tqdm(test_loader):
        for k, v in batch_dict.items():
            batch_dict[k] = v.to(device)
        y_pred = model(batch_dict)
        all_preds.append(y_pred.cpu().numpy())

predictions = np.concatenate(all_preds)

# ========================
# Export
# ========================
sub = pd.DataFrame()
sub['ID'] = range(len(predictions))
sub['Task2'] = predictions
sub.to_csv("prediction_fibinet.csv", index=False)

with zipfile.ZipFile("submission_fibinet.zip", 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write("prediction_fibinet.csv")

print("✅ Fichier 'submission_fibinet.zip' généré avec succès !")