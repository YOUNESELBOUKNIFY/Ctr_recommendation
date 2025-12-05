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

# Ajout du chemin src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloader import ParquetDataset
# IMPORTANT : On charge le modèle FiBiNET++
from model_fibinet_plus import build_model
from utils import set_seed

# ========================
# Inference Collator
# ========================
class InferenceCollator:
    def __init__(self, max_len, column_index, item_info_path):
        self.max_len = max_len
        self.column_index = column_index
        print("📥 Chargement item_info pour l'inférence...")
        self.item_info = pd.read_parquet(item_info_path).set_index("item_id")
        
    def __call__(self, batch):
        batch_tensor = default_collate(batch)
        batch_dict = {}
        
        # Reconstruction du dictionnaire
        for col, idx in self.column_index.items():
            if isinstance(idx, list):
                batch_dict[col] = batch_tensor[:, idx]
            else:
                batch_dict[col] = batch_tensor[:, idx].squeeze(-1)

        # Récupération des embeddings (Image 128d)
        item_ids = batch_dict["item_id"].numpy()
        try:
            # Fallback robuste (remplit les items inconnus par 0)
            batch_item_info = self.item_info.reindex(item_ids).fillna(0)
            emb_vals = np.stack(batch_item_info["item_emb_d128"].values)
        except Exception as e:
            # Fallback ultime
            emb_vals = np.zeros((len(item_ids), 128))

        batch_dict["item_emb_d128"] = torch.tensor(emb_vals, dtype=torch.float32)

        # Gestion de l'historique
        if "item_seq" in batch_dict:
            seq = batch_dict["item_seq"]
            if seq.shape[1] > self.max_len:
                seq = seq[:, -self.max_len:]
            batch_dict["item_seq"] = seq.long()

        return batch_dict

# ========================
# 1. Config & Load
# ========================
config_path = "../config/fibinet_plus_config.yaml"
# Fallback path
if not os.path.exists(config_path): 
    config_path = "config/fibinet_plus_config.yaml"

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

dataset_id = cfg["dataset_id"]
dataset_cfg = cfg["dataset_config"][dataset_id]
model_cfg = cfg[cfg["base_expid"]]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Inférence FiBiNET++ sur : {device}")

# ========================
# 2. Construction Modèle
# ========================
print("🏗️  Reconstruction de l'architecture FiBiNET++...")
model = build_model(None, model_cfg)

checkpoint_path = "../checkpoints/FiBiNET_Plus_best.pth"
if not os.path.exists(checkpoint_path): 
    # Fallback local
    checkpoint_path = "checkpoints/FiBiNET_Plus_best.pth"

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"❌ Checkpoint introuvable : {checkpoint_path}")

print(f"📥 Chargement des poids : {checkpoint_path}")
state_dict = torch.load(checkpoint_path, map_location=device)

# Correction DataParallel (suppression du préfixe module.)
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)

model.to(device)
model.eval()

# ========================
# 3. Data Test
# ========================
test_path = dataset_cfg["test_data"]
test_dataset = ParquetDataset(test_path)

collator = InferenceCollator(
    max_len=int(model_cfg.get("max_len", 20)),
    column_index=test_dataset.column_index,
    item_info_path=dataset_cfg["item_info"]
)

# Batch size large pour aller vite
infer_batch_size = 8192

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=infer_batch_size,
    shuffle=False, # Ne jamais mélanger le test
    num_workers=4,
    collate_fn=collator
)

# ========================
# 4. Prediction
# ========================
print(f"🔮 Démarrage des prédictions ({len(test_dataset)} échantillons)...")
all_preds = []

with torch.no_grad():
    for batch_dict in tqdm(test_loader):
        for k, v in batch_dict.items():
            batch_dict[k] = v.to(device)
        y_pred = model(batch_dict)
        all_preds.append(y_pred.cpu().numpy())

predictions = np.concatenate(all_preds)

# ========================
# 5. Export
# ========================
print("📝 Génération des fichiers...")
sub = pd.DataFrame()
sub['ID'] = range(len(predictions))
sub['Task2'] = predictions

csv_name = "prediction_fibinet_plus.csv"
zip_name = "submission_fibinet_plus.zip"

sub.to_csv(csv_name, index=False)

with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(csv_name)

print(f"✅ Terminé ! Fichier prêt : {zip_name}")