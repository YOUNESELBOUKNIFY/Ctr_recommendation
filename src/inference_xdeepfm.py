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

# Ajout du chemin src pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloader import ParquetDataset
# IMPORTANT : On importe le constructeur du modèle xDeepFM
from model_xdeepfm import build_model
from utils import set_seed

# ========================
# Inference Collator
# ========================
class InferenceCollator:
    """
    Collator spécifique pour l'inférence (pas de label, gestion des séquences).
    """
    def __init__(self, max_len, column_index, item_info_path):
        self.max_len = max_len
        self.column_index = column_index
        print("📥 Chargement des embeddings items (item_info)...")
        self.item_info = pd.read_parquet(item_info_path).set_index("item_id")
        
    def __call__(self, batch):
        batch_tensor = default_collate(batch)
        batch_dict = {}
        
        # Reconstruction du dictionnaire de features
        for col, idx in self.column_index.items():
            if isinstance(idx, list):
                batch_dict[col] = batch_tensor[:, idx]
            else:
                batch_dict[col] = batch_tensor[:, idx].squeeze(-1)

        # Récupération des embeddings multimodaux (Image 128d)
        item_ids = batch_dict["item_id"].numpy()
        try:
            # .reindex est plus robuste que .loc pour les IDs manquants éventuels
            batch_item_info = self.item_info.reindex(item_ids).fillna(0)
            emb_vals = np.stack(batch_item_info["item_emb_d128"].values)
        except Exception as e:
            print(f"⚠️ Warning batch items: {e}")
            emb_vals = np.zeros((len(item_ids), 128))

        batch_dict["item_emb_d128"] = torch.tensor(emb_vals, dtype=torch.float32)

        # Gestion de l'historique séquentiel
        if "item_seq" in batch_dict:
            seq = batch_dict["item_seq"]
            # On tronque à max_len (ex: 20) pour correspondre à l'entraînement
            if seq.shape[1] > self.max_len:
                seq = seq[:, -self.max_len:]
            batch_dict["item_seq"] = seq.long()

        return batch_dict

# ========================
# 1. Configuration
# ========================
config_path = "../config/xdeepfm_config.yaml"
# Fallback si lancé depuis la racine
if not os.path.exists(config_path): 
    config_path = "config/xdeepfm_config.yaml"

if not os.path.exists(config_path):
    raise FileNotFoundError(f"❌ Fichier de config introuvable : {config_path}")

print(f"⚙️  Chargement de la configuration : {config_path}")
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

dataset_id = cfg["dataset_id"]
dataset_cfg = cfg["dataset_config"][dataset_id]
# Récupération de la section spécifique MM_xDeepFM_Run
model_cfg = cfg[cfg["base_expid"]]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Inférence xDeepFM sur : {device}")

# ========================
# 2. Construction du Modèle
# ========================
print("🏗️  Reconstruction de l'architecture xDeepFM (CIN + DNN)...")
model = build_model(None, model_cfg)

# Chemin du checkpoint sauvegardé par train_xdeepfm.py
checkpoint_path = "../checkpoints/xDeepFM_best.pth"
if not os.path.exists(checkpoint_path):
    checkpoint_path = "checkpoints/xDeepFM_best.pth"

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"❌ Checkpoint manquant : {checkpoint_path}. Avez-vous lancé l'entraînement ?")

print(f"📥 Chargement des poids depuis : {checkpoint_path}")
state_dict = torch.load(checkpoint_path, map_location=device)

# Nettoyage des clés DataParallel (suppression du préfixe 'module.')
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

# ========================
# 3. Préparation des Données Test
# ========================
test_path = dataset_cfg["test_data"]
if not test_path.endswith(".parquet"): test_path += ".parquet"

print(f"📂 Lecture du fichier test : {test_path}")
test_dataset = ParquetDataset(test_path)

collator = InferenceCollator(
    max_len=int(model_cfg.get("max_len", 20)),
    column_index=test_dataset.column_index,
    item_info_path=dataset_cfg["item_info"]
)

# Batch size large pour accélérer l'inférence
infer_batch_size = 8192

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=infer_batch_size,
    shuffle=False, # Important : ne jamais mélanger le test
    num_workers=4,
    collate_fn=collator
)

# ========================
# 4. Boucle de Prédiction
# ========================
print(f"🔮 Démarrage des prédictions ({len(test_dataset)} échantillons)...")
all_preds = []

with torch.no_grad():
    for batch_dict in tqdm(test_loader, desc="Progression"):
        # Transfert GPU
        for k, v in batch_dict.items():
            batch_dict[k] = v.to(device)
        
        # Forward Pass
        y_pred = model(batch_dict)
        
        # Stockage CPU
        all_preds.append(y_pred.cpu().numpy())

# Fusion des batchs
predictions = np.concatenate(all_preds)

# ========================
# 5. Export des Résultats
# ========================
print("📝 Génération des fichiers de soumission...")
sub = pd.DataFrame()
sub['ID'] = range(len(predictions))
sub['Task2'] = predictions

csv_name = "prediction_xdeepfm.csv"
zip_name = "submission_xdeepfm.zip"

# Sauvegarde CSV
sub.to_csv(csv_name, index=False)
print(f"✅ Fichier CSV créé : {csv_name}")

# Compression ZIP (Requis par Codabench)
with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(csv_name)

print(f"🎉 Terminé ! Fichier prêt à soumettre : {zip_name}")