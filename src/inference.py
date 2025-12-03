import torch
import yaml
import pandas as pd
import numpy as np
import os
import sys
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

# Ajout du chemin src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloader import ParquetDataset # On réutilise le Dataset, mais pas le Loader entier
from model_new import build_model     # On charge MMDIN
from utils import set_seed

# ========================
# 1. Collator Spécial Inférence
# ========================
class InferenceCollator:
    """
    Similaire au BatchCollator de dataloader.py, mais ne cherche pas de 'label'.
    """
    def __init__(self, max_len, column_index, item_info_path):
        self.max_len = max_len
        self.column_index = column_index
        print(f"Chargement des embeddings items pour l'inférence...")
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

        # Gestion des embeddings items
        item_ids = batch_dict["item_id"].numpy()
        
        # Récupération sécurisée des embeddings
        # Si un item du test n'est pas dans item_info, on met des zéros (fallback)
        try:
            batch_item_info = self.item_info.loc[item_ids]
            emb_vals = np.stack(batch_item_info["item_emb_d128"].values)
        except KeyError:
            # Fallback robuste pour items inconnus
            valid_mask = self.item_info.index.isin(item_ids)
            # On ne peut pas facilement gérer les manquants ici sans ralentir, 
            # on suppose que item_info est complet pour le challenge.
            # En cas de crash ici, vérifiez que item_info.parquet couvre tout.
            batch_item_info = self.item_info.loc[item_ids] # Retry or fail
            emb_vals = np.stack(batch_item_info["item_emb_d128"].values)

        batch_dict["item_emb_d128"] = torch.tensor(emb_vals, dtype=torch.float32)

        # Gestion historique
        if "item_seq" in batch_dict:
            seq = batch_dict["item_seq"]
            if seq.shape[1] > self.max_len:
                seq = seq[:, -self.max_len:]
            batch_dict["item_seq"] = seq.long()

        # PAS DE LABEL EN TEST
        return batch_dict

# ========================
# 2. Configuration
# ========================
config_path = "../config/tabtransformer_config.yaml"
if not os.path.exists(config_path):
    config_path = "config/tabtransformer_config.yaml"

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

dataset_id = cfg["dataset_id"]
dataset_cfg = cfg["dataset_config"][dataset_id]
exp_id = cfg.get("base_expid", "TabTransformer_default")
model_cfg = cfg[exp_id]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Inference on device: {device}")

# ========================
# 3. Chargement Modèle & Data
# ========================
# A. Modèle
print("Construction du modèle...")
model = build_model(None, model_cfg)

# Chargement des poids (Best Model)
checkpoint_path = "../checkpoints/MMDIN_best.pth"
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Le fichier {checkpoint_path} n'existe pas. Entraînez d'abord le modèle !")

print(f"Chargement des poids depuis {checkpoint_path}...")
state_dict = torch.load(checkpoint_path, map_location=device)
# Si le modèle a été sauvegardé avec DataParallel, les clés commencent par 'module.'
if list(state_dict.keys())[0].startswith('module.'):
    state_dict = {k[7:]: v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.to(device)
model.eval()

# B. DataLoader Test (Manuel pour utiliser InferenceCollator)
test_path = dataset_cfg["test_data"]
if not test_path.endswith(".parquet"): test_path += ".parquet"

print(f"Chargement des données de test : {test_path}")
test_dataset = ParquetDataset(test_path)
batch_size = 16384 # Gros batch pour aller vite en inférence

collator = InferenceCollator(
    max_len=int(model_cfg.get("max_len", 20)),
    column_index=test_dataset.column_index,
    item_info_path=dataset_cfg["item_info"]
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=collator
)

# ========================
# 4. Boucle de Prédiction
# ========================
print("Démarrage des prédictions...")
all_preds = []

with torch.no_grad():
    for batch_dict in tqdm(test_loader, desc="Predicting"):
        # Envoi sur GPU
        for k, v in batch_dict.items():
            batch_dict[k] = v.to(device)
        
        # Prédiction
        y_pred = model(batch_dict)
        all_preds.append(y_pred.cpu().numpy())

# Concaténation
predictions = np.concatenate(all_preds)

# ========================
# 5. Création du CSV Submission
# ========================
print("Création du fichier submission...")

# Création du DataFrame
sub = pd.DataFrame()
sub['ID'] = range(len(predictions)) # 0 à N-1
sub['Task2'] = predictions          # Probabilités

# Vérification du format
print(sub.head())
print(f"Nombre de lignes : {len(sub)}")

# Sauvegarde
output_file = "prediction.csv"
sub.to_csv(output_file, index=False)
print(f"✅ Fichier '{output_file}' créé avec succès !")

# Optionnel : Créer un fichier ZIP pour Codabench
import zipfile
zip_name = "submission.zip"
with zipfile.ZipFile(zip_name, 'w') as zf:
    zf.write(output_file)
print(f"✅ Archive '{zip_name}' prête à être soumise !")