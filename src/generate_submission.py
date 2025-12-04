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

# Ajout du chemin src pour importer vos modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloader import ParquetDataset
from model_new import build_model  # Votre modèle MMDIN final
from utils import set_seed

# ==========================================
# 1. Collator Spécial pour l'Inférence
# ==========================================
class InferenceCollator:
    """
    Prépare les batchs pour le test.
    Différence majeure avec le train : NE CHERCHE PAS DE LABEL.
    """
    def __init__(self, max_len, column_index, item_info_path):
        self.max_len = max_len
        self.column_index = column_index
        print(f"📥 Chargement des embeddings items pour l'inférence...")
        # Chargement optimisé en mémoire
        self.item_info = pd.read_parquet(item_info_path).set_index("item_id")
        
    def __call__(self, batch):
        batch_tensor = default_collate(batch)
        batch_dict = {}
        
        # Reconstruction du dictionnaire à partir des indices colonnes
        for col, idx in self.column_index.items():
            if isinstance(idx, list):
                batch_dict[col] = batch_tensor[:, idx]
            else:
                batch_dict[col] = batch_tensor[:, idx].squeeze(-1)

        # Récupération des embeddings multimodaux (Image 128d)
        item_ids = batch_dict["item_id"].numpy()
        
        # Recherche sécurisée (Fallback à 0 si item inconnu, rare mais possible)
        try:
            # .reindex est plus rapide et sûr que .loc pour les batchs
            batch_item_info = self.item_info.reindex(item_ids).fillna(0)
            
            # Extraction des embeddings (suppose que la colonne contient des listes/arrays)
            # Si c'est lent, on peut pré-calculer une matrice numpy géante, 
            # mais reindex est suffisant pour l'inférence.
            emb_vals = np.stack(batch_item_info["item_emb_d128"].values)
            
        except Exception as e:
            print(f"Warning sur le batch : {e}")
            emb_vals = np.zeros((len(item_ids), 128)) # Fallback zéro

        batch_dict["item_emb_d128"] = torch.tensor(emb_vals, dtype=torch.float32)

        # Gestion de l'historique (Sequence)
        if "item_seq" in batch_dict:
            seq = batch_dict["item_seq"]
            # Troncature si l'historique est trop long pour le modèle
            if seq.shape[1] > self.max_len:
                seq = seq[:, -self.max_len:]
            batch_dict["item_seq"] = seq.long()

        return batch_dict

# ==========================================
# 2. Configuration & Initialisation
# ==========================================
config_path = "../config/din_conf.yaml"
# Fallback path
if not os.path.exists(config_path): config_path = "config/din_conf.yaml"

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

# Récupération des paramètres
dataset_id = cfg["dataset_id"]
dataset_cfg = cfg["dataset_config"][dataset_id]
base_exp_id = cfg.get("base_expid", "MMDIN_Optimized_Run")
model_cfg = cfg[base_exp_id]

# Si un fichier best_hyperparams existe, on peut vouloir l'utiliser, 
# mais pour l'inférence, on suppose que le YAML est à jour ou que le .pth contient la structure.
# Pour simplifier, on utilise model_cfg du YAML.

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Inférence sur : {device}")

# ==========================================
# 3. Chargement du Modèle et des Poids
# ==========================================
print("🏗️  Reconstruction de l'architecture MMDIN...")
model = build_model(None, model_cfg)

# Chemin vers le meilleur checkpoint
checkpoint_path = "../checkpoints/MMDIN_best.pth"
if not os.path.exists(checkpoint_path):
    # Essayons le chemin local
    checkpoint_path = "checkpoints/MMDIN_best.pth"

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"❌ Impossible de trouver le modèle : {checkpoint_path}")

print(f"📥 Chargement des poids depuis : {checkpoint_path}")
state_dict = torch.load(checkpoint_path, map_location=device)

# --- CORRECTION DATAPARALLEL ---
# Si le modèle a été entraîné sur plusieurs GPU, les clés ont un préfixe "module."
# Il faut le retirer pour charger sur un modèle simple ou sur un seul GPU.
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v # Enlève 'module.'
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict)
model.to(device)
model.eval() # Mode évaluation (désactive dropout)

# ==========================================
# 4. Préparation des Données Test
# ==========================================
test_path = dataset_cfg["test_data"]
if not test_path.endswith(".parquet"): test_path += ".parquet"

print(f"📂 Lecture du fichier test : {test_path}")
test_dataset = ParquetDataset(test_path)

# Batch size large pour l'inférence (plus rapide)
infer_batch_size = 16384 

collator = InferenceCollator(
    max_len=int(model_cfg.get("max_len", 20)),
    column_index=test_dataset.column_index,
    item_info_path=dataset_cfg["item_info"]
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=infer_batch_size,
    shuffle=False, # IMPORTANT : Ne jamais mélanger le test !
    num_workers=4,
    collate_fn=collator
)

# ==========================================
# 5. Boucle de Prédiction
# ==========================================
print("🔮 Démarrage des prédictions...")
all_preds = []

with torch.no_grad():
    for batch_dict in tqdm(test_loader, desc="Progression"):
        # Envoi sur GPU
        for k, v in batch_dict.items():
            batch_dict[k] = v.to(device)
        
        # Forward MMDIN
        y_pred = model(batch_dict)
        
        # Stockage CPU
        all_preds.append(y_pred.cpu().numpy())

# Fusion des résultats
predictions = np.concatenate(all_preds)

# ==========================================
# 6. Création du CSV et ZIP
# ==========================================
print(f"📝 Génération du fichier CSV ({len(predictions)} lignes)...")

# Format demandé par la compétition : ID (0 à N-1) et Task2 (Score)
sub = pd.DataFrame()
sub['ID'] = range(len(predictions))
sub['Task2'] = predictions

csv_name = "prediction.csv"
sub.to_csv(csv_name, index=False)
print(f"✅ Fichier '{csv_name}' créé.")

# Création du ZIP (Souvent requis par Codabench)
zip_name = "submission.zip"
print(f"📦 Compression dans '{zip_name}'...")
with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(csv_name)

print("\n🎉 TERMINÉ ! Vous pouvez soumettre 'submission.zip' sur le leaderboard.")