import torch
import yaml
import numpy as np
import os
import sys

# Ajout du chemin src au path si nécessaire
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloader import MMCTRDataLoader
from model import build_model
from utils import set_seed, compute_auc, compute_logloss

# ========================
# 1. Charger config YAML
# ========================
config_path = "../config/tabtransformer_config.yaml"

# Vérification du fichier
if not os.path.exists(config_path):
    # Fallback si on lance depuis src/
    config_path = "config/tabtransformer_config.yaml"

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

# Seed
seed = cfg.get("base_config", {}).get("seed", 2025)
set_seed(seed)

dataset_id = cfg["dataset_id"]
dataset_cfg = cfg["dataset_config"][dataset_id]
exp_id = cfg.get("base_expid", "TabTransformer_default")
model_cfg = cfg[exp_id]

# ========================
# 2. Device et multi-GPU
# ========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ========================
# 3. DataLoader
# ========================
print("Initialisation des DataLoaders...")
train_loader = MMCTRDataLoader(
    feature_map=model_cfg.get("feature_map"),
    data_path=dataset_cfg["train_data"],
    item_info_path=dataset_cfg["item_info"],
    batch_size=int(model_cfg.get("batch_size", 8192)),
    shuffle=True,
    num_workers=4,
    max_len=int(model_cfg.get("max_len", 5))
)

valid_loader = MMCTRDataLoader(
    feature_map=model_cfg.get("feature_map"),
    data_path=dataset_cfg["valid_data"],
    item_info_path=dataset_cfg["item_info"],
    batch_size=int(model_cfg.get("batch_size", 8192)),
    shuffle=False,
    num_workers=4,
    max_len=int(model_cfg.get("max_len", 5))
)

# ========================
# 4. Modèle
# ========================
print("Construction du modèle...")
# On passe None pour feature_map car notre modèle corrigé gère ça en dur pour l'instant
model = build_model(None, model_cfg)

multi_gpu = torch.cuda.device_count() > 1
if multi_gpu:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

model.to(device)

# Optimizer et loss
learning_rate = float(model_cfg.get("learning_rate", 1e-3))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.BCELoss() # Car le modèle sort déjà une Sigmoid

# Répertoire checkpoints
os.makedirs("../checkpoints", exist_ok=True)

# ========================
# 5. Entraînement
# ========================
epochs = int(model_cfg.get("epochs", 10))
print(f"Début de l'entraînement pour {epochs} époques.")

best_auc = 0.0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    steps = 0
    
    # NOTE: Le dataloader corrigé renvoie (batch_dict, labels)
    for batch_dict, labels in train_loader:
        
        # Transfert vers GPU
        for k, v in batch_dict.items():
            batch_dict[k] = v.to(device)
        
        labels = labels.to(device)

        # Forward
        optimizer.zero_grad()
        
        # Le modèle prend le dict complet
        y_pred = model(batch_dict) 
        
        # Calcul de la loss
        loss = loss_fn(y_pred, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        steps += 1
        
        if steps % 100 == 0:
            print(f"Step {steps}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / steps if steps > 0 else 0
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}")

    # ========================
    # Validation
    # ========================
    model.eval()
    y_trues, y_preds = [], []
    
    with torch.no_grad():
        for batch_dict, labels in valid_loader:
            
            # Transfert vers GPU
            for k, v in batch_dict.items():
                batch_dict[k] = v.to(device)
            labels = labels.to(device)

            # Forward
            y_pred = model(batch_dict)

            y_trues.append(labels.cpu().numpy())
            y_preds.append(y_pred.cpu().numpy())

    # Concaténation des résultats
    if len(y_trues) > 0:
        y_trues = np.concatenate(y_trues)
        y_preds = np.concatenate(y_preds)
        
        auc = compute_auc(y_trues, y_preds)
        logloss = compute_logloss(y_trues, y_preds)
        print(f"Epoch {epoch+1}: Valid AUC={auc:.4f}, LogLoss={logloss:.4f}")

        # Sauvegarde du meilleur modèle
        if auc > best_auc:
            best_auc = auc
            checkpoint_path = "../checkpoints/TabTransformer_best.pth"
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state_dict, checkpoint_path)
            print(f"New Best AUC! Model saved in {checkpoint_path}")
    else:
        print("Warning: Validation set empty or error.")

print("Entraînement terminé.")