import torch
import yaml
import numpy as np
import os
import sys
import time

# Ajout du chemin src au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloader import MMCTRDataLoader
from model_new import build_model  # Charge le modèle MMDIN final
from utils import set_seed, compute_auc, compute_logloss

# ========================
# 1. Chargement de la Configuration
# ========================
config_path = "../config/din_conf.yaml"
if not os.path.exists(config_path):
    config_path = "config/din_conf.yaml"

print(f"📂 Chargement de la configuration depuis : {config_path}")
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

# Récupération des sections
dataset_id = cfg["dataset_id"]
dataset_cfg = cfg["dataset_config"][dataset_id]
base_exp_id = cfg.get("base_expid", "MMDIN_Optimized_Run")
model_cfg = cfg[base_exp_id]

# --- AUTO-OVERRIDE : Chargement des meilleurs hyperparamètres (si dispo) ---
best_params_path = "best_hyperparams_mmdin.yaml"
if os.path.exists(best_params_path):
    print(f"✨ FICHIER DE TUNING DÉTECTÉ : {best_params_path}")
    print("   -> Mise à jour automatique des hyperparamètres...")
    with open(best_params_path, "r") as f:
        best_params = yaml.safe_load(f)
    
    # Mise à jour de model_cfg
    for k, v in best_params.items():
        # Gestion spécifique pour les listes (ex: attention units stocké en string par yaml parfois)
        if k == "attention_hidden_units" and isinstance(v, str):
            v = [int(x) for x in v.split(',')]
        
        model_cfg[k] = v
        print(f"   * {k}: {v}")
else:
    print("ℹ️ Aucun fichier de tuning trouvé, utilisation des paramètres du YAML.")

# ========================
# 2. Initialisation
# ========================
# Seed
seed = model_cfg.get("seed", 2025)
set_seed(seed)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Training sur : {device}")

# ========================
# 3. DataLoaders
# ========================
batch_size = int(model_cfg.get("batch_size", 4096))
max_len = int(model_cfg.get("max_len", 20))

print(f"📦 Création des DataLoaders (Batch: {batch_size}, Hist Len: {max_len})...")

train_loader = MMCTRDataLoader(
    feature_map=None,
    data_path=dataset_cfg["train_data"],
    item_info_path=dataset_cfg["item_info"],
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    max_len=max_len
)

valid_loader = MMCTRDataLoader(
    feature_map=None,
    data_path=dataset_cfg["valid_data"],
    item_info_path=dataset_cfg["item_info"],
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    max_len=max_len
)

# ========================
# 4. Modèle & Optimiseur
# ========================
print("🏗️  Construction du modèle MMDIN...")
model = build_model(None, model_cfg)

# Multi-GPU
if torch.cuda.device_count() > 1:
    print(f"🚀 Multi-GPU activé : {torch.cuda.device_count()} GPUs.")
    model = torch.nn.DataParallel(model)

model.to(device)

learning_rate = float(model_cfg.get("learning_rate", 1e-3))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.BCELoss()

# ========================
# 5. Boucle d'Entraînement
# ========================
epochs = int(model_cfg.get("epochs", 20))
best_auc = 0.0
checkpoint_dir = "../checkpoints" if os.path.exists("../checkpoints") else "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
best_model_path = os.path.join(checkpoint_dir, "MMDIN_best.pth")

print(f"\n🔥 Démarrage de l'entraînement pour {epochs} époques...")
start_time = time.time()

for epoch in range(epochs):
    epoch_start = time.time()
    model.train()
    total_loss = 0
    steps = 0
    
    # --- TRAINING ---
    for batch_dict, labels in train_loader:
        # Transfert GPU
        for k, v in batch_dict.items():
            batch_dict[k] = v.to(device)
        labels = labels.to(device)

        # Forward & Backward
        optimizer.zero_grad()
        y_pred = model(batch_dict)
        loss = loss_fn(y_pred, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        steps += 1
        
        # Log fréquent (tous les 100 steps)
        if steps % 100 == 0:
            sys.stdout.write(f"\rEpoch {epoch+1} | Step {steps} | Loss: {loss.item():.4f}")
            sys.stdout.flush()

    avg_train_loss = total_loss / steps if steps > 0 else 0
    
    # --- VALIDATION ---
    model.eval()
    y_trues, y_preds = [], []
    
    with torch.no_grad():
        for batch_dict, labels in valid_loader:
            for k, v in batch_dict.items():
                batch_dict[k] = v.to(device)
            
            y_pred = model(batch_dict)
            y_trues.append(labels.cpu().numpy())
            y_preds.append(y_pred.cpu().numpy())

    # Calcul Métriques
    epoch_time = time.time() - epoch_start
    
    if len(y_trues) > 0:
        y_trues = np.concatenate(y_trues)
        y_preds = np.concatenate(y_preds)
        
        auc = compute_auc(y_trues, y_preds)
        logloss = compute_logloss(y_trues, y_preds)
        
        print(f"\n✅ Epoch {epoch+1}/{epochs} [{epoch_time:.0f}s] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Valid AUC: {auc:.4f} | LogLoss: {logloss:.4f}")

        # Sauvegarde du meilleur modèle
        if auc > best_auc:
            best_auc = auc
            # Gestion DataParallel pour la sauvegarde
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state_dict, best_model_path)
            print(f"🏆 Nouveau Record ! Modèle sauvegardé dans : {best_model_path}")
    else:
        print("\n⚠️  Attention : Le set de validation est vide ou a échoué.")

total_time = time.time() - start_time
print(f"\n🏁 Entraînement terminé en {total_time/60:.1f} minutes.")
print(f"Meilleur AUC Final : {best_auc:.4f}")