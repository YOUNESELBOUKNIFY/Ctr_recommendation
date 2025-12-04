import torch
import yaml
import numpy as np
import os
import sys
import time

# Ajout du chemin src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloader import MMCTRDataLoader
from model_fibinet import build_model  # Charge FiBiNET
from utils import set_seed, compute_auc

# ========================
# 1. Config
# ========================
config_path = "../config/fibinet_config.yaml"
if not os.path.exists(config_path):
    config_path = "config/fibinet_config.yaml"

print(f"⚙️  Chargement config FiBiNET : {config_path}")
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

dataset_id = cfg["dataset_id"]
dataset_cfg = cfg["dataset_config"][dataset_id]
model_cfg = cfg[cfg["base_expid"]]

# ========================
# 2. Setup
# ========================
set_seed(model_cfg.get("seed", 2025))
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Training FiBiNET sur : {device}")

# ========================
# 3. DataLoaders
# ========================
batch_size = int(model_cfg.get("batch_size", 4096))
max_len = int(model_cfg.get("max_len", 20))

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
print("🏗️  Construction du modèle MM-FiBiNET...")
model = build_model(None, model_cfg)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

model.to(device)

lr = float(model_cfg.get("learning_rate", 1e-3))
weight_decay = float(model_cfg.get("weight_decay", 1e-5))
epochs = int(model_cfg.get("epochs", 30))

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_fn = torch.nn.BCELoss()

# --- SCHEDULER OneCycleLR ---
# Permet de booster la convergence et sortir des minimas locaux
steps_per_epoch = len(train_loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr * 10, # Monte jusqu'à 10x le LR de base
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    pct_start=0.3,  # 30% du temps pour monter, 70% pour descendre
    div_factor=25.0,
    final_div_factor=1000.0
)

# ========================
# 5. Entraînement
# ========================
best_auc = 0.0
os.makedirs("../checkpoints", exist_ok=True)
best_model_path = "../checkpoints/FiBiNET_best.pth"

print(f"\n🚀 Démarrage (FiBiNET)...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    steps = 0
    
    for batch_dict, labels in train_loader:
        for k, v in batch_dict.items():
            batch_dict[k] = v.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        y_pred = model(batch_dict)
        loss = loss_fn(y_pred, labels)
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        scheduler.step() # Mise à jour LR à chaque step
        
        total_loss += loss.item()
        steps += 1
        
        if steps % 200 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1} | Step {steps} | Loss: {loss.item():.4f} | LR: {current_lr:.6f}")

    avg_loss = total_loss / steps if steps > 0 else 0
    
    # Validation
    model.eval()
    y_trues, y_preds = [], []
    with torch.no_grad():
        for batch_dict, labels in valid_loader:
            for k, v in batch_dict.items():
                batch_dict[k] = v.to(device)
            y_pred = model(batch_dict)
            y_trues.append(labels.cpu().numpy())
            y_preds.append(y_pred.cpu().numpy())

    if len(y_trues) > 0:
        auc = compute_auc(np.concatenate(y_trues), np.concatenate(y_preds))
        print(f"✅ Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Valid AUC: {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state_dict, best_model_path)
            print(f"🏆 Nouveau record FiBiNET! Sauvegardé dans {best_model_path}")

print(f"Fin. Meilleur AUC : {best_auc:.4f}")