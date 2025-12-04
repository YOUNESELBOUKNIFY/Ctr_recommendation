import torch
import yaml
import numpy as np
import os
import sys
import time

# Ajout du chemin src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloader import MMCTRDataLoader
from model_bst import build_model  # Charge le modèle BST-DCN
from utils import set_seed, compute_auc, compute_logloss

# ========================
# 1. Config
# ========================
config_path = "../config/bst_config.yaml"
if not os.path.exists(config_path):
    config_path = "config/bst_config.yaml"

print(f"⚙️  Chargement de la configuration BST : {config_path}")
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
print(f"🔥 Training BST sur : {device}")

# ========================
# 3. DataLoaders
# ========================
batch_size = int(model_cfg.get("batch_size", 4096))
max_len = int(model_cfg.get("max_len", 20))

print(f"📦 DataLoaders (Batch: {batch_size}, Seq Len: {max_len})")
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
print("🏗️  Construction du modèle MM-BST-DCN...")
model = build_model(None, model_cfg)

if torch.cuda.device_count() > 1:
    print(f"🚀 Multi-GPU : {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)

model.to(device)

lr = float(model_cfg.get("learning_rate", 5e-4))
weight_decay = float(model_cfg.get("weight_decay", 1e-4))

# AdamW est recommandé pour les Transformers
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
# Scheduler pour réduire le LR si plateau (Optionnel mais recommandé)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
loss_fn = torch.nn.BCELoss()

# ========================
# 5. Entraînement
# ========================
epochs = int(model_cfg.get("epochs", 30))
best_auc = 0.0
os.makedirs("../checkpoints", exist_ok=True)
best_model_path = "../checkpoints/BST_best.pth"

print(f"\n🚀 Démarrage (BST-DCN)...")

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
        
        # Gradient Clipping (Important pour Transformer pour éviter l'explosion)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        total_loss += loss.item()
        steps += 1
        
        if steps % 200 == 0:
            print(f"Epoch {epoch+1} | Step {steps} | Loss: {loss.item():.4f}")

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
        
        # Step Scheduler
        scheduler.step(auc)

        if auc > best_auc:
            best_auc = auc
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state_dict, best_model_path)
            print(f"🏆 Nouveau record BST! Sauvegardé dans {best_model_path}")

print(f"Fin. Meilleur AUC : {best_auc:.4f}")