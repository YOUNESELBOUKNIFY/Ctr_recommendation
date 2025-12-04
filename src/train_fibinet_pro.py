import torch
import yaml
import numpy as np
import os
import sys
import time

# Ajout du chemin src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloader import MMCTRDataLoader
# IMPORTANT : On charge le modèle PRO
from model_fibinet_pro import build_model  
from utils import set_seed, compute_auc, compute_logloss

# ========================
# 1. Config
# ========================
config_path = "../config/fibinet_pro_config.yaml"
if not os.path.exists(config_path):
    config_path = "config/fibinet_pro_config.yaml"

print(f"⚙️  Chargement config FiBiNET Pro : {config_path}")
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

dataset_id = cfg["dataset_id"]
dataset_cfg = cfg["dataset_config"][dataset_id]
# On récupère la config spécifique MM_FiBiNET_Pro_Run
model_cfg = cfg[cfg["base_expid"]]

# ========================
# 2. Setup
# ========================
set_seed(model_cfg.get("seed", 2025))
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Training FiBiNET Pro sur : {device}")

# ========================
# 3. DataLoaders
# ========================
batch_size = int(model_cfg.get("batch_size", 4096))
max_len = int(model_cfg.get("max_len", 20))

print(f"📦 DataLoaders (Batch: {batch_size}, Hist Len: {max_len})")

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
print("🏗️  Construction du modèle MM-FiBiNET Pro...")
model = build_model(None, model_cfg)

if torch.cuda.device_count() > 1:
    print(f"🚀 Multi-GPU activé ({torch.cuda.device_count()} GPUs)")
    model = torch.nn.DataParallel(model)

model.to(device)

# Hyperparamètres
lr = float(model_cfg.get("learning_rate", 3e-4))
weight_decay = float(model_cfg.get("weight_decay", 1e-4))
epochs = int(model_cfg.get("epochs", 40))

# Optimiseur AdamW (Recommandé pour les gros modèles avec Weight Decay)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_fn = torch.nn.BCELoss()

# --- SCHEDULER OneCycleLR ---
# Stratégie de convergence SOTA : Chauffe (Warmup) puis Refroidissement (Annealing)
steps_per_epoch = len(train_loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr * 10,       # Pic du LR (x10 par rapport au LR de base)
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    pct_start=0.25,       # 25% du temps pour monter
    div_factor=25.0,      # LR initial = max_lr / 25
    final_div_factor=1000.0
)

# ========================
# 5. Entraînement
# ========================
best_auc = 0.0
checkpoint_dir = "../checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
best_model_path = os.path.join(checkpoint_dir, "FiBiNET_Pro_best.pth")

print(f"\n🚀 Démarrage (FiBiNET Pro) pour {epochs} époques...")
start_time = time.time()

for epoch in range(epochs):
    epoch_start = time.time()
    model.train()
    total_loss = 0
    steps = 0
    
    for batch_dict, labels in train_loader:
        # Transfert GPU
        for k, v in batch_dict.items():
            batch_dict[k] = v.to(device)
        labels = labels.to(device)

        # Forward
        optimizer.zero_grad()
        y_pred = model(batch_dict)
        loss = loss_fn(y_pred, labels)
        
        # Backward
        loss.backward()
        
        # Gradient Clipping (Vital pour la stabilité avec 128 dim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        scheduler.step() # Update LR
        
        total_loss += loss.item()
        steps += 1
        
        if steps % 200 == 0:
            current_lr = scheduler.get_last_lr()[0]
            sys.stdout.write(f"\rEpoch {epoch+1} | Step {steps} | Loss: {loss.item():.4f} | LR: {current_lr:.6f}")
            sys.stdout.flush()

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

    epoch_duration = time.time() - epoch_start
    
    if len(y_trues) > 0:
        auc = compute_auc(np.concatenate(y_trues), np.concatenate(y_preds))
        logloss = compute_logloss(np.concatenate(y_trues), np.concatenate(y_preds))
        
        print(f"\n✅ Epoch {epoch+1}/{epochs} [{epoch_duration:.0f}s] | "
              f"Train Loss: {avg_loss:.4f} | Valid AUC: {auc:.4f} | LogLoss: {logloss:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state_dict, best_model_path)
            print(f"🏆 Nouveau Record ! Sauvegardé dans {best_model_path}")
    else:
        print("\n⚠️ Validation set empty.")

print(f"\n🏁 Fin. Meilleur AUC : {best_auc:.4f}")