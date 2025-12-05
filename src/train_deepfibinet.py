import torch
import yaml
import numpy as np
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloader import MMCTRDataLoader
# Import du modèle hybride
from model_deepfibinet import build_model
from utils import set_seed, compute_auc

# Config
config_path = "../config/deepfibinet_config.yaml"
if not os.path.exists(config_path): config_path = "config/deepfibinet_config.yaml"

with open(config_path, "r") as f: cfg = yaml.safe_load(f)
dataset_id = cfg["dataset_id"]
dataset_cfg = cfg["dataset_config"][dataset_id]
model_cfg = cfg[cfg["base_expid"]]

# Setup
set_seed(model_cfg.get("seed", 2025))
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Training DeepFiBiNET (Hybrid SOTA) sur : {device}")

# Data
batch_size = int(model_cfg.get("batch_size", 4096))
max_len = int(model_cfg.get("max_len", 20))

train_loader = MMCTRDataLoader(
    feature_map=None, data_path=dataset_cfg["train_data"],
    item_info_path=dataset_cfg["item_info"], batch_size=batch_size,
    shuffle=True, num_workers=4, max_len=max_len
)

valid_loader = MMCTRDataLoader(
    feature_map=None, data_path=dataset_cfg["valid_data"],
    item_info_path=dataset_cfg["item_info"], batch_size=batch_size,
    shuffle=False, num_workers=4, max_len=max_len
)

# Modèle
model = build_model(None, model_cfg)
if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)
model.to(device)

# Optimiseur
lr = float(model_cfg.get("learning_rate", 5e-4))
epochs = int(model_cfg.get("epochs", 35))
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
loss_fn = torch.nn.BCELoss()

# Scheduler : Pic conservateur (0.005) car le modèle a beaucoup de paramètres
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.005, epochs=epochs, steps_per_epoch=len(train_loader),
    pct_start=0.3, div_factor=25.0, final_div_factor=1000.0
)

best_auc = 0.0
os.makedirs("../checkpoints", exist_ok=True)
best_model_path = "../checkpoints/DeepFiBiNET_best.pth"

print(f"\n🚀 Démarrage...")

for epoch in range(epochs):
    model.train()
    total_loss, steps = 0, 0
    for batch_dict, labels in train_loader:
        for k, v in batch_dict.items(): batch_dict[k] = v.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        y_pred = model(batch_dict)
        loss = loss_fn(y_pred, labels)
        loss.backward()
        
        # Gradient Clipping un peu plus fort pour stabiliser DCN
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        steps += 1
        
        if steps % 200 == 0:
            sys.stdout.write(f"\rEpoch {epoch+1} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            sys.stdout.flush()

    # Validation
    model.eval()
    y_trues, y_preds = [], []
    with torch.no_grad():
        for batch_dict, labels in valid_loader:
            for k, v in batch_dict.items(): batch_dict[k] = v.to(device)
            y_pred = model(batch_dict)
            y_trues.append(labels.cpu().numpy())
            y_preds.append(y_pred.cpu().numpy())

    if len(y_trues) > 0:
        auc = compute_auc(np.concatenate(y_trues), np.concatenate(y_preds))
        print(f"\n✅ Epoch {epoch+1} | Valid AUC: {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state_dict, best_model_path)
            print("🏆 Nouveau Record !")

print(f"Fin. Meilleur AUC : {best_auc:.4f}")