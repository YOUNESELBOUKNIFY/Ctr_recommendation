import torch
import yaml
import numpy as np
import os
from dataloader import MMCTRDataLoader
from model import build_model
from utils import set_seed, compute_auc, compute_logloss

# ========================
# Charger config YAML
# ========================
config_path = "../config/tabtransformer_config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

# Seed
seed = cfg.get("base_config", {}).get("seed", 20242025)
set_seed(seed)

dataset_id = cfg["dataset_id"]
dataset_cfg = cfg["dataset_config"][dataset_id]
exp_id = cfg.get("base_expid", "TabTransformer_default")
model_cfg = cfg[exp_id]

# ========================
# Device et multi-GPU
# ========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
multi_gpu = torch.cuda.device_count() > 1
if multi_gpu:
    print(f"Using {torch.cuda.device_count()} GPUs!")

# ========================
# DataLoader
# ========================
train_loader = MMCTRDataLoader(
    feature_map=model_cfg.get("feature_map"),
    data_path=dataset_cfg["train_data"],
    item_info_path=dataset_cfg["item_info"],
    batch_size=int(model_cfg.get("batch_size", 32)),
    shuffle=True,
    num_workers=4,
    max_len=int(model_cfg.get("max_len", 5))
)

valid_loader = MMCTRDataLoader(
    feature_map=model_cfg.get("feature_map"),
    data_path=dataset_cfg["valid_data"],
    item_info_path=dataset_cfg["item_info"],
    batch_size=int(model_cfg.get("batch_size", 32)),
    shuffle=False,
    num_workers=4,
    max_len=int(model_cfg.get("max_len", 5))
)

# ========================
# Modèle
# ========================
feature_map = train_loader.dataset  # placeholder si build_model en a besoin
model = build_model(feature_map, model_cfg)

if multi_gpu:
    model = torch.nn.DataParallel(model)

model.to(device)

# Optimizer et loss
learning_rate = float(model_cfg.get("learning_rate", 1e-3))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.BCELoss()

# ========================
# Répertoire checkpoints
# ========================
os.makedirs("../checkpoints", exist_ok=True)

# ========================
# Entraînement
# ========================
epochs = int(model_cfg.get("epochs", 10))
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_dict, item_dict, mask in train_loader:
        # Envoyer tensors sur GPU et convertir les object
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

        for k, v in item_dict.items():
            if v.dtype in [torch.float32, torch.float64, torch.int64]:
                item_dict[k] = v.to(device)
            else:
                # v est object -> convertir
                arr = np.stack([np.array(x, dtype=np.float32) for x in v.cpu().numpy()])
                item_dict[k] = torch.tensor(arr, dtype=torch.float).to(device)

        mask = mask.to(device)

        if "label" not in batch_dict:
            continue  # ignorer batch si label absent

        y_true = batch_dict["label"].float()
        optimizer.zero_grad()
        y_pred = model((batch_dict, item_dict, mask))["y_pred"].squeeze(-1)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * y_true.size(0)

    train_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")

    # ========================
    # Validation
    # ========================
    model.eval()
    y_trues, y_preds = [], []
    with torch.no_grad():
        for batch_dict, item_dict, mask in valid_loader:
            batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

            for k, v in item_dict.items():
                if v.dtype in [torch.float32, torch.float64, torch.int64]:
                    item_dict[k] = v.to(device)
                else:
                    arr = np.stack([np.array(x, dtype=np.float32) for x in v.cpu().numpy()])
                    item_dict[k] = torch.tensor(arr, dtype=torch.float).to(device)

            mask = mask.to(device)

            if "label" not in batch_dict:
                continue

            y_true = batch_dict["label"].float()
            y_pred = model((batch_dict, item_dict, mask))["y_pred"].squeeze(-1)
            y_trues.append(y_true.cpu().numpy())
            y_preds.append(y_pred.cpu().numpy())

    y_trues = np.concatenate(y_trues)
    y_preds = np.concatenate(y_preds)
    auc = compute_auc(y_trues, y_preds)
    logloss = compute_logloss(y_trues, y_preds)
    print(f"Epoch {epoch+1}: Valid AUC={auc:.4f}, LogLoss={logloss:.4f}")

# ========================
# Sauvegarde modèle
# ========================
checkpoint_path = "../checkpoints/TabTransformer_best.pth"
torch.save(
    model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
    checkpoint_path
)
print(f"Model saved in {checkpoint_path}")
