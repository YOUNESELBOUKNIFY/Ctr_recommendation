import torch
from src.dataloader import MMCTRDataLoader
from src.model import build_model
from src.utils import set_seed, compute_auc, compute_logloss
import yaml

# ========================
# Charger config YAML
# ========================
with open("./config/tabtransformer_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

set_seed(cfg["base_config"]["seed"])

dataset_id = cfg["dataset_id"]
dataset_cfg = cfg["dataset_config"][dataset_id]
model_cfg = cfg["TabTransformer_default"]

# ========================
# Feature map (FuxiCTR)
# ========================
from fuxictr.pytorch.datasets import Dataset
dataset = Dataset(dataset_id=dataset_id, config_file="./config/tabtransformer_config.yaml")
feature_map = dataset.feature_map

# ========================
# DataLoader
# ========================
train_loader = MMCTRDataLoader(feature_map,
                               dataset_cfg["train_data"],
                               dataset_cfg["item_info"],
                               batch_size=model_cfg["batch_size"],
                               shuffle=True,
                               max_len=model_cfg["max_len"])

valid_loader = MMCTRDataLoader(feature_map,
                               dataset_cfg["valid_data"],
                               dataset_cfg["item_info"],
                               batch_size=model_cfg["batch_size"],
                               shuffle=False,
                               max_len=model_cfg["max_len"])

# ========================
# Modèle
# ========================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model(feature_map, model_cfg)
model.to(device)

# Optimizer + loss
optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg["learning_rate"])
loss_fn = torch.nn.BCELoss()  # binary classification

# ========================
# Entraînement simple
# ========================
for epoch in range(model_cfg["epochs"]):
    model.train()
    for batch_dict, item_dict, mask in train_loader:
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
        item_dict = {k: v.to(device) for k, v in item_dict.items()}
        mask = mask.to(device)

        y_true = batch_dict["label"].float()
        optimizer.zero_grad()
        y_pred = model((batch_dict, item_dict, mask))["y_pred"].squeeze(-1)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()

    # ========================
    # Evaluation sur valid
    # ========================
    model.eval()
    y_trues, y_preds = [], []
    with torch.no_grad():
        for batch_dict, item_dict, mask in valid_loader:
            batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
            item_dict = {k: v.to(device) for k, v in item_dict.items()}
            mask = mask.to(device)

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
torch.save(model.state_dict(), "./checkpoints/TabTransformer_best.pth")
print("Model saved in ./checkpoints/TabTransformer_best.pth")
