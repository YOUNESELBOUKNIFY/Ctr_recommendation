import torch
import yaml
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tab_transformer_pytorch import TabTransformer
from sklearn.metrics import roc_auc_score, log_loss

# ========================
# Charger config YAML
# ========================
with open("./config/tabtransformer_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

seed = cfg["base_config"]["seed"]
torch.manual_seed(seed)
np.random.seed(seed)

dataset_id = cfg["dataset_id"]
dataset_cfg = cfg["dataset_config"][dataset_id]
model_cfg = cfg["TabTransformer_default"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========================
# Dataset / DataLoader
# ========================
class ParquetDataset(Dataset):
    def __init__(self, data_path):
        self.df = pd.read_parquet(data_path)
        # Séparer features catégorielles et continues
        self.cat_cols = ["likes_level", "views_level", "item_id"]
        self.cont_cols = ["item_emb_d128"]
        self.label_col = "label"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x_cat = torch.tensor(row[self.cat_cols].values, dtype=torch.long)
        x_cont = torch.tensor(row[self.cont_cols].values, dtype=torch.float)
        y = torch.tensor(row[self.label_col], dtype=torch.float)
        return x_cat, x_cont, y

train_dataset = ParquetDataset(dataset_cfg["train_data"])
valid_dataset = ParquetDataset(dataset_cfg["valid_data"])

train_loader = DataLoader(train_dataset, batch_size=model_cfg["batch_size"], shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=model_cfg["batch_size"], shuffle=False, num_workers=4)

# ========================
# Modèle
# ========================
categories = [11, 11, 91718]  # likes_level, views_level, item_id
num_cont = 128  # dimension de item_emb_d128

model = TabTransformer(
    categories=categories,
    num_continuous=num_cont,
    dim=model_cfg["embedding_dim"],
    depth=4,
    heads=4,
    attn_dropout=model_cfg.get("attention_dropout", 0.1),
    ff_dropout=model_cfg.get("net_dropout", 0.0),
    mlp_hidden_mults=(4, 2),
    mlp_act=torch.nn.ReLU(),
    dim_out=1
)

# Multi-GPU
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg["learning_rate"])
loss_fn = torch.nn.BCELoss()

# ========================
# Entraînement
# ========================
for epoch in range(model_cfg["epochs"]):
    model.train()
    train_loss = 0
    for x_cat, x_cont, y_true in train_loader:
        x_cat = x_cat.to(device)
        x_cont = x_cont.to(device)
        y_true = y_true.to(device)

        optimizer.zero_grad()
        y_pred = model(x_cat, x_cont).squeeze(-1)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * y_true.size(0)

    train_loss /= len(train_dataset)

    # Validation
    model.eval()
    y_trues, y_preds = [], []
    with torch.no_grad():
        for x_cat, x_cont, y_true in valid_loader:
            x_cat = x_cat.to(device)
            x_cont = x_cont.to(device)
            y_true = y_true.to(device)
            y_pred = model(x_cat, x_cont).squeeze(-1)
            y_trues.append(y_true.cpu().numpy())
            y_preds.append(y_pred.cpu().numpy())

    y_trues = np.concatenate(y_trues)
    y_preds = np.concatenate(y_preds)
    auc = roc_auc_score(y_trues, y_preds)
    logloss = log_loss(y_trues, y_preds)

    print(f"Epoch {epoch+1}/{model_cfg['epochs']} - Train Loss: {train_loss:.4f} - Valid AUC: {auc:.4f}, LogLoss: {logloss:.4f}")

# ========================
# Sauvegarde modèle
# ========================
torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
           "./checkpoints/TabTransformer_best.pth")
print("Model saved in ./checkpoints/TabTransformer_best.pth")
