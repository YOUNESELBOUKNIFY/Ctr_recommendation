import optuna
import torch
import yaml
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloader import MMCTRDataLoader
# IMPORT DU NOUVEAU MODÈLE
from model_new import build_model
from utils import set_seed, compute_auc

# ========================
# 1. Config
# ========================
config_path = "../config/tabtransformer_config.yaml"
if not os.path.exists(config_path):
    config_path = "config/tabtransformer_config.yaml"

with open(config_path, "r") as f:
    raw_cfg = yaml.safe_load(f)

dataset_id = raw_cfg["dataset_id"]
dataset_cfg = raw_cfg["dataset_config"][dataset_id]
base_exp_id = raw_cfg.get("base_expid", "TabTransformer_default")
base_model_cfg = raw_cfg[base_exp_id]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Tuning MMDIN on device: {device}")

if torch.cuda.device_count() > 1:
    print(f"🚀 Multi-GPU activé pour le tuning.")

# ========================
# 2. Objectif Optuna
# ========================
def objective(trial):
    
    # --- A. Espace de recherche spécifique MMDIN ---
    lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4096, 8192])
    
    # Params Architecture DIN
    emb_dim = trial.suggest_categorical("embedding_dim", [32, 64, 128])
    dropout = trial.suggest_float("net_dropout", 0.0, 0.5)
    
    # Longueur de l'historique utilisateur (Impact fort sur DIN)
    max_len = trial.suggest_categorical("max_len", [10, 20, 50])
    
    # Mise à jour config
    current_cfg = base_model_cfg.copy()
    current_cfg["learning_rate"] = lr
    current_cfg["batch_size"] = batch_size
    current_cfg["embedding_dim"] = emb_dim
    current_cfg["net_dropout"] = dropout
    current_cfg["max_len"] = max_len
    
    # --- B. Data & Model ---
    set_seed(2025)
    
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
    
    model = build_model(None, current_cfg)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()

    # --- C. Training Loop (Short) ---
    epochs = 30
    best_valid_auc = 0.0

    for epoch in range(epochs):
        model.train()
        for batch_dict, labels in train_loader:
            for k, v in batch_dict.items():
                batch_dict[k] = v.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            y_pred = model(batch_dict)
            loss = loss_fn(y_pred, labels)
            loss.backward()
            optimizer.step()

        # --- D. Validation ---
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
        else:
            auc = 0.0
            
        print(f"Trial {trial.number} | Epoch {epoch+1} | Valid AUC: {auc:.4f}")
        
        # --- E. Pruning ---
        trial.report(auc, epoch)
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch+1}.")
            raise optuna.exceptions.TrialPruned()
            
        best_valid_auc = max(best_valid_auc, auc)

    del model, optimizer, train_loader, valid_loader
    torch.cuda.empty_cache()

    return best_valid_auc

# ========================
# 3. Main
# ========================
if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize", 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    )
    
    print("Début du Tuning MMDIN...")
    study.optimize(objective, n_trials=20)

    print("\n==================================")
    print("Meilleurs hyperparamètres MMDIN :")
    print(study.best_params)
    print(f"Meilleur AUC : {study.best_value}")
    
    with open("best_hyperparams_mmdin.yaml", "w") as f:
        yaml.dump(study.best_params, f)