import optuna
import torch
import yaml
import os
import sys
import numpy as np

# Ajout du chemin src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloader import MMCTRDataLoader
# On importe le modèle FiBiNET
from model_fibinet import build_model
from utils import set_seed, compute_auc

# ========================
# 1. Config de base
# ========================
config_path = "../config/fibinet_config.yaml"
if not os.path.exists(config_path):
    config_path = "config/fibinet_config.yaml"

with open(config_path, "r") as f:
    raw_cfg = yaml.safe_load(f)

dataset_id = raw_cfg["dataset_id"]
dataset_cfg = raw_cfg["dataset_config"][dataset_id]
base_exp_id = raw_cfg.get("base_expid", "MM_FiBiNET_Run")
base_model_cfg = raw_cfg[base_exp_id]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Tuning FiBiNET sur : {device}")

if torch.cuda.device_count() > 1:
    print(f"✅ Multi-GPU activé ({torch.cuda.device_count()} GPUs)")

# ========================
# 2. Objectif Optuna
# ========================
def objective(trial):
    """
    Fonction objectif pour optimiser l'AUC de FiBiNET.
    """
    
    # --- A. Espace de Recherche (Hyperparamètres) ---
    
    # 1. Paramètres d'entraînement
    lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4096, 8192])
    
    # 2. Paramètres FiBiNET Critiques
    # Important: FiBiNET a besoin d'embeddings uniformes
    emb_dim = trial.suggest_categorical("embedding_dim", [32, 64, 128])
    
    # SENet Reduction Ratio (r) : Plus il est petit, plus le SENet est complexe
    reduction_ratio = trial.suggest_categorical("senet_reduction_ratio", [2, 3, 4])
    
    # 3. Paramètres Généraux
    dropout = trial.suggest_float("net_dropout", 0.1, 0.5)
    max_len = trial.suggest_categorical("max_len", [10, 20, 50]) # Historique
    
    # --- B. Mise à jour de la config ---
    current_cfg = base_model_cfg.copy()
    current_cfg["learning_rate"] = lr
    current_cfg["batch_size"] = batch_size
    current_cfg["embedding_dim"] = emb_dim
    current_cfg["senet_reduction_ratio"] = reduction_ratio # À utiliser dans model_fibinet si paramétrable
    current_cfg["net_dropout"] = dropout
    current_cfg["max_len"] = max_len
    
    # --- C. Setup Data & Model ---
    set_seed(2025)
    
    # DataLoaders
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
    
    # Construction du modèle
    # Note: Assurez-vous que model_fibinet.py utilise current_cfg pour reduction_ratio si implémenté
    model = build_model(None, current_cfg)
    
    # Gestion Multi-GPU
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()

    # --- D. Training Loop (Short w/ Pruning) ---
    epochs = 8 # Suffisant pour voir si ça converge bien avec FiBiNET
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

        # --- E. Validation ---
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
            
        # Report pour Optuna
        trial.report(auc, epoch)
        print(f"[Trial {trial.number}] Epoch {epoch+1}: AUC = {auc:.4f}")

        # --- F. Pruning (Hyperband) ---
        if trial.should_prune():
            print(f"✂️ Trial {trial.number} PRUNED at epoch {epoch+1}.")
            raise optuna.exceptions.TrialPruned()
            
        best_valid_auc = max(best_valid_auc, auc)

    # Nettoyage mémoire GPU
    del model, optimizer, train_loader, valid_loader
    torch.cuda.empty_cache()

    return best_valid_auc

# ========================
# 3. Lancement Main
# ========================
if __name__ == "__main__":
    
    # Sampler TPE Multivarié (Apprend les relations entre params)
    sampler = optuna.samplers.TPESampler(multivariate=True, seed=2025)
    
    # Pruner Hyperband (Coupe agressivement les mauvais essais)
    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=8, reduction_factor=3)

    study = optuna.create_study(
        direction="maximize", 
        sampler=sampler,
        pruner=pruner,
        study_name="FiBiNET_Optimization"
    )
    
    print("\n🎯 Démarrage de l'optimisation FiBiNET...")
    # 30 essais pour bien explorer
    study.optimize(objective, n_trials=30)

    print("\n==================================")
    print("🏆 OPTIMISATION TERMINÉE")
    print(f"Meilleur AUC : {study.best_value:.5f}")
    print("Meilleurs Hyperparamètres :")
    print(study.best_params)
    
    # Sauvegarde
    with open("best_hyperparams_fibinet.yaml", "w") as f:
        yaml.dump(study.best_params, f)
    print("\n✅ Paramètres sauvegardés dans 'best_hyperparams_fibinet.yaml'.")