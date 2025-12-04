import optuna
import torch
import yaml
import os
import sys
import numpy as np

# Ajout du chemin src pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloader import MMCTRDataLoader
# Assurez-vous d'avoir le fichier model_new.py (ou model_din.py) avec la classe MMDIN
from model_new import build_model
from utils import set_seed, compute_auc

# ========================
# 1. Chargement de la Config de Base
# ========================
config_path = "../config/din_conf.yaml"
if not os.path.exists(config_path):
    # Fallback si lancé depuis la racine
    config_path = "config/din_conf.yaml"

with open(config_path, "r") as f:
    raw_cfg = yaml.safe_load(f)

# On récupère les infos du dataset
dataset_id = raw_cfg["dataset_id"]
dataset_cfg = raw_cfg["dataset_config"][dataset_id]

# On récupère la config du modèle de base (MMDIN_Optimized_Run)
base_exp_id = raw_cfg.get("base_expid", "MMDIN_Optimized_Run")
base_model_cfg = raw_cfg[base_exp_id]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Lancement du Tuning Optuna sur : {device}")

if torch.cuda.device_count() > 1:
    print(f"✅ Multi-GPU détecté : {torch.cuda.device_count()} GPUs seront utilisés.")

# ========================
# 2. Fonction Objective (Cœur du Tuning)
# ========================
def objective(trial):
    """
    Cette fonction est exécutée pour chaque 'Essai'.
    Elle entraîne un modèle avec des paramètres suggérés par Optuna
    et retourne l'AUC sur le set de validation.
    """
    
    # --- A. Définition de l'Espace de Recherche (Search Space) ---
    # Optuna va explorer ces plages de valeurs pour maximiser l'AUC
    
    # 1. Paramètres d'entraînement
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4096, 8192])
    
    # 2. Paramètres d'Architecture MMDIN
    emb_dim = trial.suggest_categorical("embedding_dim", [64, 128])
    dropout = trial.suggest_float("net_dropout", 0.1, 0.5)
    
    # 3. Paramètres spécifiques DIN (Attention)
    att_units = trial.suggest_categorical("attention_hidden_units", ["64,32", "128,64"])
    att_dropout = trial.suggest_float("attention_dropout", 0.0, 0.4)
    max_len = trial.suggest_categorical("max_len", [10, 20, 50]) # Longueur historique
    
    # Conversion string -> list pour att_units
    att_units_list = [int(x) for x in att_units.split(',')]

    # --- B. Mise à jour de la configuration ---
    current_cfg = base_model_cfg.copy()
    current_cfg["learning_rate"] = lr
    current_cfg["batch_size"] = batch_size
    current_cfg["embedding_dim"] = emb_dim
    current_cfg["net_dropout"] = dropout
    current_cfg["attention_hidden_units"] = att_units_list
    current_cfg["attention_dropout"] = att_dropout
    current_cfg["max_len"] = max_len
    
    # --- C. Préparation Data & Modèle ---
    set_seed(2025) # Seed fixe pour comparabilité
    
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
    
    # Gestion GPU
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()

    # --- D. Entraînement Rapide (avec Pruning) ---
    # On n'entraîne que sur 5 à 10 époques pour le tuning
    epochs = 10 
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

        # Calcul AUC
        if len(y_trues) > 0:
            auc = compute_auc(np.concatenate(y_trues), np.concatenate(y_preds))
        else:
            auc = 0.0
            
        # Reporting à Optuna
        trial.report(auc, epoch)
        print(f"[Trial {trial.number}] Epoch {epoch+1}: AUC = {auc:.4f}")

        # --- F. Pruning (Arrêt automatique si mauvais résultat) ---
        if trial.should_prune():
            print(f"✂️ Trial {trial.number} PRUNED at epoch {epoch+1} (AUC too low).")
            raise optuna.exceptions.TrialPruned()
            
        best_valid_auc = max(best_valid_auc, auc)

    # Nettoyage mémoire
    del model, optimizer, train_loader, valid_loader
    torch.cuda.empty_cache()

    return best_valid_auc

# ========================
# 3. Lancement de l'Étude
# ========================
if __name__ == "__main__":
    
    # Stratégie d'échantillonnage TPE Multivarié (State of the Art)
    sampler = optuna.samplers.TPESampler(multivariate=True, seed=2025)
    
    # Stratégie de Pruning Hyperband (Très efficace)
    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=10, reduction_factor=3)

    # Création de l'étude : ON MAXIMISE L'AUC
    study = optuna.create_study(
        direction="maximize", 
        sampler=sampler,
        pruner=pruner,
        study_name="MMDIN_AUC_Maximization"
    )
    
    print("\n🎯 Démarrage de l'optimisation pour maximiser l'AUC...")
    # Nombre d'essais (Trials)
    study.optimize(objective, n_trials=30)

    print("\n==================================")
    print("🏆 OPTIMISATION TERMINÉE")
    print("==================================")
    print(f"Meilleur AUC obtenu : {study.best_value:.5f}")
    print("Meilleurs Hyperparamètres :")
    for k, v in study.best_params.items():
        print(f"  - {k}: {v}")
    
    # Sauvegarde
    with open("best_hyperparams_mmdin.yaml", "w") as f:
        yaml.dump(study.best_params, f)
    print("\n✅ Les meilleurs paramètres ont été sauvegardés dans 'best_hyperparams_mmdin.yaml'.")