import optuna
import torch
import yaml
import os
import sys
import numpy as np

# Ajout du chemin src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloader import MMCTRDataLoader
from model import build_model
from utils import set_seed, compute_auc

# ========================
# 1. Configuration Globale
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
print(f"Tuning on device: {device}")

# ========================
# 2. Fonction Objective (Le cœur d'Optuna)
# ========================
def objective(trial):
    """
    Cette fonction est exécutée à chaque essai (trial).
    Elle doit retourner la métrique à maximiser (ici l'AUC).
    """
    
    # --- A. Définition de l'espace de recherche (Hyperparamètres) ---
    # Optuna va choisir des valeurs ici
    
    # 1. Paramètres d'apprentissage
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4096, 8192, 16384])
    
    # 2. Paramètres du modèle (Architecture)
    emb_dim = trial.suggest_categorical("embedding_dim", [32, 64, 128])
    n_layers = trial.suggest_int("transformer_n_layers", 1, 4)
    n_heads = trial.suggest_categorical("transformer_n_heads", [2, 4, 8])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    
    # Mise à jour de la config pour cet essai
    current_cfg = base_model_cfg.copy()
    current_cfg["learning_rate"] = lr
    current_cfg["batch_size"] = batch_size
    current_cfg["embedding_dim"] = emb_dim
    current_cfg["transformer_n_layers"] = n_layers
    current_cfg["transformer_n_heads"] = n_heads
    current_cfg["transformer_dropout"] = dropout
    current_cfg["net_dropout"] = dropout
    
    # --- B. Préparation des Données et Modèle ---
    # On fixe une seed différente par trial ou la même pour comparer purement les hyperparams
    set_seed(2025) 
    
    train_loader = MMCTRDataLoader(
        feature_map=None,
        data_path=dataset_cfg["train_data"],
        item_info_path=dataset_cfg["item_info"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        max_len=int(current_cfg.get("max_len", 5))
    )

    valid_loader = MMCTRDataLoader(
        feature_map=None,
        data_path=dataset_cfg["valid_data"],
        item_info_path=dataset_cfg["item_info"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        max_len=int(current_cfg.get("max_len", 5))
    )
    
    model = build_model(None, current_cfg)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()

    # --- C. Boucle d'entraînement Allégée (Max 5-10 époques pour tester vite) ---
    epochs = 5  # Pas besoin de 100 epochs pour savoir si les hyperparams sont bons
    
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
            
        # === AFFICHAGE DE L'AUC EN TEMPS RÉEL ===
        print(f"Trial {trial.number} | Epoch {epoch+1} | Valid AUC: {auc:.4f}")
        
        # --- E. Pruning (Arrêt automatique si mauvais résultats) ---
        # Optuna compare l'AUC actuel aux meilleurs essais précédents.
        # S'il est trop bas, on arrête tout de suite (TrialPruned).
        trial.report(auc, epoch)
        if trial.should_prune():
            # Petit message pour dire qu'on arrête
            print(f"Trial {trial.number} pruned at epoch {epoch+1}.")
            raise optuna.exceptions.TrialPruned()
            
        best_valid_auc = max(best_valid_auc, auc)

    # Nettoyage GPU après chaque essai pour éviter "Out of Memory"
    del model
    del optimizer
    del train_loader
    del valid_loader
    torch.cuda.empty_cache()

    return best_valid_auc

# ========================
# 3. Lancement de l'étude
# ========================
if __name__ == "__main__":
    # On crée une étude qui cherche à MAXIMISER l'AUC
    study = optuna.create_study(
        direction="maximize", 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    )
    
    print("Début de l'optimisation des hyperparamètres...")
    # n_trials = nombre d'essais différents (ex: 20 configurations différentes)
    study.optimize(objective, n_trials=20)

    print("\n==================================")
    print("Optimisation terminée !")
    print("Meilleurs hyperparamètres trouvés :")
    print(study.best_params)
    print(f"Meilleur AUC : {study.best_value}")
    print("==================================")
    
    # Sauvegarder les meilleurs params dans un fichier YAML pour les réutiliser
    with open("best_hyperparams.yaml", "w") as f:
        yaml.dump(study.best_params, f)