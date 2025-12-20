import argparse
import os
import sys
import yaml
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloader import MMCTRDataLoader
from model_fibinet import build_model
from utils import set_seed, compute_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="chemin yaml")
    ap.add_argument("--ckpt_dir", type=str, default="../checkpoints", help="dossier checkpoints")
    args = ap.parse_args()

    # ========================
    # 1) Load config
    # ========================
    print(f"⚙️  Chargement config : {args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_id = cfg["dataset_id"]
    dataset_cfg = cfg["dataset_config"][dataset_id]

    expid = cfg["base_expid"]
    model_cfg = cfg[expid]

    # seed: priorité base_config.seed puis model_cfg.seed
    seed = int(cfg.get("base_config", {}).get("seed", model_cfg.get("seed", 2025)))
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 device={device} | expid={expid} | seed={seed}")

    # ========================
    # 2) Hyperparams
    # ========================
    batch_size = int(model_cfg.get("batch_size", 4096))
    max_len = int(model_cfg.get("max_len", 20))
    epochs = int(model_cfg.get("epochs", 40))

    lr = float(model_cfg.get("learning_rate", 1e-3))
    weight_decay = float(model_cfg.get("weight_decay", 1e-5))
    opt_name = str(model_cfg.get("optimizer", "adamw")).lower()

    # Early stopping params
    use_es = bool(model_cfg.get("early_stopping", True))
    patience = int(model_cfg.get("patience", 4))
    min_delta = float(model_cfg.get("min_delta", 0.0))

    # Monitor (AUC by default)
    monitor = str(model_cfg.get("monitor", "AUC")).upper()
    monitor_mode = str(model_cfg.get("monitor_mode", "max")).lower()  # max for AUC

    if monitor != "AUC":
        print(f"⚠️  monitor={monitor} (dans ce code on valide surtout AUC).")

    print(f"ES={use_es} | patience={patience} | min_delta={min_delta} | monitor={monitor} | mode={monitor_mode}")

    # ========================
    # 3) Loaders
    # ========================
    num_workers = int(cfg.get("base_config", {}).get("num_workers", 4))

    train_loader = MMCTRDataLoader(
        data_path=dataset_cfg["train_data"],
        item_info_path=dataset_cfg["item_info"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        max_len=max_len
    )

    valid_loader = MMCTRDataLoader(
        data_path=dataset_cfg["valid_data"],
        item_info_path=dataset_cfg["item_info"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        max_len=max_len
    )

    # ========================
    # 4) Model
    # ========================
    print("🏗️  Construction du modèle MM-FiBiNET...")
    model = build_model(None, model_cfg)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)

    # Optimizer
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = torch.nn.BCELoss()

    # Scheduler OneCycleLR
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr * 10,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0
    )

    # ========================
    # 5) Checkpoint path
    # ========================
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_path = os.path.join(args.ckpt_dir, f"{expid}_best.pth")

    # ========================
    # 6) Train loop + Early stopping
    # ========================
    best_score = -1e18 if monitor_mode == "max" else 1e18
    bad_epochs = 0

    print("\n🚀 Démarrage entraînement...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        steps = 0

        for batch_dict, labels in train_loader:
            for k, v in batch_dict.items():
                batch_dict[k] = v.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            y_pred = model(batch_dict)
            loss = loss_fn(y_pred, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            optimizer.step()
            scheduler.step()

            total_loss += float(loss.item())
            steps += 1

        avg_loss = total_loss / max(1, steps)

        # ===== VALID =====
        model.eval()
        y_trues, y_preds = [], []
        with torch.no_grad():
            for batch_dict, labels in valid_loader:
                for k, v in batch_dict.items():
                    batch_dict[k] = v.to(device)
                y_pred = model(batch_dict)
                y_trues.append(labels.cpu().numpy())
                y_preds.append(y_pred.cpu().numpy())

        auc = compute_auc(np.concatenate(y_trues), np.concatenate(y_preds))

        # monitor score
        score = auc  # si tu veux logloss, il faut ajouter compute_logloss
        print(f"✅ epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | valid_auc={auc:.5f}")

        # ===== improvement? =====
        if monitor_mode == "max":
            improved = (score > best_score + min_delta)
        else:
            improved = (score < best_score - min_delta)

        if improved:
            best_score = score
            bad_epochs = 0

            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state_dict, best_path)
            print(f"🏆 save best: {best_path} | best_auc={best_score:.5f}")
        else:
            bad_epochs += 1
            print(f"⏳ no improv. bad_epochs={bad_epochs}/{patience}")

            if use_es and bad_epochs >= patience:
                print(f"🛑 Early stopping à epoch {epoch+1}. best_auc={best_score:.5f}")
                break

    print(f"\nFIN expid={expid} | best_auc={best_score:.5f}")
    print(f"Meilleur checkpoint: {best_path}")


if __name__ == "__main__":
    main()
