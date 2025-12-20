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

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_id = cfg["dataset_id"]
    dataset_cfg = cfg["dataset_config"][dataset_id]

    expid = cfg["base_expid"]
    model_cfg = cfg[expid]

    seed = int(cfg.get("base_config", {}).get("seed", model_cfg.get("seed", 2025)))
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 device={device} | expid={expid} | seed={seed}")

    batch_size = int(model_cfg.get("batch_size", 4096))
    max_len = int(model_cfg.get("max_len", 20))
    epochs = int(model_cfg.get("epochs", 30))

    train_loader = MMCTRDataLoader(
        data_path=dataset_cfg["train_data"],
        item_info_path=dataset_cfg["item_info"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(cfg.get("base_config", {}).get("num_workers", 4)),
        max_len=max_len
    )

    valid_loader = MMCTRDataLoader(
        data_path=dataset_cfg["valid_data"],
        item_info_path=dataset_cfg["item_info"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg.get("base_config", {}).get("num_workers", 4)),
        max_len=max_len
    )

    model = build_model(None, model_cfg)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    lr = float(model_cfg.get("learning_rate", 1e-3))
    weight_decay = float(model_cfg.get("weight_decay", 1e-5))
    opt_name = str(model_cfg.get("optimizer", "adam")).lower()

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = torch.nn.BCELoss()

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

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_path = os.path.join(args.ckpt_dir, f"{expid}_best.pth")

    best_auc = 0.0
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.item())
            steps += 1

        avg_loss = total_loss / max(1, steps)

        # VALID
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
        print(f"✅ epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | auc={auc:.5f}")

        if auc > best_auc:
            best_auc = auc
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state_dict, best_path)
            print(f"🏆 save best: {best_path}")

    print(f"FIN expid={expid} | best_auc={best_auc:.5f}")

if __name__ == "__main__":
    main()
