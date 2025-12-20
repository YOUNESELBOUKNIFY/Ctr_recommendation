import argparse
import os
import sys
import yaml
import zipfile
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloader import ParquetDataset
from model_fibinet import build_model

class InferenceCollator:
    def __init__(self, max_len, column_index, item_info_path):
        self.max_len = max_len
        self.column_index = column_index
        print("Chargement item_info (infer) ...")
        self.item_info = pd.read_parquet(item_info_path).set_index("item_id")

    def __call__(self, batch):
        batch_tensor = default_collate(batch)
        batch_dict = {}

        for col, idx in self.column_index.items():
            if isinstance(idx, list):
                batch_dict[col] = batch_tensor[:, idx]
            else:
                batch_dict[col] = batch_tensor[:, idx].squeeze(-1)

        item_ids = batch_dict["item_id"].detach().cpu().numpy()
        batch_item_info = self.item_info.reindex(item_ids)

        emb_vals = []
        for x in batch_item_info["item_emb_d128"].values:
            if isinstance(x, (list, np.ndarray)):
                emb_vals.append(np.asarray(x, dtype=np.float32))
            else:
                emb_vals.append(np.zeros((128,), dtype=np.float32))
        emb_vals = np.stack(emb_vals, axis=0)

        batch_dict["item_emb_d128"] = torch.tensor(emb_vals, dtype=torch.float32)

        if "item_seq" in batch_dict:
            seq = batch_dict["item_seq"]
            if seq.ndim == 1:
                seq = seq.unsqueeze(1)
            if seq.shape[1] > self.max_len:
                seq = seq[:, -self.max_len:]
            batch_dict["item_seq"] = seq.long()

        return batch_dict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["test", "valid"])
    ap.add_argument("--out_dir", type=str, default="preds")
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_id = cfg["dataset_id"]
    dataset_cfg = cfg["dataset_config"][dataset_id]
    expid = cfg["base_expid"]
    model_cfg = cfg[expid]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | expid={expid} | split={args.split}")

    model = build_model(None, model_cfg)

    sd = torch.load(args.checkpoint, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)

    model.to(device)
    model.eval()

    data_path = dataset_cfg["test_data"] if args.split == "test" else dataset_cfg["valid_data"]
    ds = ParquetDataset(data_path)

    collator = InferenceCollator(
        max_len=int(model_cfg.get("max_len", 20)),
        column_index=ds.column_index,
        item_info_path=dataset_cfg["item_info"]
    )

    loader = DataLoader(
        dataset=ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator
    )

    all_preds = []
    with torch.no_grad():
        for batch_dict in tqdm(loader):
            for k, v in batch_dict.items():
                batch_dict[k] = v.to(device)
            y_pred = model(batch_dict)
            all_preds.append(y_pred.detach().cpu().numpy())

    preds = np.concatenate(all_preds).reshape(-1)

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"prediction_{expid}_{args.split}.csv")

    sub = pd.DataFrame({"ID": np.arange(len(preds)), "Task2": preds})
    sub.to_csv(out_csv, index=False)

    out_zip = os.path.join(args.out_dir, f"submission_{expid}_{args.split}.zip")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_csv, arcname=os.path.basename(out_csv))

    print("✅ saved:", out_csv)
    print("✅ zipped:", out_zip)

if __name__ == "__main__":
    main()
