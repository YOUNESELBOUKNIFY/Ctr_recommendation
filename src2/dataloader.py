import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

# ========================
# Dataset custom pour Parquet
# ========================
class ParquetDataset(Dataset):
    """
    Charge un parquet et retourne chaque ligne sous forme d'un vecteur numpy 1D.
    Si une colonne est de type object et contient des listes (ex: item_seq),
    elle est expand en plusieurs colonnes (une par position).
    """
    def __init__(self, data_path: str):
        self.column_index = dict()
        self.darray = self.load_data(data_path)

    def __getitem__(self, index):
        return self.darray[index, :]

    def __len__(self):
        return self.darray.shape[0]

    def load_data(self, data_path):
        df = pd.read_parquet(data_path)
        data_arrays = []
        idx = 0

        for col in df.columns:
            if df[col].dtype == "object":
                col_list = df[col].to_list()
                array = np.array(col_list, dtype=object)

                # Convertir en 2D si possible (listes de longueur fixe)
                # Si certaines lignes sont None, on remplace par liste vide.
                cleaned = []
                max_len = 0
                for x in col_list:
                    if x is None:
                        x = []
                    if isinstance(x, (list, np.ndarray)):
                        max_len = max(max_len, len(x))
                    cleaned.append(x)

                # Pad si nécessaire pour faire un rectangle
                padded = []
                for x in cleaned:
                    if not isinstance(x, (list, np.ndarray)):
                        x = [x]
                    x = list(x)
                    if len(x) < max_len:
                        x = x + [0] * (max_len - len(x))
                    padded.append(x)

                array = np.asarray(padded, dtype=np.int64)  # item_seq est souvent int
                if array.ndim == 1:
                    array = array.reshape(-1, 1)

                seq_len = array.shape[1]
                self.column_index[col] = [i + idx for i in range(seq_len)]
                idx += seq_len
            else:
                array = df[col].to_numpy().reshape(-1, 1)
                self.column_index[col] = idx
                idx += 1

            data_arrays.append(array)

        return np.column_stack(data_arrays)


# ========================
# Collator Training
# ========================
class BatchCollator:
    """
    Rebuild batch_dict depuis un batch_tensor, puis injecte item_emb_d128 via item_info.
    """
    def __init__(self, max_len, column_index, item_info_path):
        self.max_len = max_len
        self.column_index = column_index

        print("Chargement item_info (train/valid) ...")
        self.item_info = pd.read_parquet(item_info_path).set_index("item_id")

    def __call__(self, batch):
        batch_tensor = default_collate(batch)
        batch_dict = {}

        for col, idx in self.column_index.items():
            if isinstance(idx, list):
                batch_dict[col] = batch_tensor[:, idx]
            else:
                batch_dict[col] = batch_tensor[:, idx].squeeze(-1)

        # item_id pour lookup embedding multimodal
        item_ids = batch_dict["item_id"].detach().cpu().numpy()

        # reindex = robuste (si ID manquant -> NaN)
        batch_item_info = self.item_info.reindex(item_ids)

        # stack embeddings (list/np.array) ou 0 si manquant
        emb_vals = []
        for x in batch_item_info["item_emb_d128"].values:
            if isinstance(x, (list, np.ndarray)):
                emb_vals.append(np.asarray(x, dtype=np.float32))
            else:
                emb_vals.append(np.zeros((128,), dtype=np.float32))
        emb_vals = np.stack(emb_vals, axis=0)
        batch_dict["item_emb_d128"] = torch.tensor(emb_vals, dtype=torch.float32)

        # history item_seq
        if "item_seq" in batch_dict:
            seq = batch_dict["item_seq"]
            if seq.ndim == 1:
                seq = seq.unsqueeze(1)
            if seq.shape[1] > self.max_len:
                seq = seq[:, -self.max_len:]
            batch_dict["item_seq"] = seq.long()

        labels = batch_dict.pop("label").float()
        return batch_dict, labels


# ========================
# DataLoader custom
# ========================
class MMCTRDataLoader(DataLoader):
    def __init__(
        self,
        data_path,
        item_info_path,
        batch_size=4096,
        shuffle=False,
        num_workers=4,
        max_len=20,
        **kwargs
    ):
        if not data_path.endswith(".parquet"):
            data_path += ".parquet"

        dataset = ParquetDataset(data_path)
        collator = BatchCollator(max_len=max_len, column_index=dataset.column_index, item_info_path=item_info_path)

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collator,
            **kwargs
        )
