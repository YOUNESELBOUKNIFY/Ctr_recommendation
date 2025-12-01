import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

# ========================
# Dataset custom pour Parquet
# ========================
class ParquetDataset(Dataset):
    def __init__(self, data_path):
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
                array = np.array(df[col].to_list())
                seq_len = array.shape[1] if len(array.shape) > 1 else 1
                self.column_index[col] = [i + idx for i in range(seq_len)]
                idx += seq_len
            else:
                array = df[col].to_numpy()
                self.column_index[col] = idx
                idx += 1
            data_arrays.append(array)
        return np.column_stack(data_arrays)

# ========================
# Collator pour gérer batch
# ========================
class BatchCollator:
    def __init__(self, feature_map, max_len, column_index, item_info_path):
        self.feature_map = feature_map
        self.max_len = max_len
        self.column_index = column_index
        self.item_info = pd.read_parquet(item_info_path).set_index("item_id")

    def __call__(self, batch):
        batch_tensor = default_collate(batch)

        # Ici on ignore feature_map s'il est None
        all_cols = set(self.column_index.keys())

        batch_dict = dict()
        for col, idx in self.column_index.items():
            if col in all_cols:
                if isinstance(idx, list):
                    batch_dict[col] = batch_tensor[:, idx]
                else:
                    batch_dict[col] = batch_tensor[:, idx].unsqueeze(1)

        # Séquence item_seq padding
        batch_seqs = batch_dict["item_seq"][:, -self.max_len:]
        mask = (batch_seqs > 0).float()
        del batch_dict["item_seq"]

        # Récupérer item_info
        item_index = batch_dict["item_id"].reshape(-1).numpy()
        del batch_dict["item_id"]
        item_batch = self.item_info.loc[item_index]

        item_dict = dict()

        # Convertir item_tags correctement
        item_tags_list = item_batch["item_tags"].to_list()
        item_dict["item_tags"] = torch.tensor(
            np.stack([np.array(x, dtype=np.int64) for x in item_tags_list]),
            dtype=torch.long
        )

        # Convertir item_emb_d128 correctement
        item_emb_list = item_batch["item_emb_d128"].to_list()
        item_emb_array = np.stack([np.array(x, dtype=np.float32) for x in item_emb_list])
        item_dict["item_emb_d128"] = torch.tensor(item_emb_array, dtype=torch.float)

        return batch_dict, item_dict, mask

# ========================
# DataLoader custom
# ========================
class MMCTRDataLoader(DataLoader):
    def __init__(self, feature_map, data_path, item_info_path, batch_size=32, shuffle=False,
                 num_workers=1, max_len=100, **kwargs):

        if not data_path.endswith(".parquet"):
            data_path += ".parquet"

        self.dataset = ParquetDataset(data_path)
        self.column_index = self.dataset.column_index
        self.batch_size = batch_size

        super().__init__(dataset=self.dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers,
                         collate_fn=BatchCollator(feature_map, max_len, self.column_index, item_info_path))

        self.num_samples = len(self.dataset)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))

    def __len__(self):
        return self.num_batches
