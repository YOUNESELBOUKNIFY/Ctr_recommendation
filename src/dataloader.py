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
            # Gestion des colonnes de type liste/objet
            if df[col].dtype == "object":
                # On convertit les listes en tableau numpy 2D
                # Attention: suppose que toutes les listes ont la même longueur ici
                # Sinon il faut padder avant.
                col_list = df[col].to_list()
                array = np.array(col_list)
                
                # Si c'est 1D (liste simple), on reshape
                if len(array.shape) == 1:
                    array = array.reshape(-1, 1)
                    
                seq_len = array.shape[1]
                self.column_index[col] = [i + idx for i in range(seq_len)]
                idx += seq_len
            else:
                array = df[col].to_numpy().reshape(-1, 1)
                self.column_index[col] = idx
                idx += 1
            data_arrays.append(array)
        
        # Concaténation horizontale sécurisée
        return np.column_stack(data_arrays)

# ========================
# Collator (CORRIGÉ)
# ========================
class BatchCollator:
    def __init__(self, feature_map, max_len, column_index, item_info_path):
        self.feature_map = feature_map
        self.max_len = max_len
        self.column_index = column_index
        # Chargement en mémoire (attention si dataset énorme, mais ok pour 1M)
        self.item_info = pd.read_parquet(item_info_path).set_index("item_id")
        
        # Pré-traitement des embeddings items pour gagner du temps
        # On convertit la colonne embedding (list) en une matrice numpy
        print("Chargement des embeddings items en mémoire...")
        emb_list = self.item_info["item_emb_d128"].to_list()
        self.item_embedding_matrix = np.stack([np.array(x, dtype=np.float32) for x in emb_list])
        # Mapping item_id -> index dans la matrice (si les item_id ne sont pas contigus de 0 à N)
        # Ici on suppose que item_info est indexé par item_id
        
    def __call__(self, batch):
        # 1. Collation par défaut de PyTorch pour empiler les lignes
        batch_tensor = default_collate(batch)
        
        batch_dict = {}
        
        # 2. Reconstruire le dictionnaire à partir des indices
        for col, idx in self.column_index.items():
            if isinstance(idx, list):
                batch_dict[col] = batch_tensor[:, idx]
            else:
                batch_dict[col] = batch_tensor[:, idx].squeeze(-1) # Enlever dim inutile

        # 3. Récupérer les Item IDs pour le lookup
        # Important : On garde item_id dans batch_dict pour le modèle !
        item_ids = batch_dict["item_id"].numpy()
        
        # 4. Lookup des Embeddings Multimodaux (Image/Texte 128d)
        # On utilise .loc pour récupérer les embeddings correspondant aux IDs
        # Note: Si item_ids contient des IDs inconnus, cela plantera.
        try:
            # Récupération rapide via l'index pandas
            batch_item_info = self.item_info.loc[item_ids]
            
            # Extraction des embeddings
            emb_vals = np.stack(batch_item_info["item_emb_d128"].values)
            emb_tensor = torch.tensor(emb_vals, dtype=torch.float32)
            
            # Ajout au dictionnaire
            batch_dict["item_emb_d128"] = emb_tensor
            
            # Si vous voulez utiliser les tags :
            # tag_vals = np.stack(batch_item_info["item_tags"].values)
            # batch_dict["item_tags"] = torch.tensor(tag_vals, dtype=torch.long)

        except KeyError as e:
            print(f"Erreur: Certains item_ids du batch ne sont pas dans item_info. {e}")
            raise e

        # 5. Gestion de item_seq (Historique utilisateur)
        # Pour TabTransformer simple, on ne l'utilise pas souvent, 
        # mais on le prépare au cas où.
        if "item_seq" in batch_dict:
            seq = batch_dict["item_seq"]
            # Garder seulement les max_len derniers items
            if seq.shape[1] > self.max_len:
                seq = seq[:, -self.max_len:]
            batch_dict["item_seq"] = seq.long()

        # Retourne un seul gros dictionnaire propre + le label
        labels = batch_dict.pop("label").float()
        
        return batch_dict, labels

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
        
        # Init du Collator avec le chemin vers item_info
        collator = BatchCollator(feature_map, max_len, self.column_index, item_info_path)

        super().__init__(dataset=self.dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers,
                         collate_fn=collator)