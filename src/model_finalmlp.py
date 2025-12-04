import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureSelection(nn.Module):
    """
    Module de sélection de features (Gating).
    Permet à chaque flux de choisir ses propres features importantes.
    """
    def __init__(self, num_fields, embedding_dim, dropout=0.0):
        super(FeatureSelection, self).__init__()
        # Un poids par feature et par dimension (Gate fine-grain)
        self.gate = nn.Sequential(
            nn.Linear(num_fields * embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_fields), # Un score par champ
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, Fields, Emb)
        batch_size, num_fields, emb_dim = x.shape
        flat_x = x.view(batch_size, -1)
        
        # Calcul des scores d'importance (Mask)
        # (B, Fields)
        mask = self.gate(flat_x)
        
        # Application du masque : (B, F, E) * (B, F, 1)
        return x * mask.unsqueeze(-1)

class InteractionBlock(nn.Module):
    """
    Bloc MLP classique pour traiter les features sélectionnées.
    """
    def __init__(self, input_dim, hidden_units, dropout=0.1):
        super(InteractionBlock, self).__init__()
        layers = []
        in_dim = input_dim
        for out_dim in hidden_units:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU()) # ReLU est plus stable ici que DICE pour FinalMLP
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        self.out_dim = in_dim

    def forward(self, x):
        return self.mlp(x)

class FusionLayer(nn.Module):
    """
    Fusion Bilinéaire des deux flux.
    Out = w0 * (Stream1 * Stream2) + w1 * Stream1 + w2 * Stream2
    """
    def __init__(self, input_dim):
        super(FusionLayer, self).__init__()
        self.linear1 = nn.Linear(input_dim, 1, bias=False)
        self.linear2 = nn.Linear(input_dim, 1, bias=False)
        self.linear3 = nn.Linear(input_dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # x1, x2: (B, Hidden_Size)
        # Interaction bilinéaire (Element-wise product)
        interact = x1 * x2
        
        # Fusion pondérée
        logits = self.linear1(interact) + self.linear2(x1) + self.linear3(x2)
        return self.sigmoid(logits)

class MM_FinalMLP(nn.Module):
    def __init__(self, feature_map, model_cfg):
        super(MM_FinalMLP, self).__init__()
        
        self.emb_dim = model_cfg.get("embedding_dim", 128)
        mm_input_dim = 128
        
        # --- Embeddings ---
        self.item_emb = nn.Embedding(91718, self.emb_dim, padding_idx=0)
        self.cate_emb = nn.Embedding(11, self.emb_dim) # Likes/Views
        
        # Projection MM
        self.mm_proj = nn.Sequential(
            nn.Linear(mm_input_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.ReLU()
        )
        
        # Champs : [Like, View, ID, Image, Hist]
        self.num_fields = 5 
        
        # --- Stream 1 (Vision globale) ---
        self.fs1 = FeatureSelection(self.num_fields, self.emb_dim)
        self.mlp1 = InteractionBlock(
            self.num_fields * self.emb_dim, 
            [512, 256, 128]
        )
        
        # --- Stream 2 (Vision focalisée) ---
        self.fs2 = FeatureSelection(self.num_fields, self.emb_dim)
        self.mlp2 = InteractionBlock(
            self.num_fields * self.emb_dim, 
            [512, 256, 128]
        )
        
        # --- Fusion ---
        # Les deux MLP doivent finir avec la même dimension (128)
        self.fusion = FusionLayer(128)

    def forward(self, batch_dict):
        # 1. Inputs & Embeddings
        item_id = batch_dict['item_id'].long()
        item_mm = batch_dict['item_emb_d128'].float()
        likes = batch_dict['likes_level'].long()
        views = batch_dict['views_level'].long()
        hist_ids = batch_dict.get('item_seq', None)
        
        # Features
        like_feat = self.cate_emb(likes)
        view_feat = self.cate_emb(views)
        item_id_feat = self.item_emb(item_id)
        item_img_feat = self.mm_proj(item_mm)
        
        # Historique (Mean Pooling)
        if hist_ids is not None:
            seq_emb = self.item_emb(hist_ids)
            mask = (hist_ids == 0)
            seq_emb_masked = seq_emb * (~mask.unsqueeze(-1)).float()
            seq_sum = torch.sum(seq_emb_masked, dim=1)
            seq_count = torch.sum((~mask).float(), dim=1, keepdim=True).clamp(min=1)
            hist_feat = seq_sum / seq_count
        else:
            hist_feat = torch.zeros_like(item_id_feat)

        # Stack : (B, 5, Emb)
        x = torch.stack([
            like_feat, view_feat, item_id_feat, item_img_feat, hist_feat
        ], dim=1)
        
        # --- Stream 1 ---
        x1 = self.fs1(x)                # Selection
        x1 = x1.view(x.size(0), -1)     # Flatten
        y1 = self.mlp1(x1)              # MLP
        
        # --- Stream 2 ---
        x2 = self.fs2(x)                # Selection Différente
        x2 = x2.view(x.size(0), -1)     # Flatten
        y2 = self.mlp2(x2)              # MLP
        
        # --- Fusion ---
        output = self.fusion(y1, y2)
        
        return output.squeeze(-1)

def build_model(feature_map, model_cfg):
    return MM_FinalMLP(feature_map, model_cfg)