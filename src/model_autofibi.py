import torch
import torch.nn as nn
import torch.nn.functional as F

class SENetLayer(nn.Module):
    """
    Squeeze-and-Excitation : Pondère l'importance des champs individuels.
    """
    def __init__(self, num_fields, reduction_ratio=3):
        super(SENetLayer, self).__init__()
        reduced_size = max(1, num_fields // reduction_ratio)
        self.excitation = nn.Sequential(
            nn.Linear(num_fields, reduced_size),
            nn.ReLU(),
            nn.Linear(reduced_size, num_fields),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (Batch, Num_Fields, Emb)
        z = torch.mean(x, dim=-1)
        weights = self.excitation(z)
        return x * weights.unsqueeze(-1)

class BilinearInteraction(nn.Module):
    """
    Interaction Bilinéaire (Type 'Each') pour la précision.
    """
    def __init__(self, input_dim, num_fields):
        super(BilinearInteraction, self).__init__()
        self.W_list = nn.ParameterList([
            nn.Parameter(torch.Tensor(input_dim, input_dim)) for _ in range(num_fields - 1)
        ])
        for w in self.W_list: nn.init.xavier_normal_(w)

    def forward(self, x):
        inputs = torch.split(x, 1, dim=1)
        p = []
        for i in range(len(inputs) - 1):
            if i < len(self.W_list):
                vid = torch.matmul(inputs[i].squeeze(1), self.W_list[i])
                for j in range(i + 1, len(inputs)):
                    # p_ij = v_i * W_i * v_j
                    p.append(vid * inputs[j].squeeze(1))
        return torch.stack(p, dim=1) # (B, Num_Pairs, Emb)

class InteractionGate(nn.Module):
    """
    MODULE AUTO : Apprend à filtrer les interactions inutiles.
    Pour chaque paire (i, j), apprend un poids w_ij.
    """
    def __init__(self, num_pairs):
        super(InteractionGate, self).__init__()
        # Un poids par paire d'interaction
        self.gate_weights = nn.Parameter(torch.Tensor(num_pairs))
        nn.init.constant_(self.gate_weights, 0.5) # Initialisé à 0.5 (neutre)
        self.sigmoid = nn.Sigmoid()

    def forward(self, interactions):
        # interactions: (B, Num_Pairs, Emb)
        
        # Calcul des poids de porte (B, Num_Pairs, 1)
        # On utilise une sigmoid pour avoir un score entre 0 et 1
        gates = self.sigmoid(self.gate_weights).view(1, -1, 1)
        
        # Filtrage: Interaction * Gate
        return interactions * gates

class MM_AutoFiBi(nn.Module):
    def __init__(self, feature_map, model_cfg):
        super(MM_AutoFiBi, self).__init__()
        
        self.emb_dim = model_cfg.get("embedding_dim", 128)
        mm_input_dim = 128
        
        # --- Embeddings ---
        self.item_emb = nn.Embedding(91718, self.emb_dim, padding_idx=0)
        self.cate_emb = nn.Embedding(11, self.emb_dim) # Likes/Views
        self.user_emb = nn.Embedding(20000, self.emb_dim) # Placeholder
        
        self.mm_proj = nn.Sequential(
            nn.Linear(mm_input_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.ReLU()
        )
        
        # 6 Champs
        self.num_fields = 6 
        
        # --- FiBiNET Core ---
        self.senet = SENetLayer(self.num_fields, reduction_ratio=2)
        self.bilinear = BilinearInteraction(self.emb_dim, self.num_fields)
        
        # --- AUTO LAYER ---
        num_pairs = (self.num_fields * (self.num_fields - 1)) // 2
        self.auto_gate = InteractionGate(num_pairs)
        
        # --- MLP ---
        # Input = SENet Features (Originales) + Gated Bilinear (Interactions)
        total_input_dim = (self.num_fields * self.emb_dim) + (num_pairs * self.emb_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(model_cfg.get("net_dropout", 0.2)),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(model_cfg.get("net_dropout", 0.2)),
            
            nn.Linear(256, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch_dict):
        # 1. Inputs
        item_id = batch_dict['item_id'].long()
        item_mm = batch_dict['item_emb_d128'].float()
        likes = batch_dict['likes_level'].long()
        views = batch_dict['views_level'].long()
        hist_ids = batch_dict.get('item_seq', None)
        
        batch_size = item_id.size(0)
        
        # 2. Features
        user_feat = torch.zeros((batch_size, self.emb_dim), device=item_id.device)
        like_feat = self.cate_emb(likes)
        view_feat = self.cate_emb(views)
        item_id_feat = self.item_emb(item_id)
        item_img_feat = self.mm_proj(item_mm)
        
        if hist_ids is not None:
            seq_emb = self.item_emb(hist_ids)
            mask = (hist_ids == 0)
            seq_emb = seq_emb.masked_fill(mask.unsqueeze(-1), 0)
            seq_sum = torch.sum(seq_emb, dim=1)
            seq_count = torch.sum((~mask).float(), dim=1, keepdim=True).clamp(min=1)
            hist_feat = seq_sum / seq_count
        else:
            hist_feat = torch.zeros_like(item_id_feat)

        # Stack
        raw_inputs = torch.stack([
            user_feat, like_feat, view_feat, item_id_feat, item_img_feat, hist_feat
        ], dim=1)
        
        # 3. SENet (Pondération des champs)
        senet_output = self.senet(raw_inputs)
        
        # 4. Bilinear (Toutes les paires)
        bilinear_output = self.bilinear(senet_output)
        
        # 5. AUTO-GATING (Filtrage intelligent)
        # C'est ici que AutoFiBi fait la différence : il atténue les paires bruyantes
        gated_bilinear = self.auto_gate(bilinear_output)
        
        # 6. Fusion
        c_input = torch.cat([
            senet_output.view(batch_size, -1),   # Features originales pondérées
            gated_bilinear.view(batch_size, -1)  # Interactions filtrées
        ], dim=1)
        
        return self.sigmoid(self.mlp(c_input)).squeeze(-1)

def build_model(feature_map, model_cfg):
    return MM_AutoFiBi(feature_map, model_cfg)