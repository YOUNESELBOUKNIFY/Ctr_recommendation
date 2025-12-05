import torch
import torch.nn as nn
import torch.nn.functional as F

class SENetLayer(nn.Module):
    """Squeeze-and-Excitation Network (Optimisé)"""
    def __init__(self, num_fields, reduction_ratio=2):
        super(SENetLayer, self).__init__()
        reduced_size = max(1, num_fields // reduction_ratio)
        self.excitation = nn.Sequential(
            nn.Linear(num_fields, reduced_size),
            nn.GELU(),
            nn.Linear(reduced_size, num_fields),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = torch.mean(x, dim=-1)
        weights = self.excitation(z)
        return x * weights.unsqueeze(-1)

class BilinearInteraction(nn.Module):
    """Interaction Bilinéaire (Type 'Each')"""
    def __init__(self, input_dim, num_fields):
        super(BilinearInteraction, self).__init__()
        # Une matrice par champ
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
                    p.append(vid * inputs[j].squeeze(1))
        return torch.stack(p, dim=1)

class CrossNetV2(nn.Module):
    """
    Deep Cross Network V2.
    Capture les interactions d'ordre élevé (High-order features).
    Formule: x_{l+1} = x_0 * (W_l * x_l + b_l) + x_l
    """
    def __init__(self, input_dim, num_layers=3):
        super(CrossNetV2, self).__init__()
        self.num_layers = num_layers
        self.kernels = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, input_dim)) for _ in range(num_layers)
        ])
        self.bias = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)
        ])
        for w in self.kernels: nn.init.xavier_normal_(w)

    def forward(self, x):
        # x: (Batch, Total_Flatten_Dim)
        x_0 = x
        for i in range(self.num_layers):
            xl_w = torch.matmul(x, self.kernels[i]) + self.bias[i]
            # Interaction explicite + Connexion résiduelle
            x = x_0 * xl_w + x
        return x

class MM_DeepFiBiNET(nn.Module):
    def __init__(self, feature_map, model_cfg):
        super(MM_DeepFiBiNET, self).__init__()
        
        self.emb_dim = model_cfg.get("embedding_dim", 128)
        mm_input_dim = 128
        
        # --- Embeddings ---
        self.item_emb = nn.Embedding(91718, self.emb_dim, padding_idx=0)
        self.cate_emb = nn.Embedding(11, self.emb_dim)
        # On apprend le vecteur User moyen
        self.user_emb = nn.Parameter(torch.zeros(1, self.emb_dim)) 
        
        # Projection Multimodale
        self.mm_proj = nn.Sequential(
            nn.Linear(mm_input_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.GELU()
        )
        
        # 6 Champs : [User, Like, View, Item_ID, Item_Image, Hist]
        self.num_fields = 6 
        
        # --- Branche 1 : FiBiNET ---
        self.senet = SENetLayer(self.num_fields, reduction_ratio=2)
        self.bilinear = BilinearInteraction(self.emb_dim, self.num_fields)
        
        # --- Branche 2 : DCN V2 ---
        # Input size pour DCN = Num_Fields * Emb_Dim
        dcn_input_dim = self.num_fields * self.emb_dim
        self.crossnet = CrossNetV2(dcn_input_dim, num_layers=3)
        
        # --- Fusion & MLP Final ---
        # FiBiNET output size (Paires bilinéaires)
        num_pairs = (self.num_fields * (self.num_fields - 1)) // 2
        fibinet_out_dim = num_pairs * self.emb_dim
        
        # Taille totale = Sortie Bilinéaire + Sortie CrossNet
        total_input_dim = fibinet_out_dim + dcn_input_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(model_cfg.get("net_dropout", 0.2)),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(model_cfg.get("net_dropout", 0.2)),
            
            nn.Linear(512, 1)
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
        
        # 2. Features Stack
        user_feat = self.user_emb.expand(batch_size, -1)
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

        # (Batch, 6, 128)
        sparse_inputs = torch.stack([
            user_feat, like_feat, view_feat, item_id_feat, item_img_feat, hist_feat
        ], dim=1)
        
        # --- Branche FiBiNET ---
        senet_output = self.senet(sparse_inputs)
        bilinear_output = self.bilinear(senet_output) # (B, 15, 128)
        bilinear_flat = bilinear_output.view(batch_size, -1)
        
        # --- Branche CrossNet ---
        # DCN prend l'input original concaténé
        dcn_input = sparse_inputs.view(batch_size, -1) # (B, 6*128)
        cross_output = self.crossnet(dcn_input)
        
        # --- Fusion ---
        # On concatène les interactions implicites (Bilinear) et explicites (Cross)
        combined_input = torch.cat([bilinear_flat, cross_output], dim=1)
        
        logits = self.mlp(combined_input)
        return self.sigmoid(logits).squeeze(-1)

def build_model(feature_map, model_cfg):
    return MM_DeepFiBiNET(feature_map, model_cfg)