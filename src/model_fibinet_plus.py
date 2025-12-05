import torch
import torch.nn as nn
import torch.nn.functional as F

class SENetLayer(nn.Module):
    """Squeeze-and-Excitation : Recalibre l'importance des features"""
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
        z = torch.mean(x, dim=-1) 
        weights = self.excitation(z) 
        return x * weights.unsqueeze(-1)

class BilinearInteraction(nn.Module):
    """
    Interaction Bilinéaire améliorée (Type 'Each').
    Utilise une matrice spécifique pour chaque champ, plus précis que 'All'.
    """
    def __init__(self, input_dim, num_fields):
        super(BilinearInteraction, self).__init__()
        # Une matrice W par champ (sauf le dernier qui n'en a pas besoin pour les paires)
        self.W_list = nn.ParameterList([
            nn.Parameter(torch.Tensor(input_dim, input_dim)) for _ in range(num_fields - 1)
        ])
        for w in self.W_list:
            nn.init.xavier_normal_(w)

    def forward(self, x):
        # x: (Batch, Num_Fields, Emb)
        inputs = torch.split(x, 1, dim=1)
        p = []
        # Interaction Field-Each : v_i * W_i * v_j
        for i in range(len(inputs) - 1):
            # Projection de v_i une seule fois
            vid = torch.matmul(inputs[i].squeeze(1), self.W_list[i]) 
            for j in range(i + 1, len(inputs)):
                # Produit de Hadamard
                p.append(vid * inputs[j].squeeze(1))
        return torch.stack(p, dim=1)

class CrossNetV2(nn.Module):
    """
    DCN V2 : Capture les interactions explicites d'ordre élevé.
    Très efficace en parallèle du MLP.
    """
    def __init__(self, input_dim, num_layers=2):
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
        x_0 = x
        for i in range(self.num_layers):
            # x_l+1 = x_0 * (W * x_l + b) + x_l
            xl_w = torch.matmul(x, self.kernels[i]) + self.bias[i]
            x = x_0 * xl_w + x
        return x

class TargetAwareAttention(nn.Module):
    """Attention simplifiée pour l'historique (inspirée de DIN)"""
    def __init__(self, input_dim):
        super(TargetAwareAttention, self).__init__()
        # Q, K, Q-K, Q*K -> 4 * dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, target, history, mask):
        # target: (B, Emb)
        # history: (B, Seq, Emb)
        seq_len = history.size(1)
        target = target.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Interactions
        x = torch.cat([target, history, target-history, target*history], dim=-1)
        scores = self.mlp(x).squeeze(-1) # (B, Seq)
        
        # Masking
        scores = scores.masked_fill(mask, -1e9)
        weights = F.softmax(scores, dim=1)
        
        # Weighted Sum
        return torch.sum(weights.unsqueeze(-1) * history, dim=1)

class MM_FiBiNET_Plus(nn.Module):
    def __init__(self, feature_map, model_cfg):
        super(MM_FiBiNET_Plus, self).__init__()
        
        self.emb_dim = model_cfg.get("embedding_dim", 128)
        mm_input_dim = 128
        
        # --- Embeddings ---
        self.item_emb = nn.Embedding(91718, self.emb_dim, padding_idx=0)
        self.cate_emb = nn.Embedding(11, self.emb_dim) # Likes/Views
        self.user_emb = nn.Embedding(20000, self.emb_dim) # Placeholder
        
        # Projection Multimodale
        self.mm_proj = nn.Sequential(
            nn.Linear(mm_input_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.GELU() # GELU souvent mieux que ReLU
        )
        
        # --- Modules Avancés ---
        self.hist_attn = TargetAwareAttention(self.emb_dim)
        
        # 5 Champs : Like, View, ID, Image, Hist_Attn
        self.num_fields = 5 
        
        # FiBiNET Core
        self.senet = SENetLayer(self.num_fields, reduction_ratio=2) # Ratio plus faible = plus précis
        self.bilinear = BilinearInteraction(self.emb_dim, self.num_fields)
        
        # Calcul dimensions
        num_pairs = (self.num_fields * (self.num_fields - 1)) // 2
        
        # Dimension totale après concatenation (SENet Original + Bilinear)
        total_dim = (self.num_fields + num_pairs) * self.emb_dim
        
        # DCN Branch
        self.cross_net = CrossNetV2(total_dim, num_layers=2)
        
        # MLP Final
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.LayerNorm(512), # LayerNorm stabilise mieux les gros modèles
            nn.GELU(),
            nn.Dropout(model_cfg.get("net_dropout", 0.2)),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
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
        
        # 2. Features
        like_feat = self.cate_emb(likes)
        view_feat = self.cate_emb(views)
        item_id_feat = self.item_emb(item_id)
        item_img_feat = self.mm_proj(item_mm)
        
        # Target combinée pour l'attention (ID + Image)
        target_combined = item_id_feat + item_img_feat
        
        # 3. Historique avec Attention (Amélioration majeure vs MeanPooling)
        if hist_ids is not None:
            seq_emb = self.item_emb(hist_ids)
            mask = (hist_ids == 0)
            hist_feat = self.hist_attn(target_combined, seq_emb, mask)
        else:
            hist_feat = torch.zeros_like(item_id_feat)

        # 4. Stack Fields
        sparse_inputs = torch.stack([
            like_feat, view_feat, item_id_feat, item_img_feat, hist_feat
        ], dim=1)
        
        # 5. FiBiNET Flow
        senet_output = self.senet(sparse_inputs)
        bilinear_output = self.bilinear(senet_output)
        
        # 6. Fusion Flat
        # (B, Num_Fields + Num_Pairs, Emb) -> (B, Total_Dim)
        dnn_input = torch.cat([
            senet_output.view(item_id.size(0), -1),
            bilinear_output.view(item_id.size(0), -1)
        ], dim=1)
        
        # 7. Cross Network & MLP (En parallèle)
        cross_out = self.cross_net(dnn_input)
        mlp_out = self.mlp(dnn_input) # On peut aussi faire self.mlp(cross_out)
        
        # Somme des logits (Residual connection du pauvre mais efficace)
        # On combine la puissance explicite (Cross) et implicite (MLP)
        final_logit = mlp_out + torch.mean(cross_out, dim=1, keepdim=True) # Projection simple ou Linear
        # Pour simplifier et rester stable, on passe cross_out dans le MLP ou on fait une somme pondérée
        # Ici on va rester simple : MLP prend la sortie CrossNet
        
        # Version stable : CrossNet -> MLP
        # dnn_input -> CrossNet -> MLP -> Logit
        combined_out = self.cross_net(dnn_input)
        logits = self.mlp(combined_out)
        
        return self.sigmoid(logits).squeeze(-1)

def build_model(feature_map, model_cfg):
    return MM_FiBiNET_Plus(feature_map, model_cfg)