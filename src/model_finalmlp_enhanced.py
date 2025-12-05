import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Composants Avancés
# ==========================================

class DICE(nn.Module):
    """Activation DICE pour la performance CTR"""
    def __init__(self, emb_size, dim=2, epsilon=1e-8):
        super(DICE, self).__init__()
        self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim
        self.alpha = nn.Parameter(torch.zeros((emb_size,)))

    def forward(self, x):
        if self.dim == 2:
            x_p = self.bn(x)
            alpha = self.alpha.unsqueeze(0)
        else:
            x_p = self.bn(x.transpose(1, 2)).transpose(1, 2)
            alpha = self.alpha.view(1, 1, -1)
        gate = self.sigmoid(x_p)
        return gate * x + (1 - gate) * alpha * x

class AttentionPooling(nn.Module):
    """
    Attention DIN : Pondère l'historique en fonction de la cible.
    C'est ce qui manquait au FinalMLP standard.
    """
    def __init__(self, embedding_dim):
        super(AttentionPooling, self).__init__()
        # Input: Q, K, Q-K, Q*K
        self.mlp = nn.Sequential(
            nn.Linear(4 * embedding_dim, 64),
            DICE(64, dim=2),
            nn.Linear(64, 1)
        )

    def forward(self, query, history, mask):
        # query: (B, Emb)
        # history: (B, Seq, Emb)
        seq_len = history.size(1)
        queries = query.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Interactions
        att_input = torch.cat([queries, history, queries-history, queries*history], dim=-1)
        scores = self.mlp(att_input.view(-1, 4 * query.size(-1))).view(query.size(0), seq_len)
        
        scores = scores.masked_fill(mask, -1e9)
        weights = F.softmax(scores, dim=1)
        return torch.sum(weights.unsqueeze(-1) * history, dim=1)

class FeatureSelection(nn.Module):
    """Gating pour sélectionner les features par branche"""
    def __init__(self, num_fields, embedding_dim):
        super(FeatureSelection, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(num_fields * embedding_dim, num_fields),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, F, E)
        batch_size = x.size(0)
        flat_x = x.view(batch_size, -1)
        mask = self.gate(flat_x)
        return x * mask.unsqueeze(-1)

class CrossNetV2(nn.Module):
    """Interactions explicites d'ordre élevé"""
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
        x_0 = x
        for i in range(self.num_layers):
            xl_w = torch.matmul(x, self.kernels[i]) + self.bias[i]
            x = x_0 * xl_w + x
        return x

class FusionLayer(nn.Module):
    """Fusionne les 2 streams MLP + le stream CrossNet"""
    def __init__(self, input_dim):
        super(FusionLayer, self).__init__()
        self.w1 = nn.Linear(input_dim, 1, bias=False)
        self.w2 = nn.Linear(input_dim, 1, bias=False)
        self.w3 = nn.Linear(input_dim, 1, bias=False) # Interaction Stream1 * Stream2
        self.w_cross = nn.Linear(1, 1, bias=False)    # Poids pour le CrossNet
        self.bias = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, cross_logit):
        interaction = x1 * x2
        logit = self.bias + self.w1(x1) + self.w2(x2) + self.w3(interaction) + self.w_cross(cross_logit)
        return self.sigmoid(logit)

# ==========================================
# 2. Architecture Principale
# ==========================================

class MM_FinalMLP_Enhanced(nn.Module):
    def __init__(self, feature_map, model_cfg):
        super(MM_FinalMLP_Enhanced, self).__init__()
        
        self.emb_dim = model_cfg.get("embedding_dim", 128)
        mm_input_dim = 128
        
        # Embeddings
        self.item_emb = nn.Embedding(91718, self.emb_dim, padding_idx=0)
        self.cate_emb = nn.Embedding(11, self.emb_dim)
        
        # Projection MM
        self.mm_proj = nn.Sequential(
            nn.Linear(mm_input_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            DICE(self.emb_dim, dim=2)
        )
        
        # Attention pour l'historique (LE FIX MAJEUR)
        self.hist_attn = AttentionPooling(self.emb_dim)
        
        # 5 Champs : Like, View, ID, Image, Hist_Attn
        self.num_fields = 5 
        flatten_dim = self.num_fields * self.emb_dim
        
        # --- Dual Stream MLP ---
        self.fs1 = FeatureSelection(self.num_fields, self.emb_dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(flatten_dim, 512), nn.BatchNorm1d(512), DICE(512), nn.Dropout(0.2),
            nn.Linear(512, 128)
        )
        
        self.fs2 = FeatureSelection(self.num_fields, self.emb_dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(flatten_dim, 512), nn.BatchNorm1d(512), DICE(512), nn.Dropout(0.2),
            nn.Linear(512, 128)
        )
        
        # --- Cross Network Branch ---
        self.cross_net = CrossNetV2(flatten_dim, num_layers=3)
        self.cross_linear = nn.Linear(flatten_dim, 1)
        
        # --- Fusion ---
        self.fusion = FusionLayer(128)

    def forward(self, batch_dict):
        # 1. Données
        item_id = batch_dict['item_id'].long()
        item_mm = batch_dict['item_emb_d128'].float()
        likes = batch_dict['likes_level'].long()
        views = batch_dict['views_level'].long()
        hist_ids = batch_dict.get('item_seq', None)
        
        batch_size = item_id.size(0)
        
        # 2. Features de base
        like_feat = self.cate_emb(likes)
        view_feat = self.cate_emb(views)
        item_id_feat = self.item_emb(item_id)
        item_img_feat = self.mm_proj(item_mm)
        
        # Cible combinée (ID + Image) pour l'attention
        target_combined = item_id_feat + item_img_feat
        
        # 3. Historique avec Attention
        if hist_ids is not None:
            seq_emb = self.item_emb(hist_ids)
            mask = (hist_ids == 0)
            hist_feat = self.hist_attn(target_combined, seq_emb, mask)
        else:
            hist_feat = torch.zeros_like(item_id_feat)

        # 4. Stack & Flatten
        x_stack = torch.stack([like_feat, view_feat, item_id_feat, item_img_feat, hist_feat], dim=1)
        x_flat = x_stack.view(batch_size, -1)
        
        # 5. Calcul des 3 Branches
        # Stream 1 (Vision A)
        x1_sel = self.fs1(x_stack).view(batch_size, -1)
        y1 = self.mlp1(x1_sel)
        
        # Stream 2 (Vision B)
        x2_sel = self.fs2(x_stack).view(batch_size, -1)
        y2 = self.mlp2(x2_sel)
        
        # Stream 3 (Cross Interactions)
        cross_out = self.cross_net(x_flat)
        cross_logit = self.cross_linear(cross_out)
        
        # 6. Fusion
        return self.fusion(y1, y2, cross_logit).squeeze(-1)

def build_model(feature_map, model_cfg):
    return MM_FinalMLP_Enhanced(feature_map, model_cfg)