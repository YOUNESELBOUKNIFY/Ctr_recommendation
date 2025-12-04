import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Composants Avancés
# ==========================================

class DICE(nn.Module):
    """Activation adaptative pour maximiser la performance sur les données CTR."""
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

class SENetLayer(nn.Module):
    """Apprend l'importance de chaque Feature (Field)."""
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
        # x: (B, Num_Fields, Emb)
        z = torch.mean(x, dim=-1) # Squeeze
        weights = self.excitation(z) # Excitation
        return x * weights.unsqueeze(-1) # Reweight

class BilinearInteraction(nn.Module):
    """Interactions de second ordre (Produit de Hadamard)."""
    def __init__(self, input_dim, num_fields, bilinear_type="all"):
        super(BilinearInteraction, self).__init__()
        self.bilinear_type = bilinear_type
        if bilinear_type == "all":
            self.W = nn.Parameter(torch.Tensor(input_dim, input_dim))
            nn.init.xavier_normal_(self.W)

    def forward(self, x):
        # x: (B, Num_Fields, Emb)
        inputs = torch.split(x, 1, dim=1)
        p = []
        if self.bilinear_type == "all":
            vid = torch.matmul(x, self.W) # Projection globale
            for i in range(len(inputs)):
                for j in range(i + 1, len(inputs)):
                    # Interaction: v_i * (W . v_j)
                    p.append(inputs[i].squeeze(1) * vid[:, j, :])
        return torch.stack(p, dim=1)

class AttentionPooling(nn.Module):
    """
    Attention de type DIN pour l'historique.
    Extrait l'information pertinente de l'historique par rapport à l'item cible.
    """
    def __init__(self, embedding_dim):
        super(AttentionPooling, self).__init__()
        # Input: Query(Target) + Key(Hist) + Q-K + Q*K
        self.mlp = nn.Sequential(
            nn.Linear(4 * embedding_dim, 64),
            nn.Sigmoid(), # Sigmoid simple ici pour la rapidité
            nn.Linear(64, 1)
        )

    def forward(self, query, history, mask):
        # query: (B, Emb)
        # history: (B, Seq, Emb)
        # mask: (B, Seq) True si padding
        
        seq_len = history.size(1)
        queries = query.unsqueeze(1).expand(-1, seq_len, -1) # (B, Seq, Emb)
        
        # Interactions pour le calcul du score
        att_input = torch.cat([
            queries, history, queries - history, queries * history
        ], dim=-1)
        
        scores = self.mlp(att_input).squeeze(-1) # (B, Seq)
        
        # Masquage (-inf sur le padding)
        scores = scores.masked_fill(mask, -1e9)
        
        weights = F.softmax(scores, dim=1) # (B, Seq)
        
        # Somme pondérée
        output = torch.sum(weights.unsqueeze(-1) * history, dim=1) # (B, Emb)
        return output

# ==========================================
# 2. Modèle FiBiNET Pro
# ==========================================

class MM_FiBiNET_Pro(nn.Module):
    def __init__(self, feature_map, model_cfg):
        super(MM_FiBiNET_Pro, self).__init__()
        
        # HYPERPARAMETRE CLE : Augmenter la dim si possible
        self.emb_dim = model_cfg.get("embedding_dim", 128) 
        mm_input_dim = 128
        
        # Embeddings
        self.item_emb = nn.Embedding(91718, self.emb_dim, padding_idx=0)
        self.cate_emb = nn.Embedding(11, self.emb_dim) # Likes/Views
        
        # Projection Multimodale avec activation forte
        self.mm_proj = nn.Sequential(
            nn.Linear(mm_input_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            DICE(self.emb_dim, dim=2)
        )
        
        # Attention Pooling pour l'historique (Amélioration vs Mean Pooling)
        self.hist_attention = AttentionPooling(self.emb_dim)
        
        # SENet
        # Champs: [Like, View, ID, Image, Hist_Attn] -> 5 Champs
        self.num_fields = 5 
        self.senet = SENetLayer(self.num_fields, reduction_ratio=2)
        
        # Bilinear
        self.bilinear = BilinearInteraction(self.emb_dim, self.num_fields, bilinear_type="all")
        
        # MLP Final (Plus profond + DICE)
        num_pairs = (self.num_fields * (self.num_fields - 1)) // 2
        total_input_dim = (self.num_fields + num_pairs) * self.emb_dim
        
        # Architecture profonde [1024, 512, 256] pour capturer la complexité
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, 1024),
            nn.BatchNorm1d(1024),
            DICE(1024),
            nn.Dropout(0.3), # Dropout plus fort car modèle plus gros
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            DICE(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            DICE(256),
            nn.Dropout(0.2),
            
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
        
        # 2. Features (Fields)
        like_feat = self.cate_emb(likes)
        view_feat = self.cate_emb(views)
        item_id_feat = self.item_emb(item_id)
        item_img_feat = self.mm_proj(item_mm)
        
        # Fusion Item pour l'attention (ID + Image)
        target_combined = item_id_feat + item_img_feat

        # 3. Traitement Historique Avancé (Attention)
        if hist_ids is not None:
            seq_emb = self.item_emb(hist_ids)
            mask = (hist_ids == 0)
            # L'attention regarde "Target Combined" vs "History ID"
            hist_feat = self.hist_attention(target_combined, seq_emb, mask)
        else:
            hist_feat = torch.zeros_like(item_id_feat)

        # 4. Empilement des champs
        # Note: On a retiré User (souvent bruyant) pour se concentrer sur Item/Context
        sparse_inputs = torch.stack([
            like_feat, view_feat, item_id_feat, item_img_feat, hist_feat
        ], dim=1)
        
        # 5. SENet & Bilinear
        senet_output = self.senet(sparse_inputs)
        bilinear_output = self.bilinear(senet_output)
        
        # 6. Concat & MLP
        c_input = torch.cat([
            senet_output.view(item_id.size(0), -1),
            bilinear_output.view(item_id.size(0), -1)
        ], dim=1)
        
        logits = self.mlp(c_input)
        return self.sigmoid(logits).squeeze(-1)

def build_model(feature_map, model_cfg):
    return MM_FiBiNET_Pro(feature_map, model_cfg)