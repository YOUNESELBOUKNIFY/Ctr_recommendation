import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Composants Avancés (Transformer & CrossNet)
# ==========================================

class Dice(nn.Module):
    """Activation adaptative (toujours utile)"""
    def __init__(self, input_dim, epsilon=1e-8):
        super(Dice, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim, eps=epsilon)
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.zeros((input_dim,)))

    def forward(self, x):
        if x.dim() == 3:
            x_p = self.bn(x.transpose(1, 2)).transpose(1, 2)
        else:
            x_p = self.bn(x)
        gate = self.sigmoid(x_p)
        return gate * x + (1 - gate) * self.alpha * x

class CrossNetV2(nn.Module):
    """
    Deep Cross Network V2 (Matrix Version).
    Capture les interactions d'ordre élevé de manière explicite.
    Beaucoup plus puissant qu'un simple MLP pour les features catégorielles.
    """
    def __init__(self, input_dim, num_layers=2):
        super(CrossNetV2, self).__init__()
        self.num_layers = num_layers
        self.kernels = nn.ParameterList(
            [nn.Parameter(torch.randn(input_dim, input_dim)) for _ in range(num_layers)]
        )
        self.bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)]
        )
        # Init
        for weight in self.kernels:
            nn.init.xavier_normal_(weight)

    def forward(self, x):
        x_0 = x
        for i in range(self.num_layers):
            # Formule DCNv2: x_{l+1} = x_0 * (W_l * x_l + b_l) + x_l
            # On utilise matmul pour W * x
            xl_w = torch.matmul(x, self.kernels[i]) + self.bias[i]
            x = x_0 * xl_w + x
        return x

class TransformerLayer(nn.Module):
    """
    Encoder Layer standard de Transformer pour modéliser la séquence historique.
    """
    def __init__(self, d_model, nhead=2, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed Forward
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU() # On reste simple dans le Transformer

    def forward(self, src, src_key_padding_mask=None):
        # src: (Batch, Seq, Emb)
        # mask: (Batch, Seq) True si padding
        
        # 1. Self Attention
        # Note: MultiheadAttention attend key_padding_mask où True = ignoré
        src2, _ = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # 2. Feed Forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# ==========================================
# 2. Modèle Principal : MM-BST-DCN
# ==========================================

class MM_BST(nn.Module):
    def __init__(self, feature_map, model_cfg):
        super(MM_BST, self).__init__()
        
        self.emb_dim = model_cfg.get("embedding_dim", 64)
        mm_input_dim = 128
        max_len = model_cfg.get("max_len", 20)
        
        # --- Embeddings ---
        self.item_emb = nn.Embedding(91718, self.emb_dim, padding_idx=0)
        self.cate_emb = nn.Embedding(11, 16) # Likes/Views
        
        # Positional Encoding pour le Transformer (Sequence)
        self.position_emb = nn.Parameter(torch.randn(max_len, self.emb_dim))
        
        # Projection Multimodale
        self.mm_proj = nn.Sequential(
            nn.Linear(mm_input_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim), # LayerNorm souvent mieux pour Transformer
            nn.GELU()
        )
        
        # --- Transformer (BST Core) ---
        # Remplace l'attention simple de DIN
        self.transformer = TransformerLayer(d_model=self.emb_dim, nhead=2, dropout=0.1)
        
        # Attention finale (Target vs Transformed Sequence)
        # On utilise une attention simple pour agréger la sortie du Transformer par rapport à la cible
        self.attention_query_proj = nn.Linear(self.emb_dim, self.emb_dim)
        
        # --- Deep Cross Network ---
        # Feature sizes: User(Placeholder 0) + Context(32) + Target(Emb) + Seq(Emb)
        total_input_dim = self.emb_dim + 32 + self.emb_dim + self.emb_dim
        self.cross_net = CrossNetV2(total_input_dim, num_layers=2)
        
        # --- MLP Final ---
        hidden_units = [256, 128, 64]
        layers = []
        input_dim = total_input_dim # On concatène la sortie du CrossNet (Residual) ou juste l'input modifié
        
        for unit in hidden_units:
            layers.append(nn.Linear(input_dim, unit))
            layers.append(Dice(unit))
            layers.append(nn.Dropout(0.2))
            input_dim = unit
            
        self.mlp = nn.Sequential(*layers)
        self.final_linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch_dict):
        # 1. Inputs
        item_id = batch_dict['item_id'].long()
        item_mm = batch_dict['item_emb_d128'].float()
        hist_ids = batch_dict.get('item_seq', None)
        likes = batch_dict['likes_level'].long()
        views = batch_dict['views_level'].long()
        
        # 2. Embeddings Target (Fusion MM)
        target_id = self.item_emb(item_id)
        target_mm = self.mm_proj(item_mm)
        target_emb = target_id + target_mm
        
        # 3. Traitement Séquence (Transformer)
        if hist_ids is not None:
            # (Batch, Seq, Emb)
            seq_emb = self.item_emb(hist_ids)
            
            # Ajout Positional Encoding
            # On s'assure que la dimension correspond (slicing si batch sequence < max_len)
            seq_len = seq_emb.size(1)
            positions = self.position_emb[:seq_len, :].unsqueeze(0) # (1, Seq, Emb)
            seq_emb = seq_emb + positions
            
            # Masque de padding pour Transformer (True là où c'est du padding)
            # hist_ids == 0 -> True
            padding_mask = (hist_ids == 0)
            
            # Passage Transformer
            seq_transformed = self.transformer(seq_emb, src_key_padding_mask=padding_mask)
            
            # 4. Attention Pooling (Target-Aware)
            # Au lieu de prendre juste le dernier item, on regarde quels items transformés matchent la cible
            # Query = Target, Keys/Values = Transformed Sequence
            
            # Simple Dot-Product Attention
            # (B, 1, Emb)
            query = self.attention_query_proj(target_emb).unsqueeze(2) 
            # (B, Seq, Emb) x (B, Emb, 1) -> (B, Seq, 1) scores
            att_scores = torch.bmm(seq_transformed, query)
            
            # Masking (mettre -inf sur le padding)
            att_scores = att_scores.masked_fill(padding_mask.unsqueeze(-1), -1e9)
            
            att_weights = F.softmax(att_scores, dim=1)
            # (B, Seq, Emb) * (B, Seq, 1) -> Sum -> (B, Emb)
            user_history_rep = torch.sum(seq_transformed * att_weights, dim=1)
            
        else:
            user_history_rep = torch.zeros_like(target_emb)

        # 5. Features Contexte
        ctx_feat = torch.cat([self.cate_emb(likes), self.cate_emb(views)], dim=1)
        # Placeholder User (optimisation possible: ajouter Embedding User)
        user_feat = torch.zeros((item_id.size(0), self.emb_dim), device=item_id.device)
        
        # 6. Concaténation Globale
        # [Placeholder User, Context, Target, History]
        all_features = torch.cat([user_feat, ctx_feat, target_emb, user_history_rep], dim=-1)
        
        # 7. Deep Cross Network (Interactions explicites)
        cross_out = self.cross_net(all_features)
        
        # 8. MLP & Prediction
        # On peut sommer la sortie du CrossNet avec l'input original ou juste passer au MLP
        # Ici on passe la sortie enrichie par DCN au MLP
        dnn_out = self.mlp(cross_out)
        logits = self.final_linear(dnn_out)
        
        return self.sigmoid(logits).squeeze(-1)

def build_model(feature_map, model_cfg):
    return MM_BST(feature_map, model_cfg)