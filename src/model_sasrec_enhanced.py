import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossNetV2(nn.Module):
    """
    Deep Cross Network V2.
    Mélange l'intention séquentielle avec le contexte statique.
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
            xl_w = torch.matmul(x, self.kernels[i]) + self.bias[i]
            x = x_0 * xl_w + x
        return x

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(hidden_units, hidden_units * 4)
        self.linear2 = nn.Linear(hidden_units * 4, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU() # GELU > ReLU pour Transformer

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class SASRecBlock(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(SASRecBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_units)
        self.self_attention = nn.MultiheadAttention(hidden_units, num_heads, dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm2 = nn.LayerNorm(hidden_units)
        self.feed_forward = PointWiseFeedForward(hidden_units, dropout_rate)

    def forward(self, x, mask):
        # Pre-Norm Architecture (Plus stable)
        norm_x = self.layer_norm1(x)
        att_out, _ = self.self_attention(norm_x, norm_x, norm_x, key_padding_mask=mask)
        x = x + self.dropout(att_out)
        
        norm_x = self.layer_norm2(x)
        ff_out = self.feed_forward(norm_x)
        x = x + self.dropout(ff_out)
        return x

class MM_SASRec_Enhanced(nn.Module):
    def __init__(self, feature_map, model_cfg):
        super(MM_SASRec_Enhanced, self).__init__()
        
        self.emb_dim = model_cfg.get("embedding_dim", 128)
        self.max_len = model_cfg.get("max_len", 20)
        mm_input_dim = 128
        
        # --- Embeddings ---
        self.item_emb = nn.Embedding(91718, self.emb_dim, padding_idx=0)
        self.cate_emb = nn.Embedding(11, self.emb_dim)
        self.pos_emb = nn.Embedding(self.max_len, self.emb_dim)
        
        # Projection Multimodale
        self.mm_proj = nn.Sequential(
            nn.Linear(mm_input_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.GELU()
        )
        
        # --- SASRec Encoder ---
        num_blocks = model_cfg.get("sasrec_blocks", 2)
        num_heads = model_cfg.get("sasrec_heads", 4)
        dropout = model_cfg.get("net_dropout", 0.2)
        
        self.sas_blocks = nn.ModuleList([
            SASRecBlock(self.emb_dim, num_heads, dropout) for _ in range(num_blocks)
        ])
        
        # --- Target Attention (Fusion Cible - Séquence) ---
        # Pour savoir quel item de l'historique est pertinent pour la cible actuelle
        self.attn_linear = nn.Linear(self.emb_dim, self.emb_dim)
        
        # --- Cross Network (Fusion Finale) ---
        # User(Placeholder) + Ctx + Target + History
        input_dim = self.emb_dim * 4 
        self.cross_net = CrossNetV2(input_dim, num_layers=2)
        
        # --- MLP Final ---
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch_dict):
        item_id = batch_dict['item_id'].long()
        item_mm = batch_dict['item_emb_d128'].float()
        likes = batch_dict['likes_level'].long()
        views = batch_dict['views_level'].long()
        hist_ids = batch_dict.get('item_seq', None)
        
        batch_size = item_id.size(0)
        device = item_id.device
        
        # 1. Target Features (ID + Image)
        target_id = self.item_emb(item_id)
        target_mm = self.mm_proj(item_mm)
        target_combined = target_id + target_mm 
        
        # 2. Context Features
        ctx_feat = self.cate_emb(likes) + self.cate_emb(views)
        
        # 3. SASRec (Sequence Modeling)
        if hist_ids is not None:
            # (Batch, Seq)
            seq_emb = self.item_emb(hist_ids)
            
            # --- AMELIORATION : On ne peut pas facilement ajouter l'image à l'historique
            # car le dataset ne donne pas item_emb_seq. 
            # On garde donc ID only pour la séquence pour l'instant.
            
            # Positional Encoding
            positions = torch.arange(self.max_len, device=device).unsqueeze(0).expand(batch_size, -1)
            # Slicing si la sequence est plus courte
            if seq_emb.size(1) < self.max_len:
                positions = positions[:, :seq_emb.size(1)]
            
            seq_emb += self.pos_emb(positions)
            
            # Mask (True = Padding = 0)
            padding_mask = (hist_ids == 0)
            
            # Transformer Pass
            sas_feat = seq_emb
            for block in self.sas_blocks:
                sas_feat = block(sas_feat, padding_mask)
            
            # Target-Aware Attention Pooling
            # On cherche dans la sortie du SASRec ce qui ressemble à la cible
            query = self.attn_linear(target_combined).unsqueeze(2) # (B, D, 1)
            
            # (B, Seq, D) x (B, D, 1) -> (B, Seq, 1)
            att_scores = torch.bmm(sas_feat, query).squeeze(-1)
            att_scores = att_scores.masked_fill(padding_mask, -1e9)
            att_weights = F.softmax(att_scores, dim=1).unsqueeze(-1)
            
            history_repr = torch.sum(sas_feat * att_weights, dim=1) # (B, Emb)
        else:
            history_repr = torch.zeros_like(target_combined)

        # 4. Fusion Finale (DCN)
        user_feat = torch.zeros_like(target_combined) # Placeholder
        
        # Concatenation : [User, Context, Target, History]
        all_features = torch.cat([user_feat, ctx_feat, target_combined, history_repr], dim=1)
        
        # Interactions Explicites via CrossNet
        cross_out = self.cross_net(all_features)
        
        # Prédiction
        return self.sigmoid(self.mlp(cross_out)).squeeze(-1)

def build_model(feature_map, model_cfg):
    return MM_SASRec_Enhanced(feature_map, model_cfg)