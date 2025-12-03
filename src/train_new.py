import torch
import torch.nn as nn
import numpy as np

# ==========================================
# 1. Composants internes (Style FuxiCTR)
# ==========================================

class Dice(nn.Module):
    """Activation Dice officielle de FuxiCTR / DIN Paper"""
    def __init__(self, input_dim, epsilon=1e-8):
        super(Dice, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim, eps=epsilon)
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.zeros((input_dim,)))

    def forward(self, x):
        # Support pour 2D ou 3D inputs
        if x.dim() == 3:
            x_p = self.bn(x.transpose(1, 2)).transpose(1, 2)
        else:
            x_p = self.bn(x)
        gate = self.sigmoid(x_p)
        return gate * x + (1 - gate) * self.alpha * x

class MLP_Block(nn.Module):
    """Bloc MLP standard"""
    def __init__(self, input_dim, hidden_units, activation='ReLU', dropout=0.0):
        super(MLP_Block, self).__init__()
        layers = []
        prev_dim = input_dim
        for unit in hidden_units:
            layers.append(nn.Linear(prev_dim, unit))
            layers.append(nn.BatchNorm1d(unit))
            
            if activation == 'Dice':
                layers.append(Dice(unit))
            elif activation == 'ReLU':
                layers.append(nn.ReLU())
            elif activation == 'PReLU':
                layers.append(nn.PReLU())
                
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = unit
        
        self.mlp = nn.Sequential(*layers)
        self.out_dim = prev_dim

    def forward(self, x):
        return self.mlp(x)

class DIN_Attention(nn.Module):
    """Couche d'Attention DIN"""
    def __init__(self, embedding_dim, attention_units=[64, 32], activation='Dice'):
        super(DIN_Attention, self).__init__()
        # Input de l'attention : Query + Key + Query-Key + Query*Key
        input_dim = 4 * embedding_dim
        self.mlp = MLP_Block(input_dim, attention_units, activation=activation)
        self.fc = nn.Linear(attention_units[-1], 1)

    def forward(self, query, facts, mask):
        # query: (B, 1, Emb)
        # facts: (B, Seq, Emb)
        # mask: (B, Seq, 1)
        
        B, Seq, Emb = facts.size()
        queries = query.expand(-1, Seq, -1)
        
        # Interactions
        attention_input = torch.cat([
            queries, 
            facts, 
            queries - facts, 
            queries * facts
        ], dim=-1) # (B, Seq, 4*Emb)
        
        # On passe dans le MLP (on aplatit pour le Batchnorm)
        attention_input = attention_input.view(-1, 4 * Emb)
        attn_out = self.mlp(attention_input)
        scores = self.fc(attn_out).view(B, Seq, 1) # (B, Seq, 1)
        
        # Masquage
        paddings = torch.ones_like(scores) * (-1e9)
        scores = torch.where(mask.bool(), scores, paddings)
        
        # Softmax & Pooling
        weights = torch.softmax(scores, dim=1)
        output = torch.sum(weights * facts, dim=1) # (B, Emb)
        return output

# ==========================================
# 2. Modèle DIN (Structure FuxiCTR adaptée)
# ==========================================

class DIN(nn.Module):
    def __init__(self, feature_map, model_cfg):
        super(DIN, self).__init__()
        
        self.emb_dim = model_cfg.get("embedding_dim", 64)
        mm_input_dim = 128
        
        # --- Embeddings ---
        # IDs
        self.item_emb = nn.Embedding(91718, self.emb_dim, padding_idx=0)
        self.user_emb = nn.Embedding(20000, self.emb_dim) # Placeholder
        self.cate_emb = nn.Embedding(11, 16) # Likes/Views levels
        
        # Projection Multimodale (Image -> Emb space)
        self.mm_proj = nn.Sequential(
            nn.Linear(mm_input_dim, self.emb_dim),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU()
        )
        
        # --- DIN Core ---
        self.attention = DIN_Attention(
            self.emb_dim, 
            attention_units=[64, 32], 
            activation='Dice'
        )
        
        # --- MLP Final ---
        # Entrée = User + Contexte + Target(Emb) + Attention(Emb)
        dnn_input_dim = self.emb_dim + (16*2) + self.emb_dim + self.emb_dim
        
        self.dnn = MLP_Block(
            dnn_input_dim, 
            hidden_units=model_cfg.get("dnn_hidden_units", [512, 128, 64]),
            activation="Dice",
            dropout=model_cfg.get("net_dropout", 0.1)
        )
        self.final_linear = nn.Linear(self.dnn.out_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch_dict):
        # 1. Récupération des Inputs
        item_id = batch_dict['item_id'].long()
        item_mm = batch_dict['item_emb_d128'].float()
        hist_ids = batch_dict.get('item_seq', None)
        
        likes = batch_dict['likes_level'].long()
        views = batch_dict['views_level'].long()
        
        # 2. Embedding Item Cible (ID + Image)
        target_id_emb = self.item_emb(item_id)
        target_mm_emb = self.mm_proj(item_mm)
        target_emb = target_id_emb + target_mm_emb # Fusion
        
        # 3. Attention sur Historique
        if hist_ids is not None:
            # Mask (B, Seq, 1)
            mask = (hist_ids > 0).unsqueeze(-1)
            
            # Pour l'historique, on utilise l'embedding ID (car MM seq non dispo souvent)
            sequence_emb = self.item_emb(hist_ids)
            
            # DIN Attention Layer
            # Target (Query) vs Sequence (Facts)
            pooling_emb = self.attention(target_emb.unsqueeze(1), sequence_emb, mask)
        else:
            pooling_emb = torch.zeros_like(target_emb)

        # 4. Features Utilisateur & Contexte
        user_feat = torch.zeros((item_id.size(0), self.emb_dim), device=item_id.device)
        ctx_feat = torch.cat([self.cate_emb(likes), self.cate_emb(views)], dim=1)
        
        # 5. Concaténation
        # [User, Context, Target Item, History Interest]
        feature_emb = torch.cat([user_feat, ctx_feat, target_emb, pooling_emb], dim=-1)
        
        # 6. MLP & Prediction
        dnn_out = self.dnn(feature_emb)
        y_pred = self.final_linear(dnn_out)
        
        return self.sigmoid(y_pred).squeeze(-1)

# Wrapper pour l'appel depuis train.py
def build_model(feature_map, model_cfg):
    return DIN(feature_map, model_cfg)