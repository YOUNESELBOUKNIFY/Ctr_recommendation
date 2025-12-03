import torch
import torch.nn as nn
import torch.nn.functional as F

class DICE(nn.Module):
    """Activation Function avancée souvent utilisée dans les modèles CTR (Data Adaptive Activation)"""
    def __init__(self, emb_size=1, dim=2, epsilon=1e-8):
        super(DICE, self).__init__()
        assert dim == 2 or dim == 3
        self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim

    def forward(self, x):
        # Normalisation adaptée à la dimension (Batch ou Sequence)
        if self.dim == 2:
            x_p = self.bn(x)
        else:
            x_p = self.bn(x.transpose(1, 2)).transpose(1, 2)
        gate = self.sigmoid(x_p)
        return gate * x + (1 - gate) * 0.0 * x # Leaky part controlée

class LocalActivationUnit(nn.Module):
    """
    C'est le cœur de DIN.
    Calcule le poids d'attention entre l'Item Cible (Query) et un Item de l'historique (Key).
    """
    def __init__(self, hidden_size=[80, 40], embedding_dim=64):
        super(LocalActivationUnit, self).__init__()
        # Entrée: Query + Key + (Query-Key) + (Query*Key) -> Très puissant pour capturer la similarité
        self.dnn = nn.Sequential(
            nn.Linear(4 * embedding_dim, hidden_size[0]),
            nn.PReLU(), # Ou Dice
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.PReLU(),
            nn.Linear(hidden_size[1], 1)
        )

    def forward(self, query, user_behavior):
        # query: (Batch, 1, Emb_Dim) - L'item cible
        # user_behavior: (Batch, Seq_Len, Emb_Dim) - L'historique
        
        seq_len = user_behavior.size(1)
        # On répète la query pour chaque élément de la séquence
        queries = query.expand(-1, seq_len, -1)
        
        # Interactions explicites
        attention_input = torch.cat([
            queries, 
            user_behavior, 
            queries - user_behavior, 
            queries * user_behavior
        ], dim=-1) # (Batch, Seq_Len, 4*Emb)
        
        attention_score = self.dnn(attention_input) # (Batch, Seq_Len, 1)
        return attention_score

class MMDIN(nn.Module):
    def __init__(self, feature_map, model_cfg):
        super(MMDIN, self).__init__()
        
        self.emb_dim = model_cfg.get("embedding_dim", 64)
        mm_input_dim = 128 # Dimension fixée par le dataset (BERT/CLIP)
        
        # 1. Embeddings pour les IDs (Sparse)
        # On suppose les tailles de vocabulaire (à ajuster si différent)
        self.user_emb = nn.Embedding(20000, self.emb_dim) # user_id (approx)
        self.item_id_emb = nn.Embedding(91718, self.emb_dim, padding_idx=0) # item_id
        
        # Embeddings contextuels
        self.likes_emb = nn.Embedding(11, 16)
        self.views_emb = nn.Embedding(11, 16)
        
        # 2. Projection Multimodale
        # On projette le vecteur 128d (image) vers la même taille que l'ID (64d)
        # pour pouvoir les additionner ou concaténer proprement.
        self.mm_projector = nn.Sequential(
            nn.Linear(mm_input_dim, self.emb_dim),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU()
        )

        # 3. Attention Layer (DIN Core)
        # Input size = Emb_dim (ID) + Emb_dim (MM proj) = 2 * Emb_dim
        self.attention = LocalActivationUnit(hidden_size=[80, 40], embedding_dim=self.emb_dim)
        
        # 4. Fully Connected Layers (MLP final)
        # Entrée du MLP : 
        # User (Emb) + Context (16+16) + Target Item (Emb) + Target MM (Emb) + History Attention (Emb)
        input_size = self.emb_dim + 32 + self.emb_dim + self.emb_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(model_cfg.get("net_dropout", 0.1)),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(model_cfg.get("net_dropout", 0.1)),
            nn.Linear(128, 1)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch_dict):
        # --- A. Récupération des Inputs ---
        item_id = batch_dict['item_id'].long()         # (B,)
        item_mm = batch_dict['item_emb_d128'].float()  # (B, 128)
        
        # Context
        likes = batch_dict['likes_level'].long()
        views = batch_dict['views_level'].long()
        
        # Historique (si disponible dans batch_dict, sinon on ignore ou on gère)
        # Note: Votre dataloader actuel doit renvoyer 'item_seq' pour que DIN brille.
        # Si item_seq n'est pas dans le batch (car désactivé), ce modèle perd de son intérêt.
        # On suppose ici que vous avez item_seq.
        hist_ids = batch_dict.get('item_seq', None) # (B, Seq_Len)
        
        # --- B. Traitement de l'Item Cible (Target) ---
        target_id_emb = self.item_id_emb(item_id)          # (B, Emb)
        target_mm_emb = self.mm_projector(item_mm)         # (B, Emb)
        
        # FUSION TARDIVE: On combine ID + Image pour représenter l'item
        target_combined = target_id_emb + target_mm_emb    # (B, Emb)
        
        # --- C. Traitement de l'Historique (Attention) ---
        if hist_ids is not None:
            # Mask pour ignorer le padding (0)
            mask = (hist_ids > 0).unsqueeze(-1) # (B, Seq, 1)
            
            # Embeddings de l'historique
            # NOTE: Idéalement il faudrait aussi les embeddings MM de l'historique.
            # Pour simplifier ici (car lourd à charger), on utilise l'ID embedding de l'historique
            # Si vous avez les embeddings MM de la séquence, ajoutez-les ici.
            hist_emb = self.item_id_emb(hist_ids) # (B, Seq, Emb)
            
            # Calcul de l'attention
            # "À quel point l'historique ressemble à ma cible ?"
            att_scores = self.attention(target_combined.unsqueeze(1), hist_emb) # (B, Seq, 1)
            
            # On applique le mask (force scores très bas sur padding)
            paddings = torch.ones_like(att_scores) * (-1e9)
            att_scores = torch.where(mask, att_scores, paddings)
            
            # Softmax pour avoir des probabilités
            att_weights = F.softmax(att_scores, dim=1)
            
            # Somme pondérée
            user_interest = torch.sum(att_weights * hist_emb, dim=1) # (B, Emb)
        else:
            # Fallback si pas d'historique
            user_interest = torch.zeros_like(target_combined)

        # --- D. Feature utilisateurs & Contexte ---
        # Si vous avez un user_id stable, activez la ligne suivante, sinon random init
        # user_feat = self.user_emb(batch_dict['user_id'].long())
        user_feat = torch.zeros((item_id.size(0), self.emb_dim), device=item_id.device) # Placeholder

        ctx_feat = torch.cat([self.likes_emb(likes), self.views_emb(views)], dim=1) # (B, 32)
        
        # --- E. Concaténation Finale ---
        # On met tout ensemble : Qui est l'user ? + Contexte + Quel est l'item ? + Qu'est-ce qui l'intéresse ?
        dnn_input = torch.cat([user_feat, ctx_feat, target_combined, user_interest], dim=1)
        
        # --- F. Prédiction ---
        logit = self.mlp(dnn_input)
        return self.sigmoid(logit).squeeze(-1)

def build_model(feature_map, model_cfg):
    return MMDIN(feature_map, model_cfg)


### 2. Adaptation nécessaire du DataLoader
#Ce modèle **nécessite** la colonne `item_seq` (l'historique de l'utilisateur).
#Dans votre fichier `src/dataloader.py` que nous avons corrigé plus tôt, j'avais mis ceci :
