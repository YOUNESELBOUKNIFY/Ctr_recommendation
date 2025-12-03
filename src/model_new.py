import torch
import torch.nn as nn
import torch.nn.functional as F

class DICE(nn.Module):
    """
    Data Adaptive Activation Function (Version Finale Optimisée).
    
    Cette activation s'adapte dynamiquement à la distribution des données d'entrée.
    Le paramètre 'alpha' est apprenable (nn.Parameter), permettant au modèle de décider 
    de la pente optimale pour la partie négative (leaky) de chaque neurone.
    """
    def __init__(self, emb_size, dim=2, epsilon=1e-8):
        super(DICE, self).__init__()
        assert dim == 2 or dim == 3
        
        # Batch Normalization pour standardiser l'input
        self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim
        
        # PARAMÈTRE APPRENABLE : Alpha (pente de la partie rectifiée)
        # Initialisé à 0.0 (comme ReLU), il évoluera pendant l'entraînement par backprop.
        self.alpha = nn.Parameter(torch.zeros((emb_size,)))

    def forward(self, x):
        # 1. Normalisation (x_p)
        # On transpose si l'entrée est en 3D (Batch, Seq, Emb) car BatchNorm1d attend (N, C, L)
        if self.dim == 2:
            x_p = self.bn(x)
        else:
            x_p = self.bn(x.transpose(1, 2)).transpose(1, 2)
        
        # 2. Calcul de la probabilité d'activation (gate)
        gate = self.sigmoid(x_p)
        
        # 3. Gestion du broadcasting pour alpha
        if self.dim == 2:
            alpha = self.alpha.unsqueeze(0)       # (1, Emb) pour broadcaster sur (Batch, Emb)
        else:
            alpha = self.alpha.view(1, 1, -1)     # (1, 1, Emb) pour broadcaster sur (Batch, Seq, Emb)
            
        # 4. Formule DICE : p(s) * s + (1 - p(s)) * alpha * s
        return gate * x + (1 - gate) * alpha * x

class LocalActivationUnit(nn.Module):
    """
    Unité d'Attention Locale (Cœur de DIN).
    Détermine l'importance de chaque item de l'historique par rapport à l'item cible.
    """
    def __init__(self, hidden_size=[80, 40], embedding_dim=64):
        super(LocalActivationUnit, self).__init__()
        
        # Entrée: Query + Key + (Query-Key) + (Query*Key)
        input_dim = 4 * embedding_dim
        
        # Réseau dense avec activation DICE pour capturer les relations non-linéaires complexes
        self.dnn = nn.Sequential(
            nn.Linear(input_dim, hidden_size[0]),
            DICE(hidden_size[0], dim=2),
            nn.Linear(hidden_size[0], hidden_size[1]),
            DICE(hidden_size[1], dim=2),
            nn.Linear(hidden_size[1], 1)
        )

    def forward(self, query, user_behavior):
        # query: (Batch, 1, Emb_Dim) -> Item cible
        # user_behavior: (Batch, Seq_Len, Emb_Dim) -> Historique
        
        seq_len = user_behavior.size(1)
        # Répétition de la query pour correspondre à la séquence
        queries = query.expand(-1, seq_len, -1)
        
        # Interactions explicites
        attention_input = torch.cat([
            queries, 
            user_behavior, 
            queries - user_behavior, 
            queries * user_behavior
        ], dim=-1) # (Batch, Seq, 4*Emb)
        
        # Aplatir pour BatchNorm1d dans DICE (Batch * Seq, Features)
        batch_size = attention_input.size(0)
        attention_input = attention_input.view(-1, attention_input.size(-1)) 
        
        attention_score = self.dnn(attention_input) 
        
        # Retour au format (Batch, Seq, 1)
        return attention_score.view(batch_size, seq_len, 1)

class MMDIN(nn.Module):
    """
    Multi-Modal Deep Interest Network (MMDIN).
    Architecture finale combinant embeddings ID et Multimodaux avec Attention DIN.
    """
    def __init__(self, feature_map, model_cfg):
        super(MMDIN, self).__init__()
        
        self.emb_dim = model_cfg.get("embedding_dim", 64)
        mm_input_dim = 128 # Provenant de BERT/CLIP (dataset MicroLens)
        
        # --- 1. Embeddings Identifiants (Sparse) ---
        # Tailles vocabulaire à ajuster selon le dataset réel si besoin
        self.user_emb = nn.Embedding(20000, self.emb_dim)
        self.item_id_emb = nn.Embedding(91718, self.emb_dim, padding_idx=0)
        
        # Embeddings de contexte (Catégoriels)
        self.likes_emb = nn.Embedding(11, 16)
        self.views_emb = nn.Embedding(11, 16)
        
        # --- 2. Projection Multimodale ---
        # Transforme le vecteur image 128d pour le rendre compatible avec l'espace ID
        # Utilise DICE pour ne pas perdre d'info visuelle par coupure (ReLU)
        self.mm_projector = nn.Sequential(
            nn.Linear(mm_input_dim, self.emb_dim),
            DICE(self.emb_dim, dim=2) 
        )

        # --- 3. Couche d'Attention ---
        self.attention = LocalActivationUnit(hidden_size=[80, 40], embedding_dim=self.emb_dim)
        
        # --- 4. MLP Final ---
        # Taille entrée : User(Placeholder) + Context(32) + Target(Emb) + HistoryInterest(Emb)
        input_size = self.emb_dim + 32 + self.emb_dim + self.emb_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 256),
            DICE(256, dim=2),
            nn.Dropout(model_cfg.get("net_dropout", 0.1)),
            nn.Linear(256, 128),
            DICE(128, dim=2),
            nn.Dropout(model_cfg.get("net_dropout", 0.1)),
            nn.Linear(128, 1)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch_dict):
        # A. Récupération des données
        item_id = batch_dict['item_id'].long()
        item_mm = batch_dict['item_emb_d128'].float()
        likes = batch_dict['likes_level'].long()
        views = batch_dict['views_level'].long()
        hist_ids = batch_dict.get('item_seq', None)
        
        # B. Représentation de l'Item Cible (Target)
        target_id_emb = self.item_id_emb(item_id)          # ID Embedding
        target_mm_emb = self.mm_projector(item_mm)         # Visual Embedding projeté
        
        # Fusion : L'item est défini par son ID + son Visuel
        target_combined = target_id_emb + target_mm_emb
        
        # C. Attention sur l'Historique (Deep Interest)
        if hist_ids is not None:
            mask = (hist_ids > 0).unsqueeze(-1) # Masque pour le padding
            hist_emb = self.item_id_emb(hist_ids)
            
            # Calcul des poids d'attention
            # "Quelle partie de l'historique ressemble à l'item cible ?"
            att_scores = self.attention(target_combined.unsqueeze(1), hist_emb)
            
            # Application du masque (score très bas pour le padding)
            paddings = torch.ones_like(att_scores) * (-1e9)
            att_scores = torch.where(mask, att_scores, paddings)
            
            # Somme pondérée de l'historique
            att_weights = F.softmax(att_scores, dim=1)
            user_interest = torch.sum(att_weights * hist_emb, dim=1)
        else:
            # Fallback si pas d'historique (Cold start)
            user_interest = torch.zeros_like(target_combined)

        # D. Concaténation finale
        # Placeholder pour User Embedding (à 0 si pas d'user_id stable ou nouveau user)
        user_feat = torch.zeros((item_id.size(0), self.emb_dim), device=item_id.device)

        ctx_feat = torch.cat([self.likes_emb(likes), self.views_emb(views)], dim=1)
        
        dnn_input = torch.cat([user_feat, ctx_feat, target_combined, user_interest], dim=1)
        
        # E. Prédiction finale
        logit = self.mlp(dnn_input)
        return self.sigmoid(logit).squeeze(-1)

def build_model(feature_map, model_cfg):
    return MMDIN(feature_map, model_cfg)