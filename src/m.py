import torch
import torch.nn as nn
import torch.nn.functional as F

class SENetLayer(nn.Module):
    """
    Squeeze-and-Excitation Network (SENet).
    Apprend dynamiquement l'importance de chaque champ (Field) de feature.
    """
    def __init__(self, num_fields, reduction_ratio=3):
        super(SENetLayer, self).__init__()
        # Reduction ratio pour compresser l'information
        reduced_size = max(1, num_fields // reduction_ratio)
        
        # Squeeze: (Batch, Num_Fields, Emb) -> (Batch, Num_Fields) via Mean pooling
        # Excitation: MLP pour apprendre les poids
        self.excitation = nn.Sequential(
            nn.Linear(num_fields, reduced_size),
            nn.ReLU(),
            nn.Linear(reduced_size, num_fields),
            nn.Sigmoid() # Poids entre 0 et 1
        )

    def forward(self, x):
        # x: (Batch, Num_Fields, Emb_Dim)
        
        # 1. Squeeze (Global Average Pooling sur la dim embedding)
        z = torch.mean(x, dim=-1) # (Batch, Num_Fields)
        
        # 2. Excitation (Calcul des poids)
        weights = self.excitation(z) # (Batch, Num_Fields)
        
        # 3. Reweighting (Application des poids)
        # On unsqueeze pour multiplier (Batch, Fields, 1) * (Batch, Fields, Emb)
        return x * weights.unsqueeze(-1)

class BilinearInteraction(nn.Module):
    """
    Capture les interactions de second ordre (produit de Hadamard).
    p = v_i * W * v_j
    """
    def __init__(self, input_dim, num_fields, bilinear_type="all"):
        super(BilinearInteraction, self).__init__()
        self.bilinear_type = bilinear_type
        
        if bilinear_type == "all":
            # Matrice W partagée pour toutes les paires
            self.W = nn.Parameter(torch.Tensor(input_dim, input_dim))
            nn.init.xavier_normal_(self.W)
        elif bilinear_type == "each":
            # Une matrice W par champ
            self.W_list = nn.ParameterList([
                nn.Parameter(torch.Tensor(input_dim, input_dim)) for _ in range(num_fields - 1)
            ])
            for w in self.W_list:
                nn.init.xavier_normal_(w)
        else:
            raise ValueError("bilinear_type must be 'all' or 'each'")

    def forward(self, x):
        # x: (Batch, Num_Fields, Emb)
        # Sortie attendue: Interactions combinées
        
        batch_size, num_fields, emb_dim = x.shape
        inputs = torch.split(x, 1, dim=1) # Liste de (B, 1, Emb)
        
        p = []
        
        if self.bilinear_type == "all":
            # Astuce calculatoire: (V . W)
            # On projette tout d'un coup
            vid = torch.matmul(x, self.W) # (B, F, E)
            
            # Combinaisons
            for i in range(num_fields):
                for j in range(i + 1, num_fields):
                    # Element-wise product: v_i * (W . v_j)
                    # Note: Ici on fait une version simplifiée v_i * vid_j
                    p.append(inputs[i].squeeze(1) * vid[:, j, :])
                    
        elif self.bilinear_type == "each":
             for i in range(num_fields):
                for j in range(i + 1, num_fields):
                    # Utilise une matrice spécifique W_i pour transformer v_i
                    vid = torch.matmul(inputs[i].squeeze(1), self.W_list[i])
                    p.append(vid * inputs[j].squeeze(1))

        # Stack interactions
        return torch.stack(p, dim=1) # (B, Num_Pairs, Emb)

class MM_FiBiNET(nn.Module):
    def __init__(self, feature_map, model_cfg):
        super(MM_FiBiNET, self).__init__()
        
        self.emb_dim = model_cfg.get("embedding_dim", 64)
        mm_input_dim = 128
        
        # --- Embeddings ---
        # IMPORTANT: Tous les embeddings doivent avoir la même dimension (self.emb_dim)
        self.item_emb = nn.Embedding(91718, self.emb_dim, padding_idx=0)
        self.user_emb = nn.Embedding(20000, self.emb_dim) # Placeholder User
        self.cate_emb = nn.Embedding(11, self.emb_dim)    # Likes/Views
        
        # Projection Multimodale
        self.mm_proj = nn.Sequential(
            nn.Linear(mm_input_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.ReLU()
        )
        
        # --- SENet ---
        # Champs : [User, Like, View, Item_ID, Item_Image, History_Pooled]
        self.num_fields = 6 
        self.senet = SENetLayer(self.num_fields, reduction_ratio=2)
        
        # --- Bilinear ---
        # Interaction Type: "all" (Field-All) est plus léger et stable que "each"
        self.bilinear = BilinearInteraction(self.emb_dim, self.num_fields, bilinear_type="all")
        
        # --- MLP Final ---
        # Input Size = (Original Fields + Bilinear Pairs) * Emb_Dim
        num_pairs = (self.num_fields * (self.num_fields - 1)) // 2
        total_input_dim = (self.num_fields + num_pairs) * self.emb_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
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
        
        batch_size = item_id.size(0)
        
        # 2. Préparation des Features (Fields)
        # Chaque feature doit être (Batch, 1, Emb_Dim)
        
        # F1: User (Zero placeholder ou appris)
        user_feat = torch.zeros((batch_size, self.emb_dim), device=item_id.device) 
        
        # F2 & F3: Contexte (Likes, Views)
        like_feat = self.cate_emb(likes)
        view_feat = self.cate_emb(views)
        
        # F4: Item ID
        item_id_feat = self.item_emb(item_id)
        
        # F5: Item Image
        item_img_feat = self.mm_proj(item_mm)
        
        # F6: Historique (Pooled)
        if hist_ids is not None:
            # (B, Seq, Emb)
            seq_emb = self.item_emb(hist_ids)
            mask = (hist_ids == 0)
            
            # Mean Pooling ignorant le padding
            seq_emb_masked = seq_emb * (~mask.unsqueeze(-1)).float()
            seq_sum = torch.sum(seq_emb_masked, dim=1)
            seq_count = torch.sum((~mask).float(), dim=1, keepdim=True).clamp(min=1)
            hist_feat = seq_sum / seq_count
        else:
            hist_feat = torch.zeros_like(item_id_feat)

        # Empilement des champs : (Batch, Num_Fields, Emb_Dim)
        # Ordre: [User, Like, View, ID, Image, Hist]
        sparse_inputs = torch.stack([
            user_feat, like_feat, view_feat, item_id_feat, item_img_feat, hist_feat
        ], dim=1)
        
        # 3. SENet (Feature Importance)
        senet_output = self.senet(sparse_inputs)
        
        # 4. Bilinear Interaction (Combinaisons)
        bilinear_output = self.bilinear(senet_output)
        
        # 5. Concatenation
        c_input = torch.cat([
            senet_output.view(batch_size, -1),   # Les features originales pondérées
            bilinear_output.view(batch_size, -1) # Les interactions croisées
        ], dim=1)
        
        # 6. MLP Final
        logits = self.mlp(c_input)
        
        return self.sigmoid(logits).squeeze(-1)

def build_model(feature_map, model_cfg):
    return MM_FiBiNET(feature_map, model_cfg)