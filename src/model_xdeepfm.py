import torch
import torch.nn as nn
import torch.nn.functional as F

class DICE(nn.Module):
    """Activation SOTA pour CTR."""
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

class CIN(nn.Module):
    """
    Compressed Interaction Network (CIN).
    Le composant clé de xDeepFM pour capturer les interactions vectorielles explicites.
    Calcule des interactions d'ordre croissant à chaque couche (Ordre 2, 3, 4...).
    """
    def __init__(self, input_dim, num_fields, cin_layer_units, output_dim=1):
        super(CIN, self).__init__()
        self.num_fields = num_fields
        self.cin_layer_units = cin_layer_units
        self.fc = nn.Linear(sum(cin_layer_units), output_dim)
        
        # Filtres de convolution pour les interactions
        self.conv1d_layers = nn.ModuleList()
        prev_layer_units = num_fields
        for unit in cin_layer_units:
            # W_k : tenseur de taille (H_k, H_{k-1} * m)
            # On utilise Conv1d pour implémenter efficacement la somme pondérée
            self.conv1d_layers.append(
                nn.Conv1d(in_channels=prev_layer_units * num_fields,
                          out_channels=unit,
                          kernel_size=1)
            )
            prev_layer_units = unit

    def forward(self, x):
        # x: (Batch, Num_Fields, Emb_Dim)
        batch_size, num_fields, emb_dim = x.shape
        x0 = x.unsqueeze(2) # (B, m, 1, D)
        
        hidden_layers = []
        xi = x
        
        for i, layer in enumerate(self.conv1d_layers):
            # Produit de Hadamard explicite (Interactions)
            # xi: (B, H_{k-1}, D) -> (B, 1, H_{k-1}, D)
            # x0: (B, m, 1, D)
            # interaction: (B, m, H_{k-1}, D)
            interaction = xi.unsqueeze(1) * x0
            
            # Reshape pour Conv1d : (B, m * H_{k-1}, D)
            interaction = interaction.view(batch_size, -1, emb_dim)
            
            # Application des filtres CIN (Compression)
            # out: (B, H_k, D)
            xi = layer(interaction)
            xi = F.relu(xi) # Activation
            
            # On garde le résultat de cette couche pour le pooling final
            hidden_layers.append(xi)
        
        # Sum Pooling sur la dimension d'embedding (D)
        # Chaque couche contribue à la sortie
        final_result = []
        for layer_out in hidden_layers:
            # Sum over D -> (B, H_k)
            final_result.append(torch.sum(layer_out, dim=2))
            
        # Concaténation de toutes les interactions d'ordre k
        result = torch.cat(final_result, dim=1) # (B, sum(H_k))
        
        # Projection finale
        return self.fc(result)

class MM_xDeepFM(nn.Module):
    def __init__(self, feature_map, model_cfg):
        super(MM_xDeepFM, self).__init__()
        
        self.emb_dim = model_cfg.get("embedding_dim", 128)
        mm_input_dim = 128
        
        # --- Embeddings ---
        self.item_emb = nn.Embedding(91718, self.emb_dim, padding_idx=0)
        self.cate_emb = nn.Embedding(11, self.emb_dim) # Likes/Views
        self.user_emb = nn.Parameter(torch.zeros(1, self.emb_dim)) # User moyen
        
        # Projection Multimodale
        self.mm_proj = nn.Sequential(
            nn.Linear(mm_input_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            DICE(self.emb_dim, dim=2)
        )
        
        # 6 Champs : [User, Like, View, Item_ID, Item_Image, Hist]
        self.num_fields = 6 
        
        # --- Composant 1 : Linear (1st Order) ---
        # Capture l'impact individuel de chaque feature
        self.linear = nn.Linear(self.num_fields * self.emb_dim, 1)
        
        # --- Composant 2 : CIN (Explicit High-order) ---
        # Capture : User x Image, User x Image x Hist, etc.
        # [128, 64] signifie qu'on capture des interactions d'ordre 2 et 3
        cin_layer_units = model_cfg.get("cin_layer_units", [256, 128])
        self.cin = CIN(self.emb_dim, self.num_fields, cin_layer_units)
        
        # --- Composant 3 : DNN (Implicit High-order) ---
        flatten_dim = self.num_fields * self.emb_dim
        self.dnn = nn.Sequential(
            nn.Linear(flatten_dim, 512),
            nn.BatchNorm1d(512),
            DICE(512),
            nn.Dropout(model_cfg.get("net_dropout", 0.2)),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            DICE(256),
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
            # Mean Pooling robuste
            seq_emb = seq_emb.masked_fill(mask.unsqueeze(-1), 0)
            seq_sum = torch.sum(seq_emb, dim=1)
            seq_count = torch.sum((~mask).float(), dim=1, keepdim=True).clamp(min=1)
            hist_feat = seq_sum / seq_count
        else:
            hist_feat = torch.zeros_like(item_id_feat)

        # Stack: (B, 6, D)
        sparse_inputs = torch.stack([
            user_feat, like_feat, view_feat, item_id_feat, item_img_feat, hist_feat
        ], dim=1)
        
        # Flatten pour Linear et DNN
        flat_inputs = sparse_inputs.view(batch_size, -1)
        
        # 3. Calcul des 3 parties
        linear_out = self.linear(flat_inputs) # Ordre 1
        cin_out = self.cin(sparse_inputs)     # Ordre k (Interactions Explicites)
        dnn_out = self.dnn(flat_inputs)       # Ordre infini (Interactions Implicites)
        
        # 4. Fusion (Somme)
        total_logit = linear_out + cin_out + dnn_out
        
        return self.sigmoid(total_logit).squeeze(-1)

def build_model(feature_map, model_cfg):
    return MM_xDeepFM(feature_map, model_cfg)