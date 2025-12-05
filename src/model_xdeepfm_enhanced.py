import torch
import torch.nn as nn
import torch.nn.functional as F

class DICE(nn.Module):
    """Activation DICE (Data Adaptive) pour CTR."""
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
    Attention Cible-Historique (Style DIN).
    Au lieu de faire une moyenne de l'historique, on pondère chaque item
    selon sa ressemblance avec l'item cible.
    """
    def __init__(self, embedding_dim):
        super(AttentionPooling, self).__init__()
        # Input: Query, Key, Q-K, Q*K
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
        
        # Score d'attention (B, Seq)
        scores = self.mlp(att_input.view(-1, 4 * query.size(-1))).view(query.size(0), seq_len)
        
        # Masquage du padding
        scores = scores.masked_fill(mask, -1e9)
        weights = F.softmax(scores, dim=1)
        
        # Somme pondérée (B, Emb)
        return torch.sum(weights.unsqueeze(-1) * history, dim=1)

class CIN(nn.Module):
    """
    Compressed Interaction Network (Cœur de xDeepFM).
    Calcule les interactions explicites d'ordre élevé.
    """
    def __init__(self, input_dim, num_fields, cin_layer_units, output_dim=1):
        super(CIN, self).__init__()
        self.num_fields = num_fields
        self.cin_layer_units = cin_layer_units
        self.fc = nn.Linear(sum(cin_layer_units), output_dim)
        
        self.conv1d_layers = nn.ModuleList()
        prev_layer_units = num_fields
        for unit in cin_layer_units:
            self.conv1d_layers.append(
                nn.Conv1d(in_channels=prev_layer_units * num_fields,
                          out_channels=unit,
                          kernel_size=1)
            )
            prev_layer_units = unit

    def forward(self, x):
        # x: (B, Num_Fields, D)
        batch_size, num_fields, emb_dim = x.shape
        x0 = x.unsqueeze(2) # (B, m, 1, D)
        hidden_layers = []
        xi = x
        
        for i, layer in enumerate(self.conv1d_layers):
            interaction = xi.unsqueeze(1) * x0 # (B, m, H_{k-1}, D)
            interaction = interaction.view(batch_size, -1, emb_dim)
            xi = F.relu(layer(interaction)) # (B, H_k, D)
            hidden_layers.append(xi)
        
        # Sum Pooling sur D
        final_result = [torch.sum(layer_out, dim=2) for layer_out in hidden_layers]
        result = torch.cat(final_result, dim=1) # (B, sum(H_k))
        return self.fc(result)

class MM_xDeepFM_Enhanced(nn.Module):
    def __init__(self, feature_map, model_cfg):
        super(MM_xDeepFM_Enhanced, self).__init__()
        
        self.emb_dim = model_cfg.get("embedding_dim", 128)
        mm_input_dim = 128
        
        # Embeddings
        self.item_emb = nn.Embedding(91718, self.emb_dim, padding_idx=0)
        self.cate_emb = nn.Embedding(11, self.emb_dim) # Likes/Views
        self.user_emb = nn.Parameter(torch.zeros(1, self.emb_dim)) 
        
        # Projection Multimodale
        self.mm_proj = nn.Sequential(
            nn.Linear(mm_input_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            DICE(self.emb_dim, dim=2)
        )
        
        # 6 Champs
        self.num_fields = 6 
        
        # --- AMÉLIORATION 1 : Attention Layer ---
        self.hist_attn = AttentionPooling(self.emb_dim)
        
        # --- AMÉLIORATION 2 : CIN Profond ---
        cin_layer_units = model_cfg.get("cin_layer_units", [256, 128, 64])
        self.cin = CIN(self.emb_dim, self.num_fields, cin_layer_units)
        
        # --- AMÉLIORATION 3 : Linear Part ---
        self.linear = nn.Linear(self.num_fields * self.emb_dim, 1)
        
        # --- DNN ---
        flatten_dim = self.num_fields * self.emb_dim
        self.dnn = nn.Sequential(
            nn.Linear(flatten_dim, 1024), # Plus large
            nn.BatchNorm1d(1024),
            DICE(1024),
            nn.Dropout(model_cfg.get("net_dropout", 0.3)),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            DICE(512),
            nn.Dropout(model_cfg.get("net_dropout", 0.3)),
            
            nn.Linear(512, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch_dict):
        # Inputs
        item_id = batch_dict['item_id'].long()
        item_mm = batch_dict['item_emb_d128'].float()
        likes = batch_dict['likes_level'].long()
        views = batch_dict['views_level'].long()
        hist_ids = batch_dict.get('item_seq', None)
        
        batch_size = item_id.size(0)
        
        # Features
        user_feat = self.user_emb.expand(batch_size, -1)
        like_feat = self.cate_emb(likes)
        view_feat = self.cate_emb(views)
        item_id_feat = self.item_emb(item_id)
        item_img_feat = self.mm_proj(item_mm)
        
        # Cible combinée pour l'attention
        target_combined = item_id_feat + item_img_feat
        
        if hist_ids is not None:
            seq_emb = self.item_emb(hist_ids)
            mask = (hist_ids == 0)
            # Attention Pooling au lieu de Mean Pooling
            hist_feat = self.hist_attn(target_combined, seq_emb, mask)
        else:
            hist_feat = torch.zeros_like(item_id_feat)

        # Stack (B, 6, D)
        sparse_inputs = torch.stack([
            user_feat, like_feat, view_feat, item_id_feat, item_img_feat, hist_feat
        ], dim=1)
        
        flat_inputs = sparse_inputs.view(batch_size, -1)
        
        # Calculs parallèles
        linear_out = self.linear(flat_inputs)
        cin_out = self.cin(sparse_inputs)
        dnn_out = self.dnn(flat_inputs)
        
        return self.sigmoid(linear_out + cin_out + dnn_out).squeeze(-1)

def build_model(feature_map, model_cfg):
    return MM_xDeepFM_Enhanced(feature_map, model_cfg)