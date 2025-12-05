import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Couche GAT (Graph Attention Network) qui traite les features comme un graphe.
    """
    def __init__(self, in_dim, out_dim, dropout, alpha=0.2, heads=2):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.head_dim = out_dim // heads
        
        assert self.head_dim * heads == out_dim, "out_dim doit être divisible par heads"

        # Poids pour la projection linéaire
        self.W = nn.Parameter(torch.zeros(size=(heads, in_dim, self.head_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Poids pour l'attention
        self.a = nn.Parameter(torch.zeros(size=(heads, 2 * self.head_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h):
        # h: (Batch, Num_Fields, In_Dim)
        batch_size, num_nodes, _ = h.size()
        
        # 1. Projection Linéaire
        # (Batch, Nodes, Heads, Head_Dim)
        h_prime = torch.einsum('bni,hio->bnho', h, self.W)
        
        # 2. Préparation Attention (All-to-All)
        h_prime_i = h_prime.unsqueeze(2).expand(-1, -1, num_nodes, -1, -1)
        h_prime_j = h_prime.unsqueeze(1).expand(-1, num_nodes, -1, -1, -1)
        
        a_input = torch.cat([h_prime_i, h_prime_j], dim=-1) 
        
        # 3. Scores d'attention
        e = torch.einsum('bnmhd,hdk->bhnm', a_input, self.a).squeeze(-1)
        e = self.leakyrelu(e)
        
        # 4. Softmax
        attention = F.softmax(e, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 5. Agrégation
        h_new = torch.einsum('bhnm,bmho->bnho', attention, h_prime)
        
        # 6. Concatenation (CORRECTION ICI : .reshape au lieu de .view)
        h_new = h_new.reshape(batch_size, num_nodes, self.out_dim)
        
        # Résiduel
        return F.elu(h_new + h) if self.in_dim == self.out_dim else F.elu(h_new)

class SENetLayer(nn.Module):
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
        z = torch.mean(x, dim=-1)
        weights = self.excitation(z)
        return x * weights.unsqueeze(-1)

class BilinearInteraction(nn.Module):
    def __init__(self, input_dim, num_fields, bilinear_type="each"):
        super(BilinearInteraction, self).__init__()
        self.W_list = nn.ParameterList([
            nn.Parameter(torch.Tensor(input_dim, input_dim)) for _ in range(num_fields - 1)
        ])
        for w in self.W_list: nn.init.xavier_normal_(w)

    def forward(self, x):
        inputs = torch.split(x, 1, dim=1)
        p = []
        for i in range(len(inputs) - 1):
            if i < len(self.W_list):
                vid = torch.matmul(inputs[i].squeeze(1), self.W_list[i])
                for j in range(i + 1, len(inputs)):
                    p.append(vid * inputs[j].squeeze(1))
        return torch.stack(p, dim=1)

class MM_Graph_FiBiNET(nn.Module):
    def __init__(self, feature_map, model_cfg):
        super(MM_Graph_FiBiNET, self).__init__()
        
        self.emb_dim = model_cfg.get("embedding_dim", 128)
        mm_input_dim = 128
        
        # Embeddings
        self.item_emb = nn.Embedding(91718, self.emb_dim, padding_idx=0)
        self.cate_emb = nn.Embedding(11, self.emb_dim)
        self.user_emb = nn.Embedding(20000, self.emb_dim) # Placeholder
        
        # Projection MM
        self.mm_proj = nn.Sequential(
            nn.Linear(mm_input_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.GELU()
        )
        
        self.num_fields = 6 
        
        # --- GRAPH LAYER ---
        self.gnn = GraphAttentionLayer(self.emb_dim, self.emb_dim, dropout=0.1, heads=4)
        
        # --- FiBiNET ---
        self.senet = SENetLayer(self.num_fields, reduction_ratio=2)
        self.bilinear = BilinearInteraction(self.emb_dim, self.num_fields, bilinear_type="each")
        
        # --- MLP ---
        num_pairs = (self.num_fields * (self.num_fields - 1)) // 2
        total_input_dim = (self.num_fields * self.emb_dim) + (num_pairs * self.emb_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
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
        user_feat = torch.zeros((batch_size, self.emb_dim), device=item_id.device)
        like_feat = self.cate_emb(likes)
        view_feat = self.cate_emb(views)
        item_id_feat = self.item_emb(item_id)
        item_img_feat = self.mm_proj(item_mm)
        
        if hist_ids is not None:
            seq_emb = self.item_emb(hist_ids)
            mask = (hist_ids == 0)
            seq_emb = seq_emb.masked_fill(mask.unsqueeze(-1), 0)
            seq_sum = torch.sum(seq_emb, dim=1)
            # Protection div/0
            seq_count = torch.sum((~mask).float(), dim=1, keepdim=True).clamp(min=1)
            hist_feat = seq_sum / seq_count
        else:
            hist_feat = torch.zeros_like(item_id_feat)

        # Stack Features
        raw_features = torch.stack([
            user_feat, like_feat, view_feat, item_id_feat, item_img_feat, hist_feat
        ], dim=1)
        
        # 1. GNN : Enrichissement contextuel
        gnn_features = self.gnn(raw_features)
        
        # 2. FiBiNET : Interactions explicites
        senet_output = self.senet(gnn_features)
        bilinear_output = self.bilinear(senet_output)
        
        # 3. Fusion
        c_input = torch.cat([
            gnn_features.view(batch_size, -1),
            bilinear_output.view(batch_size, -1)
        ], dim=1)
        
        return self.sigmoid(self.mlp(c_input)).squeeze(-1)

def build_model(feature_map, model_cfg):
    return MM_Graph_FiBiNET(feature_map, model_cfg)