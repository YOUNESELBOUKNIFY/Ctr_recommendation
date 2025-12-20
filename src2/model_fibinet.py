import torch
import torch.nn as nn

class SENetLayer(nn.Module):
    """
    Squeeze-and-Excitation sur les fields.
    x: (B, F, E) -> pondère chaque field.
    """
    def __init__(self, num_fields, reduction_ratio=3):
        super().__init__()
        reduced_size = max(1, num_fields // reduction_ratio)
        self.excitation = nn.Sequential(
            nn.Linear(num_fields, reduced_size),
            nn.ReLU(),
            nn.Linear(reduced_size, num_fields),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = torch.mean(x, dim=-1)           # (B, F)
        w = self.excitation(z)             # (B, F)
        return x * w.unsqueeze(-1)         # (B, F, E)


class BilinearInteraction(nn.Module):
    """
    Bilinear interactions entre fields.
    - all : une matrice W partagée
    - each: une matrice W_i par field
    """
    def __init__(self, input_dim, num_fields, bilinear_type="all"):
        super().__init__()
        self.bilinear_type = bilinear_type
        self.num_fields = num_fields
        self.input_dim = input_dim

        if bilinear_type == "all":
            self.W = nn.Parameter(torch.empty(input_dim, input_dim))
            nn.init.xavier_normal_(self.W)
        elif bilinear_type == "each":
            self.W_list = nn.ParameterList([
                nn.Parameter(torch.empty(input_dim, input_dim)) for _ in range(num_fields)
            ])
            for w in self.W_list:
                nn.init.xavier_normal_(w)
        else:
            raise ValueError("bilinear_type must be 'all' or 'each'")

    def forward(self, x):
        # x: (B, F, E)
        B, F, E = x.shape
        inputs = torch.split(x, 1, dim=1)  # list of (B,1,E)

        p = []
        if self.bilinear_type == "all":
            # proj de tous les fields
            xw = torch.matmul(x, self.W)  # (B,F,E)
            for i in range(F):
                vi = inputs[i].squeeze(1)          # (B,E)
                for j in range(i + 1, F):
                    p.append(vi * xw[:, j, :])     # (B,E)
        else:  # each
            for i in range(F):
                vi = inputs[i].squeeze(1)          # (B,E)
                viw = torch.matmul(vi, self.W_list[i])  # (B,E)
                for j in range(i + 1, F):
                    vj = inputs[j].squeeze(1)
                    p.append(viw * vj)

        return torch.stack(p, dim=1)  # (B, num_pairs, E)


class MM_FiBiNET(nn.Module):
    def __init__(self, feature_map, model_cfg):
        super().__init__()

        self.emb_dim = int(model_cfg.get("embedding_dim", 128))
        mm_input_dim = int(model_cfg.get("mm_input_dim", 128))

        # vocab sizes (tu peux les mettre dans le YAML si tu veux)
        item_vocab = int(model_cfg.get("item_vocab_size", 91718))
        user_vocab = int(model_cfg.get("user_vocab_size", 20000))
        cate_vocab = int(model_cfg.get("cate_vocab_size", 11))

        self.item_emb = nn.Embedding(item_vocab, self.emb_dim, padding_idx=0)
        self.user_emb = nn.Embedding(user_vocab, self.emb_dim)
        self.cate_emb = nn.Embedding(cate_vocab, self.emb_dim)

        self.mm_proj = nn.Sequential(
            nn.Linear(mm_input_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.ReLU()
        )

        # Fields: [User, Like, View, ItemID, ItemMM, HistPooled]
        self.num_fields = 6
        senet_reduction = int(model_cfg.get("senet_reduction", 2))
        self.senet = SENetLayer(self.num_fields, reduction_ratio=senet_reduction)

        bilinear_type = model_cfg.get("bilinear_type", "all")
        self.bilinear = BilinearInteraction(self.emb_dim, self.num_fields, bilinear_type=bilinear_type)

        num_pairs = (self.num_fields * (self.num_fields - 1)) // 2
        total_input_dim = (self.num_fields + num_pairs) * self.emb_dim

        dropout = float(model_cfg.get("net_dropout", 0.2))
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch_dict):
        item_id = batch_dict["item_id"].long()
        item_mm = batch_dict["item_emb_d128"].float()
        likes = batch_dict["likes_level"].long()
        views = batch_dict["views_level"].long()
        hist_ids = batch_dict.get("item_seq", None)

        B = item_id.size(0)

        # User placeholder (si tu n'as pas user_id embedding utilisé)
        user_feat = torch.zeros((B, self.emb_dim), device=item_id.device)

        like_feat = self.cate_emb(likes)
        view_feat = self.cate_emb(views)
        item_id_feat = self.item_emb(item_id)
        item_mm_feat = self.mm_proj(item_mm)

        if hist_ids is not None:
            seq_emb = self.item_emb(hist_ids.long())            # (B, S, E)
            mask = (hist_ids == 0)                              # padding
            seq_emb = seq_emb * (~mask).unsqueeze(-1).float()
            seq_sum = seq_emb.sum(dim=1)
            seq_cnt = (~mask).float().sum(dim=1, keepdim=True).clamp(min=1.0)
            hist_feat = seq_sum / seq_cnt
        else:
            hist_feat = torch.zeros_like(item_id_feat)

        # stack fields -> (B,F,E)
        x = torch.stack([user_feat, like_feat, view_feat, item_id_feat, item_mm_feat, hist_feat], dim=1)

        x_senet = self.senet(x)
        x_bi = self.bilinear(x_senet)

        c = torch.cat([x_senet.reshape(B, -1), x_bi.reshape(B, -1)], dim=1)
        logits = self.mlp(c)
        return self.sigmoid(logits).squeeze(-1)


def build_model(feature_map, model_cfg):
    return MM_FiBiNET(feature_map, model_cfg)
