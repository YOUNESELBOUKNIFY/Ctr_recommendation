import torch
import torch.nn as nn
import torch.nn.functional as F

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Linear
        outputs += inputs
        return outputs

class SASRecBlock(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(SASRecBlock, self).__init__()
        self.hidden_units = hidden_units
        self.layer_norm1 = nn.LayerNorm(hidden_units)
        self.self_attention = nn.MultiheadAttention(hidden_units, num_heads, dropout=dropout_rate, batch_first=True)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.layer_norm2 = nn.LayerNorm(hidden_units)
        self.feed_forward = PointWiseFeedForward(hidden_units, dropout_rate)

    def forward(self, inputs, attention_mask):
        # inputs: (Batch, Seq, Emb)
        # mask: (Batch, Seq) -> True = Padding
        
        # Self Attention
        # Note: key_padding_mask attend True pour les éléments à ignorer
        normalized_inputs = self.layer_norm1(inputs)
        att_outputs, _ = self.self_attention(normalized_inputs, normalized_inputs, normalized_inputs, 
                                             key_padding_mask=attention_mask)
        
        att_outputs = att_outputs + inputs
        
        # Feed Forward
        normalized_att_outputs = self.layer_norm2(att_outputs)
        outputs = self.feed_forward(normalized_att_outputs)
        
        return outputs

class MM_SASRec(nn.Module):
    def __init__(self, feature_map, model_cfg):
        super(MM_SASRec, self).__init__()
        
        self.emb_dim = model_cfg.get("embedding_dim", 128)
        self.max_len = model_cfg.get("max_len", 20)
        mm_input_dim = 128
        
        # --- Embeddings ---
        self.item_emb = nn.Embedding(91718, self.emb_dim, padding_idx=0)
        self.cate_emb = nn.Embedding(11, self.emb_dim)
        
        # Positional Embedding (Crucial pour SASRec)
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
        self.attention_layernorm = nn.LayerNorm(self.emb_dim)
        self.attn_linear = nn.Linear(self.emb_dim, 1)
        
        # --- MLP Final ---
        # User + Ctx + Target + History
        input_dim = self.emb_dim * 4 
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
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
        
        # 1. Target Features
        target_id = self.item_emb(item_id)
        target_mm = self.mm_proj(item_mm)
        target_combined = target_id + target_mm # Fusion
        
        # 2. Context Features
        ctx_feat = self.cate_emb(likes) + self.cate_emb(views)
        
        # 3. SASRec Sequence Modeling
        if hist_ids is not None:
            # (Batch, Seq)
            seq_emb = self.item_emb(hist_ids)
            
            # Ajout Positional Embedding
            positions = torch.arange(self.max_len, device=device).unsqueeze(0).expand(batch_size, -1)
            seq_emb += self.pos_emb(positions)
            
            # Mask (True = Padding = 0)
            padding_mask = (hist_ids == 0)
            
            # Passage dans les blocs SASRec
            sas_feat = seq_emb
            for block in self.sas_blocks:
                sas_feat = block(sas_feat, padding_mask)
            
            # Attention par rapport à la Cible (Target-Aware)
            # Query = Target, Key = SAS Output
            # (B, 1, D) * (B, D, Seq) -> (B, 1, Seq)
            query = target_combined.unsqueeze(1)
            sas_feat = self.attention_layernorm(sas_feat)
            
            # Simple Dot Product Attention
            scores = torch.bmm(query, sas_feat.transpose(1, 2))
            scores = scores.squeeze(1).masked_fill(padding_mask, -1e9)
            weights = F.softmax(scores, dim=1).unsqueeze(-1)
            
            history_repr = torch.sum(sas_feat * weights, dim=1) # (B, Emb)
        else:
            history_repr = torch.zeros_like(target_combined)

        # 4. Fusion Finale
        # Placeholder User (Zero)
        user_feat = torch.zeros_like(target_combined)
        
        concat_feat = torch.cat([user_feat, ctx_feat, target_combined, history_repr], dim=1)
        
        return self.sigmoid(self.mlp(concat_feat)).squeeze(-1)

def build_model(feature_map, model_cfg):
    return MM_SASRec(feature_map, model_cfg)