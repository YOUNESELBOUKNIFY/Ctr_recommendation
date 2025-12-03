import torch
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer

class MMCTRModel(nn.Module):
    def __init__(self, feature_map, model_cfg):
        super(MMCTRModel, self).__init__()
        
        # 1. Configuration des inputs
        # Ordre important : [likes_level, views_level, item_id]
        # Doit correspondre à l'ordre dans le forward()
        self.cat_vocab_sizes = [11, 11, 91718] 
        
        # Dimension des embeddings multimodaux (item_emb_d128)
        self.continuous_dim = 128 
        
        # 2. Initialisation du TabTransformer
        self.tab_transformer = TabTransformer(
            categories=self.cat_vocab_sizes,    # Tuple/List des tailles de vocab
            num_continuous=self.continuous_dim, # Dimension des features continues (128)
            dim=model_cfg["embedding_dim"],     # Dim interne des embeddings (ex: 32 ou 64)
            depth=model_cfg.get("transformer_n_layers", 2),
            heads=model_cfg.get("transformer_n_heads", 4),
            attn_dropout=model_cfg.get("transformer_dropout", 0.1),
            ff_dropout=model_cfg.get("net_dropout", 0.1),
            mlp_hidden_mults=(4, 2),
            mlp_act=nn.ReLU(),
            dim_out=1  # Sortie binaire (logit)
        )
        
        # Activation finale pour obtenir une probabilité entre 0 et 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch_dict):
        """
        batch_dict contient :
         - 'likes_level': (B,)
         - 'views_level': (B,)
         - 'item_id': (B,)
         - 'item_emb_d128': (B, 128)
        """
        
        # 1. Préparer les features Catégorielles (x_categ)
        # On doit les empiler : (Batch, 3)
        # Attention aux dimensions, il faut parfois unsqueeze si (B,) -> (B, 1)
        likes = batch_dict['likes_level'].long()
        views = batch_dict['views_level'].long()
        items = batch_dict['item_id'].long()
        
        # TabTransformer attend (Batch, Num_Categories)
        x_categ = torch.stack([likes, views, items], dim=1)
        
        # 2. Préparer les features Continues (x_cont)
        # Ici c'est nos embeddings multimodaux pré-calculés
        x_cont = batch_dict['item_emb_d128'].float()
        
        # 3. Passage dans le modèle
        # TabTransformer retourne les logits directement
        logits = self.tab_transformer(x_categ, x_cont)
        
        # 4. Retourner la probabilité
        return self.sigmoid(logits).squeeze(-1)

# Fonction builder appelée par main.py
def build_model(feature_map, model_cfg):
    return MMCTRModel(feature_map, model_cfg)