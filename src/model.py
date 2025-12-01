from tab_transformer_pytorch import TabTransformer
import torch

def build_model(feature_map, model_cfg):
    # Exemple : définir les catégories et les features continues
    categories = [11, 11, 91718]         # likes_level, views_level, item_id
    num_cont = 128                        # item_emb_d128 dimension

    model = TabTransformer(
        categories=categories,
        num_continuous=num_cont,
        dim=model_cfg["embedding_dim"],
        depth=4,
        heads=4,
        attn_dropout=model_cfg.get("attention_dropout", 0.1),
        ff_dropout=model_cfg.get("net_dropout", 0.0),
        mlp_hidden_mults=(4, 2),
        mlp_act=torch.nn.ReLU(),
        dim_out=1
    )
    return model
