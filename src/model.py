from fuxictr.pytorch.models import TabTransformer

def build_model(feature_map, config):
    """
    Crée un modèle TabTransformer avec la configuration YAML
    """
    model = TabTransformer(feature_map=feature_map, **config)
    return model
