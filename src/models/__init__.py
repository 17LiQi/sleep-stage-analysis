from .attention_cnn import AttentionCNN
from .attention_resnet import AttentionResNet
from .attention_cnn_gru import AttentionCNNGRU
from .se_resnet import SEResNet
def get_model(name, **kwargs):
    models = {
        "AttentionCNN": AttentionCNN,
        "AttentionResNet": AttentionResNet,
        "AttentionCNNGRU": AttentionCNNGRU,
        "SEResNet": SEResNet
    }
    if name not in models:
        raise ValueError(f"模型 '{name}' 未在 models/__init__.py 中注册。")
    return models[name](**kwargs)