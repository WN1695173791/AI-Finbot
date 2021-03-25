import models

def test(model_class):
    model = model_class()
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)