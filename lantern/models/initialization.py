import torch


def init_weights_normal(model):
    if type(model) == torch.nn.Linear:
        if hasattr(model, 'weight'):
            torch.nn.init.kaiming_normal_(
                model.weight, a=0.0, nonlinearity='relu', mode='fan_in')
