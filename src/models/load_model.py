import torch
import torch.nn as nn
from torchvision.models import resnet18


def load_model(model_path, num_classes=2):
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Handle case where state_dict is nested in a 'model' key or is a raw dict
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)

    model.eval()
    return model

