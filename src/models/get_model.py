# src/models/get_model.py

import torch
import torchvision.models as models
# from models import best_model
from torchvision.models import resnet18, efficientnet_b0
from torchvision.models import ViT_B_16_Weights, vit_b_16

def get_model(model_name: str, num_classes=3):
    if model_name.lower() == "resnet18":
        model = resnet18(weights="DEFAULT")
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name.lower() == "efficientnet_b0":
        model = efficientnet_b0(weights="DEFAULT")
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name.lower() == "vit_b_16":
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"❌ Unknown model name: {model_name}")
    return model
