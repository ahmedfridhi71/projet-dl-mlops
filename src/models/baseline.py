import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes=2, pretrained=True):
    """ResNet-18 pre-entraine sur ImageNet, avec une nouvelle tete a 2 sorties."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model