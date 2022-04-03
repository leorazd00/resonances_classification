import torch.nn as nn
from torchvision import models


class Model:
    def __init__(self, n_classes: int):
        self.n_classes = n_classes

    def __call__(self, name: str, pretrained: bool):
        if name == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            resnet.fc = nn.Linear(512, self.n_classes)
            return resnet

        elif name == 'efficientnet_b0':
            efficientnet = models.efficientnet_b0(pretrained=pretrained)
            efficientnet.classifier[1] = nn.Linear(1280, self.n_classes)
            return efficientnet
