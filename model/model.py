import torch.nn as nn
from torchvision import models


class model:
    def __init__(self, n_classes: int = 3,
                 name: str = 'resnet18',
                 pretrained: bool = False):

        self.n_classes = n_classes
        self.name = name
        self.pretrained = pretrained

    def __call__(self):
        if self.name == 'resnet18':
            # resnet = models.resnet18(pretrained=self.pretrained)
            # resnet.fc = nn.Linear(512, self.n_classes)
            resnet = models.resnet18(pretrained=self.pretrained, num_classes=3)
            return resnet

        elif self.name == 'efficientnet_b0':
            efficientnet = models.efficientnet_b0(pretrained=self.pretrained)
            efficientnet.classifier[1] = nn.Linear(1280, self.n_classes)
            return efficientnet
