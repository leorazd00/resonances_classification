import yaml
from dataset import ResonancesDataset
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


class predict:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def __call__(self, inputs):
        probs = self.predict_one_sample(inputs.[0].unsqueeze(0))

    def predict_one_sample(self, inputs):
        with torch.no_grad():
            inputs = inputs.to(self.device)
            self.model.eval()
            logit = self.model(inputs).cpu()
            probs = torch.nn.functional.softmax(logit, dim=-1).numpy()
        return probs