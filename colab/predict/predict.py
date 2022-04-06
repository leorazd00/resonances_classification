import pickle
from tqdm import tqdm
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


class predict:
    def __init__(self, model, device, path_to_save=None):
        self.model = model
        self.device = device
        self.path_to_save = path_to_save

    def __call__(self, dataset, path_label_encoder=None):
        label_encoder = pickle.load(open(path_label_encoder, 'rb'))
        
        for idx, sample in tqdm(enumerate(dataset)):
            probs = self.predict_one_sample(sample.unsqueeze(0))
            preds = label_encoder.inverse_transform(np.argmax(probs, axis=1))

            sample = sample.permute(1, 2, 0).cpu().detach().numpy()

            plt.imshow(sample)
            plt.title(preds[0])
            plt.savefig(f'{self.path_to_save}/img{idx}.png')


    def predict_one_sample(self, inputs):
        with torch.no_grad():
            inputs = inputs.to(self.device)
            self.model.eval()
            logit = self.model(inputs).cpu()
            probs = torch.nn.functional.softmax(logit, dim=-1).numpy()
        return probs