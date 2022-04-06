import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from model import Model
from torch.utils.data import DataLoader


class ResonancesDataModule(pl.LightningDataModule):
    def __init__(self, datasets: dict, config):
        super().__init__()
        self.datasets = datasets
        self.config = config

    def setup(self, stage=None):
        self.train_dataset = self.datasets['train_dataset']
        self.val_dataset = self.datasets['val_dataset']
        self.test_dataset = self.datasets['test_dataset']

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            **self.config['train_dataloader']
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            **self.config['val_dataloader']
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            **self.config['test_dataloader']
        )


class ResonancesClassifier(pl.LightningModule):
    def __init__(self, n_classes: int = 3,
                 name: str = 'resnet18',
                 pretrained: bool = False,
                 lr=1e-4):
        super().__init__()

        self.model = Model(n_classes)(name=name, pretrained=True)
        self.n_classes = n_classes
        self.lr = lr

        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        pred = self(inputs)
        loss = self.criterion(pred, labels)

        self.train_acc(pred, labels)
        self.log('train_acc', self.train_acc, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch

        pred = self(inputs)
        loss = self.criterion(pred, labels)

        self.valid_acc(pred, labels)
        self.log('valid_loss', loss, on_epoch=True, prog_bar=True)
        self.log('valid_acc', self.valid_acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


class pl_kernel:
    def __init__(self, datasets, config):
        self.datasets = datasets
        self.config = config

    def __call__(self):
        model = ResonancesClassifier(**self.config['model'])
        dataset = ResonancesDataModule(self.datasets, self.config['model'])
        dataset.setup()

        return model, dataset
