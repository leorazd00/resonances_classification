import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from dataset import ResonancesDataset
from model import Model
from torch.utils.data import DataLoader


class ResonancesDataModule(pl.LightningDataModule):
    def __init__(self, train_img, test_img, batch_size):
        super().__init__()
        self.train_img = train_img
        self.test_img = test_img
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = ResonancesDataset(self.train_img, mode='train')
        self.val_dataset = ResonancesDataset(self.train_img, mode='val')
        self.test_dataset = ResonancesDataset(self.test_img, mode='test')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1
        )


class ResonancesClassifier(pl.LightningModule):
    def __init__(self, n_classes: int):
        super().__init__()
        self.model = Model(n_classes)(name=config['model']['name'], pretrained=config['model']['pretrained'])
        self.n_classes = n_classes
        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

    def forward(self, x, labels=None):
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
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.0001)