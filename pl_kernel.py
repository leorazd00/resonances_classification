import torch
import torch.nn as nn
from pathlib import Path
import pytorch_lightning as pl
from dataset import ResonancesDataset
from model import Model
from torch.utils.data import DataLoader
import yaml

with open('D:/projects/resonances_classification/config.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)


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
            num_workers=0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )


class ResonancesClassifier(pl.LightningModule):
    def __init__(self, n_classes: int):
        super().__init__()
        self.model = Model(n_classes)(name=config['model']['name'], pretrained=config['model']['pretrained'])
        self.n_classes = n_classes
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, labels=None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        inputs = batch[0]
        labels = batch[1]

        loss, outputs = self(inputs, labels)
        self.log('training_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch[0]
        labels = batch[1]

        loss, outputs = self(inputs, labels)
        self.log('validation_loss', loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.0001)


TRAIN_DIR = Path(config['path_train'])
TEST_DIR = Path(config['path_test'])

train_val_files = sorted(list(TRAIN_DIR.rglob('*.png')))
test_files = sorted(list(TEST_DIR.rglob('*.png')))

data_module = ResonancesDataModule(train_val_files, test_files, batch_size=config['batch_size'])
data_module.setup()

model = ResonancesClassifier(config['model']['n_classes'])

trainer = pl.Trainer(max_epochs=1)
trainer.fit(model, data_module)














