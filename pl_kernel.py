from pathlib import Path
import pytorch_lightning as pl
from dataset import ResonancesDataset
from torch.utils.data import DataLoader
import yaml

with open('D:/projects/resonances_classification/config.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.BaseLoader)

TRAIN_DIR = Path(config['path_train'])
TEST_DIR = Path(config['path_test'])

train_val_files = sorted(list(TRAIN_DIR.rglob('*.png')))
test_files = sorted(list(TEST_DIR.rglob('*.png')))


class ResonancesDataModule(pl.LightningDataModule):
    def __init__(self, train_img, test_img, batch_size):
        super().__init__()
        self.train_img = train_img
        self.test_img = test_img
        self.batch_size = batch_size

    def setup(self):
        self.train_dataset = ResonancesDataset(self.train_img, mode='train')
        self.val_dataset = ResonancesDataset(self.train_img, mode='val')
        self.test_dataset = ResonancesDataset(self.test_img, mode='test')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=-1
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


data_module = ResonancesDataModule(train_val_files, test_files, batch_size=config['batch_size'])
data_module.setup()

print(data_module)