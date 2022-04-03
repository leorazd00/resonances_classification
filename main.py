from pathlib import Path
from dataset import ResonancesDataset

TRAIN_DIR = Path('D:/projects/resonances_classification/data/train')
TEST_DIR = Path('D:/projects/resonances_classification/data/test')

train_val_files = sorted(list(TRAIN_DIR.rglob('*.png')))
test_files = sorted(list(TEST_DIR.rglob('*.png')))

data = ResonancesDataset(train_val_files, 'train')
print(data[0][0].shape)