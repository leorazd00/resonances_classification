# from pathlib import Path
# from dataset import ResonancesDataset

# TRAIN_DIR = Path('D:/projects/resonances_classification/data/train')
# TEST_DIR = Path('D:/projects/resonances_classification/data/test')

# train_val_files = sorted(list(TRAIN_DIR.rglob('*.png')))
# test_files = sorted(list(TEST_DIR.rglob('*.png')))

# data = ResonancesDataset(train_val_files, 'train')
# print(data[0][0].shape)
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from dataset import ResonancesDataset
from pl_kernel import pl_kernel

with open('config.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

TRAIN_DIR = Path('data/train')
TEST_DIR = Path('data/test')

train_val_files = sorted(list(TRAIN_DIR.rglob('*.png')))
test_files = sorted(list(TEST_DIR.rglob('*.png')))

train_val_labels = [path.parent.name for path in train_val_files]
train_files, val_files = train_test_split(train_val_files, test_size=0.25, \
                                          stratify=train_val_labels)

datasets = {'train_dataset': ResonancesDataset(train_files, mode='train'),
            'val_dataset': ResonancesDataset(train_files, mode='val'),
            'test_dataset': ResonancesDataset(train_files, mode='test')}

kernel = pl_kernel(datasets, config)
model, data_module = kernel()