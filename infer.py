import yaml
import torch
from pathlib import Path
from predict import predict
from pl_kernel import pl_kernel
from dataset import ResonancesDataset


with open('config.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

TEST_DIR = Path('data/test')
test_files = sorted(list(TEST_DIR.rglob('*.png')))

test_dataset = ResonancesDataset(test_files, mode='test')

kernel = pl_kernel(config)
model = kernel.load_checkpoint(config['path_to_checkpoints'])

pred = predict(model, 'cpu', 'first_infer')

pred(test_dataset, path_label_encoder='label_encoder.pkl')
