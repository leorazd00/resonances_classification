import cv2
import pickle
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from torch.utils.data import Dataset

DATA_MODES = ['train', 'val', 'test']
RESCALE_SIZE = 256

class ResonancesDataset(Dataset):
    def __init__(self, files, mode):
        super().__init__()
        self.files = sorted(files)
        self.mode = mode

        if self.mode not in DATA_MODES:
            print(f'{self.mode}  is not correct; correct modes: {DATA_MODES}')
            raise NameError

        self.label_encoder = LabelEncoder()

        if self.mode != 'test':
            self.labels = [path.parent.name for path in self.files]
            self.label_encoder.fit(self.labels)

            with open('label_encoder.pkl', 'wb') as le_dump_file:
                pickle.dump(self.label_encoder, le_dump_file)

    def __len__(self):
        return len(self.files)

    def load_sample(self, file):
        image = cv2.imread(str(file))
        return image

    def __getitem__(self, index):
        if self.mode == 'train':
            transform = transforms.Compose([
                # transforms.Resize(size=(RESCALE_SIZE, RESCALE_SIZE)),
                # transforms.RandomRotation(degrees=30),
                # transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(hue=.1, saturation=.1),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                # transforms.Resize(size=(RESCALE_SIZE, RESCALE_SIZE)),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        x = self.load_sample(self.files[index])
        x = transform(x)
        if self.mode == 'test':
            return x
        else:
            label = self.labels[index]
            label_id = self.label_encoder.transform([label])
            y = label_id.item()
            return x, y