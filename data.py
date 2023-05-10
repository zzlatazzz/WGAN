import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

from configs import TrainConfig


def image_transform(images):
    return np.log10(images + 1)

def inverse_image_transform(images):
    return 10 ** images - 1

class standard_scaler():
    def fit_transform(self, data):
        self.data_mean = np.mean(data, axis=0, keepdims=True)
        self.data_std = np.std(data, axis=0, keepdims=True)
        return (data - self.data_mean) / self.data_std

    def transform(self, data):
        return (data - self.data_mean) / self.data_std

    def inverse_transform(self, data):
        return data * self.data_std + self.data_mean

class MyDataset(Dataset):
    def __init__(self, imgs, momentums, points):
        self.imgs = torch.tensor(imgs, dtype=torch.float32)
        self.momentums = torch.tensor(momentums, dtype=torch.float32)
        self.points = torch.tensor(points, dtype=torch.float32)
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        return self.imgs[idx], self.momentums[idx], self.points[idx]


def get_data():
    config = TrainConfig()

    data_path = config.data_path
    data = np.load(data_path, allow_pickle=False)

    images = data['EnergyDeposit'].reshape((-1, 1, 30, 30))

    momentums = data['ParticleMomentum']
    points = data['ParticlePoint'][:,:2]

    images_train, images_test, momentums_train, momentums_test, points_train, points_test = train_test_split(image_transform(images), momentums, points, test_size=0.1, random_state=config.seed)

    momentums_transform = standard_scaler()
    points_transform = standard_scaler()

    momentums_train = momentums_transform.fit_transform(momentums_train)
    momentums_test = momentums_transform.transform(momentums_test)
    points_train = points_transform.fit_transform(points_train)
    points_test = points_transform.transform(points_test)

    train_dataset = MyDataset(images_train, momentums_train, points_train)
    test_dataset = MyDataset(images_test, momentums_test, points_test)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    train_loader2 = DataLoader(train_dataset, batch_size=config.batch_size_test, shuffle=True, pin_memory=True)

    test_loader = DataLoader(test_dataset, batch_size=config.batch_size_test, shuffle=True, pin_memory=True)

    return train_loader, test_loader, momentums_transform, points_transform
