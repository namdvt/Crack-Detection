import pywt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import torch
import cv2
import random
from glob import glob

from torch.utils.data.dataset import random_split


def normalize(x):
    maximum = x.max()
    minimum = x.min()
    if maximum == minimum:
        return x
    return (x - minimum)/(maximum - minimum)


def split(dataset):
    num_train = int(len(dataset) * 0.8)
    num_val = len(dataset) - num_train
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    return train_dataset, val_dataset


class ImageDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.indexes = glob(root + '/cracked/*') + glob(root + '/uncracked/*')

    def __getitem__(self, index):
        image_address = self.indexes[index]
        image = cv2.imread(image_address, cv2.IMREAD_GRAYSCALE)

        label = image_address.split('/train/')[1].split('/')[0]
        if label == 'cracked':
            label = 1
        else:
            label = 0

        return image, label

    def __len__(self):
        return len(self.indexes)


class DataTransformation(Dataset):
    def __init__(self, dataset, augment=False):
        self.dataset = dataset
        self.augment = augment

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        if self.augment:
            # rotate
            rotate = random.randint(0, 3)
            if rotate == 1:
                image = cv2.rotate(image, cv2.ROTATE_180)
            if rotate == 2:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            if rotate == 3:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # flip
            flip = random.randint(0, 2)
            if flip == 1:
                image = cv2.flip(image, 0)
            if flip == 2:
                image = cv2.flip(image, 1)

        # resize and return
        image = cv2.resize(image, dsize=(444, 444))
        coeffs2 = pywt.dwt2(image, 'bior1.3')
        _, (LH, HL, HH) = coeffs2
        LH = normalize(F.to_tensor(LH))
        HL = normalize(F.to_tensor(HL))
        HH = normalize(F.to_tensor(HH))

        image = torch.cat([LH, HL, HH], dim=0)

        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        label = torch.tensor(label)

        return image.float(), label

    def __len__(self):
        return len(self.dataset)


def get_loader(root, batch_size):
    dataset = ImageDataset(root=root)

    train, val = split(dataset)
    train = DataTransformation(train, augment=True)
    val = DataTransformation(val)

    train_loader = DataLoader(dataset=train,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=0,
                              pin_memory=True)

    val_loader = DataLoader(dataset=val,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=0,
                            pin_memory=True)

    return train_loader, val_loader


if __name__ == '__main__':
    train_loader, val_loader = get_loader(root='data/crack-identification-ce784a-2020-iitk/train', batch_size=2)
    for i, LH, label in train_loader:
        print()
