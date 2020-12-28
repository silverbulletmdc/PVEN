from torch.utils.data import DataLoader, Dataset
import os
import pickle
import numpy as np
import json
from pathlib import Path

from vehicle_reid_pytorch.utils.iotools import read_rgb_image
from vehicle_reid_pytorch.utils.visualize import visualize_img as visualize_img
from vehicle_reid_pytorch.utils.math import pad_image_size_to_multiples_of
from vehicle_reid_pytorch.data.transforms import AlbuRandomErasing
import matplotlib.pyplot as plt
import albumentations as albu
import cv2
from functools import partial


def get_training_albumentations():
    train_transform = [
        albu.LongestMaxSize(244),
        albu.HorizontalFlip(),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.PadIfNeeded(min_height=224, min_width=224, always_apply=True, border_mode=0),
        albu.RandomCrop(height=224, width=224, always_apply=True),
        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        # AlbuRandomErasing(0.5)
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    test_transform = [
        albu.Lambda(image=pad_image_to_multiplys_of(32), mask=pad_image_to_multiplys_of(32))
        # albu.LongestMaxSize(224),
        # albu.Lambda(image=pad_image_to_multiplys_of(32), mask=pad_image_to_multiplys_of(32))

        # albu.RandomCrop(height=320, width=320, always_apply=True)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def pad_image_to_multiplys_of(multiply=32, **kwargs):
    def _pad_image_to_multiplys_of(x, **kwargs):
        return pad_image_size_to_multiples_of(x, multiply, align='top-left')

    return _pad_image_to_multiplys_of


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor)
    ]
    return albu.Compose(_transform)


class VeRi3kParsingDataset(Dataset):
    CLASSES = ["background", "front", "back", "roof", "side"]

    def __init__(self, image_path, masks_path, augmentation=None, preprocessing=None,
                 subset='trainval'):
        self.metas = [os.path.splitext(fname)[0] for fname in os.listdir(masks_path)]
        self.masks_path = Path(masks_path)
        self.image_path = Path(image_path)
        if subset == 'trainval':
            # self.metas = self.metas[:-500]
            self.metas = self.metas
        elif subset == 'train':
            self.metas = self.metas[:-500]
        else:
            self.metas = self.metas[-500:]

        self.class_values = [self.CLASSES.index(cls) for cls in self.CLASSES]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, item):
        image_name = self.metas[item]
        img = read_rgb_image(f"{self.image_path/image_name}.jpg", format="ndarray")
        mask = cv2.imread(f"{self.masks_path/image_name}.png", cv2.IMREAD_UNCHANGED)
        masks = [mask == v for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float32')

        if self.augmentation:
            sample = self.augmentation(image=img, mask=mask)
            img = sample["image"]
            mask = sample["mask"]

        if self.preprocessing:
            sample = self.preprocessing(image=img, mask=mask)
            img = sample["image"]
            mask = sample["mask"]

        return img, mask

    def __len__(self):
        return len(self.metas)


class VehicleReIDParsingDataset(Dataset):
    """
    将reid的数据集转化成parsing数据集，仅测试使用
    """
    CLASSES = ["background", "back", "front", "side", "roof"]

    def __init__(self, dataset, augmentation=None, preprocessing=None, with_extra=False):
        self.augmetation = augmentation
        self.preprocessing = preprocessing
        self.dataset = dataset
        self.with_extra = with_extra

    def __getitem__(self, item):
        img_path = self.dataset[item]["image_path"]
        assert Path(img_path).exists(), f'{img_path} does not exist!'
        image = read_rgb_image(img_path)
        image = np.array(image)
        if self.augmetation:
            sample = self.augmetation(image=image)
            image = sample["image"]
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample["image"]

        if self.with_extra:
            return image, self.dataset[item]

        else:
            return image

    def __len__(self):
        return len(self.dataset)
