import albumentations as albu
from albumentations.pytorch import ToTensor
import numpy as np
import cv2
import torch

from .transforms import AlbuRandomErasing, ResizeWithKp, MultiScale


def get_training_albumentations(size=(256, 256), pad=10, re_prob=0.5, with_keypoints=False, ms_prob=0.5):
    h, w = size
    train_transform = [
        MultiScale(p=ms_prob),
        ResizeWithKp(h, w, interpolation=cv2.INTER_CUBIC),
        albu.PadIfNeeded(h + 2 * pad, w + 2 * pad, border_mode=cv2.BORDER_CONSTANT, value=0),
        albu.RandomCrop(height=h, width=w, always_apply=True),
        AlbuRandomErasing(re_prob),
    ]
    if with_keypoints:
        return albu.Compose(train_transform, keypoint_params=albu.KeypointParams(format='xy', remove_invisible=False))
    else:
        return albu.Compose(train_transform)


def get_validation_augmentations(size=(256, 256), with_keypoints=False):
    h, w = size
    test_transform = [
        ResizeWithKp(h, w),
    ]
    if with_keypoints:
        return albu.Compose(test_transform, keypoint_params=albu.KeypointParams(format='xy', remove_invisible=False))
    else:
        return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    x = np.transpose(x, [2, 0, 1])
    return torch.tensor(x)


def get_preprocessing(mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225)):
    _transform = [
        albu.Normalize(mean, std),
        albu.Lambda(image=to_tensor, mask=to_tensor)
    ]
    return albu.Compose(_transform)
