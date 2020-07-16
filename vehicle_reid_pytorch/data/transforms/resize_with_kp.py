import albumentations as albu
from albumentations.augmentations import functional
import numpy as np
import cv2
import random


class ResizeWithKp(albu.Resize):
    def apply_to_keypoint(self, keypoint, **params):
        x = int(keypoint[0] / params["cols"] * self.width)
        y = int(keypoint[1] / params["rows"] * self.height)
        return (x, y, 0, 0)

class MultiScale(albu.ImageOnlyTransform):
    def __init__(self, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1):
        super(MultiScale, self).__init__(always_apply, p)
        self.interpolation = interpolation
        
    def apply(self, image, **params):
        height, width, _ = image.shape
        if width > 320 or height > 320:
            scale = random.random() * 0.4 + 0.2
            image = functional.resize(image, height=int(height * scale), width=int(width * scale), interpolation=self.interpolation)
            return functional.resize(image, height=height, width=width, interpolation=self.interpolation)
        else:
            return image
    
    def get_transform_init_args_names(self):
        return ("interpolation", )
    