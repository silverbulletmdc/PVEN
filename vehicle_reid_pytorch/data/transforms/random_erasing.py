# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import math
import random
import numpy as np
import albumentations as albu


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def handle_numpy(self, img):
        """
        img为未归一化的(H,W,C)，为albumentation使用
        :param img:
        :return:
        """
        shape = img.shape
        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = shape[0] * shape[1]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < shape[1] and h < shape[0]:
                x1 = random.randint(0, shape[0] - h)
                y1 = random.randint(0, shape[1] - w)
                if shape[2] == 3:
                    img[x1:x1 + h, y1:y1 + w, 0] = self.mean[0] * 255
                    img[x1:x1 + h, y1:y1 + w, 1] = self.mean[1] * 255
                    img[x1:x1 + h, y1:y1 + w, 2] = self.mean[2] * 255
                else:
                    img[x1:x1 + h, y1:y1 + w, 0] = self.mean[0]
                return img

        return img

    def handel_pil(self, img):
        shape = img.size()
        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = shape[1] * shape[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < shape[2] and h < shape[1]:
                x1 = random.randint(0, shape[1] - h)
                y1 = random.randint(0, shape[2] - w)
                if shape[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            return self.handle_numpy(img)
        else:
            return self.handel_pil(img)


def AlbuRandomErasing(probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
    fun = RandomErasing(probability, sl, sh, r1, mean)

    def wrapper(x, **kwargs):
        return fun(x)

    return albu.Lambda(image=wrapper, mask=wrapper)
