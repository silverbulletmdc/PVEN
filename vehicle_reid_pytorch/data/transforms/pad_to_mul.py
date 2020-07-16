from functools import partial
from vehicle_reid_pytorch.utils.math import pad_image_size_to_multiples_of
import albumentations as albu
import numpy as np


def pad_image_to_shape(img, shape, *, return_padding=False):
    """
    Zeros pad the given image to given shape whiling keeping the image
    in the center;
    :param shape: (h, w)
    :param return_padding:
    """
    shape = list(shape[:2])
    if img.ndim > 2:
        shape.extend(img.shape[2:])
    shape = tuple(shape)

    h, w = img.shape[:2]
    assert w <= shape[1] and h <= shape[0]
    pad_width = shape[1] - w
    pad_height = shape[0] - h

    pad_w0 = pad_width // 2
    pad_w1 = shape[1] - (pad_width - pad_w0)
    pad_h0 = pad_height // 2
    pad_h1 = shape[0] - (pad_height - pad_h0)

    ret = np.zeros(shape, dtype=img.dtype)
    ret[pad_h0:pad_h1, pad_w0:pad_w1] = img
    if return_padding:
        return ret, (pad_h0, pad_w0)
    else:
        return ret


def AlbuPadImageToMultipliesOf(multiply=32, align="top-left", **kwargs):
    fun = partial(pad_image_size_to_multiples_of, multiply=multiply, align=align)
    return albu.Lambda(image=fun, mask=fun)
