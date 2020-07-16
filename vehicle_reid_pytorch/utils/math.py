import torch
import numpy as np
import cv2


def euclidean_dist(x, y):
    """

    :param torch.Tensor x:
    :param torch.Tensor y:
    :rtype: torch.Tensor
    :return:  dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).view(m, 1)
    yy = torch.pow(y, 2).sum(1, keepdim=True).view(n, 1).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    return dist.clamp(0).sqrt()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def near_convex(xys, threshold=20):
    """
    检查一个四边形是否为凸四边形

    近似是凸的也不能取。因此检查四个内角，如果内角接近180°则直接pass。

    :param np.ndarray xys: 4*2, 4个点的xy坐标
    :param threshold: abs degree of the diff with 180°
    :return:
    """
    vectors = np.empty([4, 2])
    vectors[:3, :] = xys[1:] - xys[:-1]
    vectors[3, :] = xys[0] - xys[-1]

    # 通过叉积判断凸性
    cross = np.cross(vectors, vectors[[3, 0, 1, 2]])
    if np.any(cross > 0) and np.any(cross < 0):
        return True

    # 近似凸也要去掉
    angles = np.empty(4)
    norm_dot = np.sqrt(np.sum(vectors[:3, :] ** 2, axis=1) * np.sum(vectors[1:, :] ** 2, axis=1)).clip(1e-7, None)
    angles[:3] = np.arccos(np.sum(vectors[:3, :] * vectors[1:, :], axis=1) / norm_dot)
    norm_dot:np.ndarray = (np.sum(vectors[0, :] ** 2) * np.sum(vectors[3, :] ** 2)).clip(1e-7, None)
    angles[3] = np.arccos(np.sum(vectors[0, :] * vectors[3, :]) / norm_dot)

    if np.any(np.abs(angles) < threshold / 180 * np.pi):
        return True
    return False


def perspective_transform(image, quad_pts, target_pts=None, output_size=(128, 128)):
    """

    :param image:
    :param quad_pts:
    :param output_size:
    :param context_size:
    :return:
    """
    quad_pts = quad_pts.astype(np.float32)

    x, y = output_size
    if target_pts is None:
        target_pts = [[0, 0], [x, 0], [x,  y], [0, y]]

    target_pts = np.array(target_pts).astype(np.float32)
    m = cv2.getPerspectiveTransform(quad_pts, target_pts)
    warp_img = cv2.warpPerspective(image, m, output_size)

    return warp_img

def pad_image_size_to_multiples_of(img, multiple, *, align):
    """
    '''Pad of image such that size of its edge is the least number that is a
    multiple of given multiple and larger than original image. The image
    will be placed in the center using pad_image_to_shape.

    :param multiple: the dividend of the targeting size of the image
    :param align: one of 'top-left' or 'center'
    """

    assert align in {'top-left', 'center'}, align

    h, w = img.shape[:2]
    d = multiple

    def canonicalize(s):
        v = s // d
        return (v + (v * d != s)) * d

    th, tw = map(canonicalize, (h, w))
    if align == 'top-left':
        tshape = (th, tw)
        if img.ndim == 3:
            tshape = tshape + (img.shape[2],)
        ret = np.zeros(tshape, dtype=img.dtype)
        ret[:h, :w] = img
        return ret
    else:
        assert align == 'center', align
        return pad_image_to_shape(img, (th, tw))