# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from bisect import bisect_right
import torch
from logzero import logger
import cv2
import numpy as np
import time
import asranger as ranger

# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=1.0 / 3,
            warmup_iters=500,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
                # warmup_factor = self.warmup_factor * self.last_epoch
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


def make_optimizer(optim_name, model, base_lr, weight_decay, bias_lr_factor, momentum):
    """
    调低所有bias项的学习率。

    :param optim_name:
    :param model:
    :param base_lr:
    :param weight_decay:
    :param bias_lr_factor:
    :param momentum:
    :return:
    """
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = base_lr
        if "bias" in key:
            lr = base_lr * bias_lr_factor
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if optim_name == 'SGD':
        optimizer = getattr(torch.optim, optim_name)(params, momentum=momentum)
    elif 'Ranger' in optim_name:
        optimizer = getattr(ranger, optim_name)(params)
    else:
        optimizer = getattr(torch.optim, optim_name)(params)
    return optimizer


def make_warmup_scheduler(optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500,
                          warmup_method="linear",
                          last_epoch=-1):
    if last_epoch == 0:
        last_epoch = -1  # init时会自动变成0.否则会初始化错误
    scheduler = WarmupMultiStepLR(optimizer, milestones, gamma, warmup_factor, warmup_iters, warmup_method,
                                  last_epoch=last_epoch)
    return scheduler


def featuremap_perspective_transform(featuremap: torch.Tensor, bpts: torch.Tensor, btarget_pts: torch.Tensor,
                                     output_size):
    """对一个batch的featuremap做投影变换

    Arguments:
        featuremap {torch.Tensor} -- [B, C, H, W]
        pts {torch.Tensor} -- [B, 4, 2] xy格式
        target_pts {torch.Tensor} -- [B, 4, 2] xy格式
        output_shape {torch.Tensor} -- [2] w, h
    """
    device = featuremap.device
    B, C, H, W = featuremap.shape
    w, h = output_size

    # 求解投影矩阵
    bpts_np = bpts.cpu().float().numpy()
    btarget_pts_np = btarget_pts.cpu().float().numpy()

    trans_mats = []

    for pts_np, target_pts_np in zip(bpts_np, btarget_pts_np):
        trans_mat = cv2.getPerspectiveTransform(pts_np, target_pts_np)
        if np.linalg.matrix_rank(trans_mat) < 3:
            trans_mat = np.identity(3, dtype=np.float)
        trans_mats.append(torch.from_numpy(trans_mat))
    inv_trans_mats = torch.stack(trans_mats).float().inverse().to(device)

    # 坐标反变换
    x, y = torch.meshgrid(torch.arange(h), torch.arange(w))
    z = torch.ones_like(x)
    cors = torch.stack([x, y, z]).view(1, 3, -1).to(device).float()
    cors = cors.repeat(B, 1, 1)

    reversed_cors = torch.bmm(inv_trans_mats, cors)
    reversed_cors = reversed_cors[:, :2, :] / \
                    reversed_cors[:, 2, :].view(B, 1, -1)  # [B, 2, wh]
    reversed_cors = reversed_cors.view(-1, 2, h, w).permute(0, 2, 3, 1)
    norm_cors = ((reversed_cors / reversed_cors.new_tensor([W, H])) - 0.5) * 2

    # 插值结果
    output = torch.nn.functional.grid_sample(featuremap, norm_cors, padding_mode='border')
    assert not torch.any(torch.isnan(output)), "Found NaN"
    tmp = output + 1
    return output


