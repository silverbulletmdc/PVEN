from vehicle_reid_pytorch.loss.triplet_loss import normalize, euclidean_dist, hard_example_mining
from vehicle_reid_pytorch.models import Baseline
import torch
from torch import nn
import torch.nn.functional as F
from functools import reduce
from math_tools import clck_dist
from pprint import pprint


class ParsingReidModel(Baseline):

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, num_local_branches=4):
        super(ParsingReidModel, self).__init__(num_classes, last_stride, model_path, neck, neck_feat, model_name,
                                               pretrain_choice)

        # self.local_bn_neck = nn.BatchNorm1d(2048*num_local_branches)
        self.local_bn_neck = nn.BatchNorm1d(2048*num_local_branches)
        self.local_classifier = nn.Conv1d(2048, num_classes, 1)

    def forward(self, image, mask=None, **kwargs):
        """

        :param torch.Tensor x: [B, 3, H, W]
        :param torch.Tensor mask: [B, N, H, W] front/back, side, window
        :return:
        """
        # Remove bg

        if mask is not None:
            mask = mask[:, 1:, :, :]
            B, N, H, W = mask.shape
        else:
            B, _, H, W = image.shape
            N = 4
            mask = image.new_zeros(B, 4, H, W)

        x = self.base(image)

        B, C, h, w = x.shape
        mask = F.interpolate(mask, x.shape[2:])
        # mask = F.softmax(mask, dim=1)
        # mask = F.adaptive_max_pool2d(mask, output_size=x.shape[2:]).view(B, N, h, w)

        global_feat = self.gap(x)  # (b, 2048, 1, 1)

        global_feat = global_feat.view(
            global_feat.shape[0], -1)  # flatten to (bs, 2048)

        vis_score = mask.sum(dim=[2, 3]) + 1  # Laplace平滑
        local_feat_map = torch.mul(mask.unsqueeze(
            dim=2), x.unsqueeze(dim=1))  # (B, N, C, h, w)
        local_feat_map = local_feat_map.view(B, -1, h, w)
        local_feat_before = F.adaptive_avg_pool2d(local_feat_map, output_size=(1, 1)).view(B, N, C).permute(
            [0, 2, 1]) * (h * w / vis_score.unsqueeze(dim=1))  # (B, C, N)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            # normalize for angular softmax
            feat = self.bottleneck(global_feat)
            local_feat = self.local_bn_neck(
                local_feat_before.contiguous().view(B, -1)).view(B, -1, N)  # 这一步会使其不为0

        if self.training:
            cls_score = self.classifier(feat)
            local_cls_score = self.local_classifier(local_feat)
            # global feature for triplet loss
            return {"cls_score": cls_score,
                    "global_feat": global_feat,
                    "local_cls_score": local_cls_score,
                    "local_feat": local_feat,
                    "vis_score": vis_score}

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return {"global_feat": feat,
                        "local_feat": local_feat,
                        "vis_score": vis_score}
            else:
                # print("Test with feature before BN")
                return {"global_feat": global_feat, 
                         "local_feat": local_feat_before, 
                         "vis_score": vis_score}


class ParsingTripletLoss:
    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, local_feat, vis_score, target, normalize_feature=False):
        """

        :param torch.Tensor local_feature: (B, C, N)
        :param torch.Tensor visibility_score: (B, N)
        :param torch.Tensor target: (B)
        :return:
        """
        B, C, _ = local_feat.shape
        if normalize_feature:
            local_feat = normalize(local_feat, 1)

        dist_mat = clck_dist(local_feat, local_feat,
                             vis_score, vis_score)

        dist_ap, dist_an = hard_example_mining(dist_mat, target)
        y = dist_an.new().resize_as_(dist_an).fill_(1)

        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)

        return loss, dist_ap, dist_an


def build_model(cfg, num_classes):
    # if cfg.MODEL.NAME == 'resnet50':
    #     model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    model = ParsingReidModel(num_classes, cfg.model.last_stride, cfg.model.pretrain_model, cfg.model.neck,
                             cfg.test.neck_feat, cfg.model.name, cfg.model.pretrain_choice)
    return model


if __name__ == '__main__':
    from tensorboardX import SummaryWriter

    dummy_input = torch.rand(4, 3, 224, 224)
    model = Baseline(576, 1, '/home/mengdechao/.cache/torch/checkpoints/resnet50-19c8e357.pth', 'bnneck', 'after',
                     'resnet50', 'imagenet')
    model.train()
    with SummaryWriter(comment="baseline") as w:
        w.add_graph(model, [dummy_input, ])
