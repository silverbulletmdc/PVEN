import torch
from torch import nn
from torch import functional as F
from vehicle_reid_pytorch.utils.math import euclidean_dist


def generate_tuplets(K, P):
    """
    生成tuplets

    :param torch.Tensor label: [B]
    :return:
    """
    tuplets = []
    for k in range(K):
        for p in range(P):
            index = k + p * K
            positives = torch.arange(K * p, K * (p + 1))
            positives = positives[positives != index]
            negative_labels = torch.arange(P)
            negative_labels = negative_labels[negative_labels != p]
            for positive in positives:
                negatives = torch.randint(K, (P - 1,)) + (negative_labels * K)
                tuplet = torch.tensor([index, positive, *negatives])
                tuplets.append(tuplet)
    tuplets = torch.stack(tuplets)
    return tuplets


def _tuplet_loss(tuplet_feats, s, beta):
    """

    :param tuplet_feats: [B, P+1, C]
    :param s:
    :return:
    """
    B, P, C = tuplet_feats.shape
    P = P - 1
    anchors = tuplet_feats[:, 0]  # B C
    positives = tuplet_feats[:, 1]  # B C
    negatives = tuplet_feats[:, 2:]  # B P-1 C
    cos_ap = torch.sum(anchors * positives, 1)  # B
    if beta != 0:
        theta_ap = torch.acos(cos_ap)
    cos_ap_beta = torch.cos(theta_ap - beta)
    cos_an = torch.sum(anchors.view(B, 1, -1) * negatives, 2)  # B, P-1
    return torch.log(1 + torch.sum(torch.exp(s * (cos_an - cos_ap_beta.view(-1, 1))), 1))


class TupletLoss(object):
    """
    An reproduce of the margin tuplet loss proposed by
    "Deep Metric Learning with Tuplet Margin Loss", ICCV2019

    """

    def __init__(self, K, P, s=32, beta=0.):
        """


        :param K: number of images per classes in a minibatch
        :param P: numebr of classes in a minibatch
        :param s: scale factor
        :param beta: slack margin
        """
        self.K = K
        self.P = P
        self.s = s
        self.beta = beta

    def __call__(self, feats):
        """

        :param torch.Tensor feats: [N, C]
        :return:
        """
        feats = feats / F.norm(feats, dim=1).view(-1, 1)
        tuplets = generate_tuplets(self.K, self.P).to(feats.device)
        tuplet_feats = feats[tuplets]

        loss = _tuplet_loss(tuplet_feats, self.s, self.beta).mean()
        return loss
