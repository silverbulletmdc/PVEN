import torch
from torch import nn


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y, split=0):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
      split: When the CUDA memory is not sufficient, we can split the dataset into different parts
             for the computing of distance.
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    if split == 0:
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        distmat = xx + yy
        distmat.addmm_(x, y.t(), beta=1, alpha=-2)

    else:
        distmat = x.new(m, n)
        start = 0
        x = x.cuda()

        while start < n:
            end = start + split if (start + split) < n else n
            num = end - start

            sub_distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, num) + \
                    torch.pow(y[start:end].cuda(), 2).sum(dim=1, keepdim=True).expand(num, m).t()
            # sub_distmat.addmm_(1, -2, x, y[start:end].t())
            sub_distmat.addmm_(x, y[start:end].cuda().t(), beta=1, alpha=-2)
            distmat[:, start:end] = sub_distmat.cpu()
            start += num

    distmat = distmat.clamp(min=1e-12).sqrt()  # for numerical stability
    return distmat


def hard_example_mining(dist_mat, labels, mask=None, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      mask: pytorch Tensor, with shape [N, N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    if mask is None:
        mask = torch.ones_like(dist_mat)

    aux_mat = torch.zeros_like(dist_mat)
    aux_mat[mask==0] -= 10
    dist_mat = dist_mat + aux_mat

    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)



    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    # dist_mat[dist_mat == 0] += 10000  # 处理非法值。归一化后的最大距离为2
    aux_mat = torch.zeros_like(dist_mat)
    aux_mat[mask==0] += 10000
    dist_mat = dist_mat + aux_mat
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]



    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, mask=None, normalize_feature=False):
        """

        :param global_feat:
        :param labels:
        :param mask:  [N, N] 可见性mask。不可见的mask将不会被选择。若全部不可见，则对结果*0
        :param normalize_feature:
        :return:
        """
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels, mask=mask)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, keep_dim=False):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.keep_dim = keep_dim

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = inputs.new_zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data, 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.keep_dim:
            loss = (- targets * log_probs).sum(1)
        else:
            loss = (- targets * log_probs).mean(0).sum()
        return loss
