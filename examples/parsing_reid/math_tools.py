import os

import torch
from torch.nn import functional as F
import numpy as np

from vehicle_reid_pytorch.metrics import eval_func
from vehicle_reid_pytorch.loss.triplet_loss import normalize, euclidean_dist
from functools import reduce

from vehicle_reid_pytorch.metrics.rerank import re_ranking


def clck_dist(feat1, feat2, vis_score1, vis_score2, split=0):
    """
    计算vpm论文中的clck距离

    :param torch.Tensor feat1: [B1, C, 3]
    :param torch.Tensor feat2: [B2, C, 3]
    :param torch.Tensor vis_score: [B, 3]
    :rtype torch.Tensor
    :return: clck distance. [B1, B2]
    """

    B, C, N = feat1.shape
    dist_mat = 0
    ckcl = 0
    for i in range(N):
        parse_feat1 = feat1[:, :, i]
        parse_feat2 = feat2[:, :, i]
        ckcl_ = torch.mm(vis_score1[:, i].view(-1, 1), vis_score2[:, i].view(1, -1))  # [N, N]
        ckcl += ckcl_
        dist_mat += euclidean_dist(parse_feat1, parse_feat2, split=split) * ckcl_

    return dist_mat / ckcl


class Clck_R1_mAP:
    def __init__(self, num_query, *, max_rank=50, feat_norm=True, output_path='', rerank=False, remove_junk=True,
                 lambda_=0.5):
        """
        计算VPM中的可见性距离并计算性能

        :param num_query:
        :param max_rank:
        :param feat_norm:
        :param output_path:
        :param rerank:
        :param remove_junk:
        :param lambda_: distmat = global_dist + lambda_ * local_dist, default 0.5
        """
        super(Clck_R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.output_path = output_path
        self.rerank = rerank
        self.remove_junk = remove_junk
        self.lambda_ = lambda_
        self.reset()

    def reset(self):
        self.global_feats = []
        self.local_feats = []
        self.vis_scores = []
        self.pids = []
        self.camids = []
        self.paths = []

    def update(self, output):
        global_feat, local_feat, vis_score, pid, camid, paths = output
        self.global_feats.append(global_feat)
        self.local_feats.append(local_feat)
        self.vis_scores.append(vis_score)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.paths += paths

    def compute(self, split=0):
        """
        split: When the CUDA memory is not sufficient, we can split the dataset into different parts
               for the computing of distance.
        """
        global_feats = torch.cat(self.global_feats, dim=0)
        local_feats = torch.cat(self.local_feats, dim=0)
        vis_scores = torch.cat(self.vis_scores).cpu()
        if self.feat_norm:
            print("The test feature is normalized")
            global_feats = F.normalize(global_feats, dim=1, p=2)
            local_feats = F.normalize(local_feats, dim=1, p=2)
        # 全局距离
        print('Calculate distance matrixs...')
        # query
        qf = global_feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = global_feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        qf = qf.cuda()
        m, n = qf.shape[0], gf.shape[0]


        if self.output_path:
            print('Saving results...')
            outputs = {
                "global_feats": global_feats,
                "vis_scores": vis_scores,
                "local_feats": local_feats,
                "pids": self.pids,
                "camids": self.camids,
                "paths": self.paths,
                "num_query": self.num_query
            }
            torch.save(outputs, os.path.join(self.output_path, 'test_output.pkl'), pickle_protocol=4)

        if self.rerank:
            distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)

        else:
            # qf: M, F
            # gf: N, F
            if split == 0:
                distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                        torch.pow(gf.cuda(), 2).sum(dim=1, keepdim=True).expand(n, m).t()
                distmat.addmm_(1, -2, qf, gf.t())
            else:
                distmat = gf.new(m, n)
                start = 0
                while start < n:
                    end = start + split if (start + split) < n else n
                    num = end - start

                    sub_distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, num) + \
                            torch.pow(gf[start:end].cuda(), 2).sum(dim=1, keepdim=True).expand(num, m).t()
                    # sub_distmat.addmm_(1, -2, qf, gf[start:end].t())
                    sub_distmat.addmm_(qf, gf[start:end].cuda().t(), beta=1, alpha=-2)
                    distmat[:, start:end] = sub_distmat.cpu()

                    start += num

            distmat = distmat.numpy()

        # 局部距离
        print('Calculate local distances...')
        local_distmat = clck_dist(local_feats[:self.num_query], local_feats[self.num_query:],
                                  vis_scores[:self.num_query], vis_scores[self.num_query:], split=split)

        local_feats = local_feats.cpu()
        local_distmat = local_distmat.cpu().numpy()

        # 保存结果
        # query_paths = self.paths[:self.num_query]
        # gallery_paths = self.paths[self.num_query:]
        # if self.output_path != '':
        #     with open(os.path.join(self.output_path, 'test_output.pkl'), 'wb') as f:
        #         torch.save({
        #             'gallery_paths': gallery_paths,
        #             'query_paths': query_paths,
        #             'gallery_ids': g_pids,
        #             'query_ids': q_pids,
        #             # 'query_features': qf,
        #             # 'gallery_features': gf,
        #             'query_cams': q_camids,
        #             'gallery_cams': g_camids,
        #             'distmat': distmat,
        #             'localdistmat': local_distmat
        #         }, f)

        print('Eval...')
        cmc, mAP = eval_func(distmat + self.lambda_ * local_distmat, q_pids, g_pids, q_camids, g_camids,
                             remove_junk=self.remove_junk)

        return cmc, mAP
