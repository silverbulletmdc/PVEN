# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import pickle
import os
import numpy as np
import torch
import torch.nn.functional

from .eval_reid import eval_func
from .rerank import re_ranking


def build_metric(cfg, num_query):
    if 'vehicleid' in cfg.datasets.names[0]:
        metric = CMC10Times(feat_norm=cfg.test.FEAT_NORM, output_path=cfg.OUTPUT_DIR,
                            rerank=cfg.test.RE_RANKING == 'yes')
    else:
        metric = R1_mAP(num_query, max_rank=50, feat_norm=cfg.test.FEAT_NORM, output_path=cfg.OUTPUT_DIR,
                        rerank=cfg.test.RE_RANKING == 'yes', remove_junk=cfg.test.REMOVE_JUNK == 'yes')

    return metric


class R1_mAP:
    def __init__(self, num_query, *, max_rank=0, feat_norm=True, rerank=False, remove_junk=True, output_path=''):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.output_path = output_path
        self.rerank = rerank
        self.remove_junk = remove_junk
        self.reset()

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.paths = []

    def update(self, output):
        feat, pid, camid, paths = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.paths += paths

    def process_feat(self):
        self.feats = torch.cat(self.feats, dim=0)
        self.pids = np.asarray(self.pids)
        if self.feat_norm:
            self.feats = torch.nn.functional.normalize(self.feats, dim=1, p=2)


    def resplit(self, pids):
        # sorted_idxs = np.argsort(pids)
        # sorted_pid = pids[sorted_idxs]
        num_pid = len(set(pids))
        query = []
        gallery = []
        for i in range(num_pid):
            idxs:np.ndarray = (pids==i).nonzero()[0]
            choose_idx = np.random.randint(len(idxs))
            gallery.append(idxs[choose_idx])
            query.extend(idxs[:choose_idx])
            query.extend(idxs[choose_idx+1:])
        return query + gallery

    def shuffle_eval(self):
        """打乱后再计算，用于VehicleID 
        """
        indexs = self.resplit(self.pids)
        self.feats = self.feats[indexs]
        self.pids = self.pids[indexs]

        return self._compute()

    def compute(self):
        self.process_feat()
        return self._compute()

    def _compute(self):
        feats = self.feats
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        m, n = qf.shape[0], gf.shape[0]

        if self.rerank:
            distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)

        else:
            distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            distmat.addmm_(1, -2, qf, gf.t())
            distmat = torch.sqrt(distmat).cpu().numpy()

        # 保存结果
        query_paths = self.paths[:self.num_query]
        gallery_paths = self.paths[self.num_query:]

        self.write_output(gallery_paths, query_paths, g_pids, q_pids, qf, gf, q_camids, g_camids, distmat)
        self.distmat = distmat
        
        cmc, mAP = eval_func(distmat, q_pids, g_pids,
                             q_camids, g_camids, remove_junk=self.remove_junk)

        return cmc, mAP
    
    
    def write_output(self, gallery_paths, query_paths, g_pids, q_pids, qf, gf, q_camids, g_camids, distmat):
        if self.output_path != '':
        #     gallery_indexs = np.argsort(distmat, axis=1)[:100] + 1
        #     def int2str(id):
        #         return f'{id:06d}'
                
        #     with open('submit.txt', 'w') as f:
        #         for gallerys in gallery_indexs:
        #             output_str = ' '.join(map(int2str, gallerys)) + '\n'
        #             f.write(output_str)
                
            try:
                with open(os.path.join(self.output_path, 'test_output.pkl'), 'wb') as f:
                    torch.save({
                        'gallery_paths': gallery_paths,
                        'query_paths': query_paths,
                        'gallery_ids': g_pids,
                        'query_ids': q_pids,
                        'query_features': qf,
                        'gallery_features': gf,
                        'query_cams': q_camids,
                        'gallery_cams': g_camids,
                        'distmat': distmat
                    }, f)

            except OverflowError:
                print("Can't save results.")
                pass


class CMC10Times:

    def __init__(self, feat_norm='yes', output_path='', rerank=False):
        """
        VehicleID的评测算法。重复十次。每次各id随机取一张放入gallery中。

        :param num_query:
        :param max_rank:
        :param feat_norm:
        :param output_path:
        :param remove_junk:
        """
        super(CMC10Times, self).__init__()
        self.feat_norm = feat_norm
        self.output_path = output_path
        self.rerank = rerank
        self.reset()

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.paths = []

    def update(self, output):
        feat = output[0]
        pid = output[1]
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))

    def compute(self):
        self.feats = torch.cat(self.feats, dim=0)

        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            self.feats = torch.nn.functional.normalize(self.feats, dim=1, p=2)

        pids_np = np.array(self.pids)
        cmcs = []
        mAPs = []
        for i in range(10):
            # 采样
            gallery = []
            pid_set = set(self.pids)
            for pid in pid_set:
                mask = (pids_np == pid)
                idxs = np.nonzero(mask)[0]
                sample_idx = np.random.choice(idxs)
                gallery.append(sample_idx)

            # 计算
            cmc, mAP = self.compute_once(gallery)
            cmcs.append(cmc)
            mAPs.append(mAP)
        # 求均值
        cmcs = np.array(cmcs)
        mean_cmc = cmcs.mean(axis=0)
        mAPs = np.array(mAPs)
        mean_mAP = mAPs.mean()
        return mean_cmc, mean_mAP

    def compute_once(self, gallery_idxs):
        gallery_mask = torch.zeros(len(self.feats))
        gallery_mask[gallery_idxs] = 1
        query_mask = 1 - gallery_mask

        # query
        qf = self.feats[query_mask.type(torch.bool)]
        q_pids = np.asarray(self.pids)[query_mask.type(torch.bool)]
        # gallery
        gf = self.feats[gallery_mask.type(torch.bool)]
        g_pids = np.asarray(self.pids)[gallery_mask.type(torch.bool)]

        if self.rerank:
            distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        else:
            m, n = qf.shape[0], gf.shape[0]
            distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            distmat.addmm_(1, -2, qf, gf.t())
            distmat = distmat.cpu().numpy()
        cmc, mAP = eval_func(distmat, q_pids, g_pids,
                             None, None, remove_junk=False)

        return cmc, mAP
