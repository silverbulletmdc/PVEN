# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import time


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, remove_junk=True):
    """Evaluation with veri776 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.

    :param np.ndarray distmat:
    :param np.ndarray q_pids:
    :param np.ndarray g_pids:
    :param np.ndarray q_camids:
    :param np.ndarray g_camids:
    :param int max_rank:
    :param bool remove_junk:
    :return:
    """
    # compute cmc curve for each query
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in tqdm(range(num_q), desc='Calc cmc and mAP'):
        # get query pid and camid
        q_pid = q_pids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = np.argsort(distmat[q_idx])
        if remove_junk:
            q_camid = q_camids[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        else:
            remove = np.zeros_like(g_pids).astype(np.bool)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
    #     orig_cmc = matches[q_idx][keep]
        orig_cmc = (g_pids[order] == q_pid).astype(np.int32)[keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def eval_func_mp(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, remove_junk=True):
    """
    Multiprocess version for eval func
    """

    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    all_cmc = []
    all_AP = []
    print('Generating worker pools')
    t1 = time.time()
    pool = Pool(30)
    res = pool.imap(worker, [
        (
            q_pids[q_idx],
            q_camids[q_idx],
            g_pids,
            g_camids,
            distmat[q_idx],
            max_rank,
            remove_junk
        ) for q_idx in range(num_q)
    ], chunksize=32)
    print(time.time() - t1)

    for r in tqdm(res, total=num_q):
        all_AP.append(r[0])
        all_cmc.append(r[1])


    # num_valid_q = 0.  # number of valid query
    # for q_idx in tqdm(range(num_q), desc='Calc cmc and mAP'):
        # get query pid and camid
    # assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP, all_AP

def worker(args):
    q_pid, q_camid, g_pids, g_camids, dist_vec, max_rank, remove_junk = args
    # remove gallery samples that have the same pid and camid with query
    order = np.argsort(dist_vec)
    if remove_junk:
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
    else:
        remove = np.zeros_like(g_pids).astype(np.bool)
    keep = np.invert(remove)

    # compute cmc curve
    # binary vector, positions with value 1 are correct matches
#     orig_cmc = matches[q_idx][keep]
    orig_cmc = (g_pids[order] == q_pid).astype(np.int32)[keep]
    AP, cmc = calc_AP(orig_cmc)
    return AP, cmc[:max_rank]

def eval_func_th(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, remove_junk=True):
    """Evaluation with veri776 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.

    :param np.ndarray distmat:
    :param np.ndarray q_pids:
    :param np.ndarray g_pids:
    :param np.ndarray q_camids:
    :param np.ndarray g_camids:
    :param int max_rank:
    :param bool remove_junk:
    :return:
    """
    # compute cmc curve for each query
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in tqdm(range(num_q), desc='Calc cmc and mAP'):
        # get query pid and camid
        q_pid = q_pids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = np.argsort(distmat[q_idx])
        if remove_junk:
            q_camid = q_camids[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        else:
            remove = np.zeros_like(g_pids).astype(np.bool)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
    #     orig_cmc = matches[q_idx][keep]
        orig_cmc = (g_pids[order] == q_pid).astype(np.int32)[keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def calc_AP(orig_cmc):
    """Evaluation

    计算一行的AP值
    """
    # orig_cmc = (g_pids[order] == q_pid).astype(np.int32)[keep]
    if not np.any(orig_cmc):
        # this condition is true when query identity does not appear in gallery
        raise ValueError

    cmc = orig_cmc.cumsum()
    cmc[cmc > 1] = 1

    # compute average precision
    # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
    num_rel = orig_cmc.sum()
    tmp_cmc = orig_cmc.cumsum()
    tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]  # Precision
    tmp_cmc = np.asarray(tmp_cmc) * orig_cmc  # on Recall changed
    AP = tmp_cmc.sum() / num_rel
    return AP, cmc

def get_expectation_of_AP(N=10, T=3):
    """
    蒙特卡洛方法获得AP期望值。

    N: 样本总数
    T: 正例数量
    """
    for i in range(1000):
        idxs = np.random.choice(np.arange(N), T)
        cmc = np.zeros(N)
        cmc[idxs] = 1
        AP = calc_AP(cmc)[0]
        APs.append(AP)
    print(np.mean(APs))