import copy
import random
import torch
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        super(RandomIdentitySampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, item in enumerate(self.data_source):
            pid = item['id']
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


class KPSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """

    def __init__(self, data_source, batch_size, num_instances):
        super(KPSampler, self).__init__(data_source)
        self.data_source = data_source
        self.num_instances = num_instances
        self.batch_size = batch_size
        self.index_dic = defaultdict(list)
        for index, item in enumerate(data_source):
            id = item["id"]
            self.index_dic[id].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
            print(ret)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances


class SimilarIdentitySampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances, similarity_matrix):
        """

        :param list data_source: (path, pid, image_id)
        :param num_instances:
        :param np.ndarray similarity_matrix: 相似度矩阵。(i, j)代表两个id之间的相似度。
        """

        super(SimilarIdentitySampler, self).__init__(data_source)
        self.data_source = data_source
        self.num_instances = num_instances
        num_ids = similarity_matrix.shape[0]
        similarity_matrix[np.eye(num_ids, dtype=bool)] = 0
        self.similarity_matrix = similarity_matrix

        self.index_dic = defaultdict(list)
        self.batch_size = batch_size
        self.num_pids_per_batch = self.batch_size // self.num_instances
        for i, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(i)
        self.pids = list(self.index_dic.keys())
        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pid = random.sample(avai_pids, 1)[0]
            similarity = self.similarity_matrix[selected_pid, avai_pids]
            similarity /= np.sum(similarity)

            ################################################################################
            # Sample only from top-k similarity.
            order = np.argsort(similarity)
            p = np.zeros_like(similarity)
            p[order[-(self.num_pids_per_batch - 1):]] = 1 / (self.num_pids_per_batch - 1)
            ################################################################################

            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch - 1, False, p=p)
            selected_pids = list(selected_pids)
            selected_pids.insert(0, selected_pid)
            assert len(selected_pids) == self.num_pids_per_batch
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                assert len(batch_idxs) == self.num_instances
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


def test_similarity_sampler():
    """
    只能保证运行时不出错,不验证正确性.

    :return:
    """
    pids = list(range(100))
    path = "aaa"
    data_source = [(path, pid, idx) for idx, pid in enumerate(np.random.choice(pids, 10000))]
    batch_size = 64
    num_instances = 16
    similarity_matrix = np.random.rand(100, 100)
    sampler = SimilarIdentitySampler(data_source, batch_size, num_instances, similarity_matrix)
    random_sampler = RandomIdentitySampler(data_source, batch_size, num_instances)
    print("bbb")
    print(len(sampler))
    print(len(random_sampler))
    print("aaa")
    for idx1, idx2 in zip(sampler, random_sampler):
        print(data_source[idx1])
