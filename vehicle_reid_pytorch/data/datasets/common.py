# encoding: utf-8
"""
@author:  Dechao Meng
@contact: dechao.meng@vipl.ict.ac.cn
"""

import glob
import re
import os.path as osp
from pathlib import Path
import time
import numpy as np
import pickle as pkl
from vehicle_reid_pytorch.data.datasets.bases import ReIDMetaDataset, relabel, get_imagedata_info


class CommonReIDDataset(ReIDMetaDataset):
    def __init__(self, pkl_path, verbose=True, test_ext='', **kwargs):
        """
        test_ext: For VehicleID and VERIWild, there are multi test sets. Pass the test ext to select which one to use.
        """
        metas = pkl.load(open(pkl_path, 'rb'))
        self.train = metas["train"]
        self.query = metas["query" + str(test_ext)]
        self.gallery = metas["gallery" + str(test_ext)]

        self.relabel()

        if verbose:
            print("=> Dataset loaded")
            self.print_dataset_statistics()

        self.num_train_ids, self.num_train_imgs, self.num_train_cams = get_imagedata_info(self.train)
        self.num_query_ids, self.num_query_imgs, self.num_query_cams = get_imagedata_info(self.query)
        self.num_gallery_ids, self.num_gallery_imgs, self.num_gallery_cams = get_imagedata_info(self.gallery)
