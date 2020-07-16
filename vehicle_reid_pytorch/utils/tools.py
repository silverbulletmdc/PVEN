import logging
import sys
import os
import torch
import time
from logzero import logger
from yacs.config import CfgNode


def setup_logger(name, save_dir, distributed_rank, level="INFO"):
    logger = logging.getLogger(name)
    logger.setLevel(10)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(getattr(logging, level.upper()))
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt"), mode='w')
        fh.setLevel(getattr(logging, level.upper()))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def tb_log(kv_map, writter, global_steps):
    """
    接受一个字典，将里面的值发送到tensorboard中
    :param dict losses:
    :param writter:
    :param global_steps:
    :return:
    """
    for loss_name, value in kv_map.items():
        if isinstance(value, torch.Tensor):
            value = value.item()
        writter.add_scalar(loss_name, value, global_steps)
        logger.debug(f'{loss_name}: {value}')


class Session():
    def __init__(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass


class TimeCounter():
    """
    统计程序运行时间。支持使用with语句。

    """

    def __init__(self, verbose=False):
        self._verbose = verbose
        self.period = 0

    def __enter__(self):
        self._start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end = time.time()
        self.period += self._end - self._start

        if self._verbose:
            print(f'Cost time: {self.period}')


def iter_x(x):
    if isinstance(x, CfgNode):
        for key, value in x.items():
            yield (key, value)



def _flat_cfg(x):
    for key, value in iter_x(x):
        if isinstance(value, (dict, list)):
            for k, v in _flat_cfg(value):
                k = f'{key}.{k}'
                yield (k, v)
        else:
            yield (key, value)

def flat_cfg(x):
    output = {}
    for k, v in _flat_cfg(x):
        output[k] = v
    return output
