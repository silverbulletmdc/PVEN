import torch
import glob
import re
import os.path as osp
from PIL import Image
import cv2
import numpy as np
import socket

def read_any_img(img_path: str, format='ndarray'):
    """
    单通道图返回原图，多通道图返回RGB。支持文件系统

    :param img_path:
    :param format:
    :return:
    """
    img = read_rgb_image(img_path, format)
    return img


def read_rgb_image(img_path, format='ndarray'):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            if format == 'PIL':
                img = Image.open(img_path).convert("RGB")
            elif format == 'ndarray':
                img = cv2.imread(img_path)
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
    return img



def load_checkpoint(output_dir, device="cpu", epoch=0, exclude=None, **kwargs):
    """
    含关键字model和optimizer会被正确加载到指定的设备上.
    如果不指定epoch，自动读取最大的epoch。
    如果指定exclude,将会删除含该关键字的参数

    :param str output_dir:
    :param device:
    :param epoch:
    :param kwargs:
    :return:
    """

    if output_dir[-1] == '/':
        output_dir = output_dir[:-1]

    # 不指定epoch则读取已保存的最大epoch
    if epoch == 0:
        for key in kwargs.keys():
            pths = glob.glob(f'{output_dir}/{key}_*.pth')
            epochs = [re.findall(rf'{output_dir}/{key}_([0-9]+)\.pth', name)[0] for name in pths]
            epochs = list(map(int, epochs))
            epoch = max(epochs)
            break

    for key, obj in kwargs.items():
        state_dict = torch.load(f'{output_dir}/{key}_{epoch}.pth', map_location=device)
        if exclude is not None:
            exclude_keys = []
            for k in state_dict.keys():
                if exclude in k:
                    exclude_keys.append(k)
            for k in exclude_keys:
                del state_dict[k]

        obj: torch.nn.Module
        try:
            obj.load_state_dict(state_dict, strict=False)
        except TypeError:
            obj.load_state_dict(state_dict)


        # move to target device
        if 'model' in key:
            obj.to(device)

        elif 'optimizer' in key:
            for state in obj.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

    return epoch + 1


def save_checkpoint(epoch, output_dir, **kwargs):
    for key, obj in kwargs.items():
        try:
            obj = obj.module
        except AttributeError:
            pass

        torch.save(obj.state_dict(), f'{output_dir}/{key}_{epoch}.pth')


def merge_configs(cfg, config_files, cmd_config):
    """
    融合不同的配置。依次加载默认配置，配置文件和命令行参数。配置文件用,隔开

    :param CfgNode cfg:
    :param str config_files:
    :param list cmd_config:
    :return:
    """
    if config_files != "":
        config_files = config_files.split(",")
        for config_file in config_files:
            cfg.merge_from_file(config_file)

    cfg.merge_from_list(cmd_config)
    return cfg


def get_host_ip():
    """
    查询本机ip地址
    :return: ip
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip
