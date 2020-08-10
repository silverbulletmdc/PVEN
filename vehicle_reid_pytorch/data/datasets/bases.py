import numpy as np
import cv2
import pickle as pkl
from torch.utils.data import Dataset
from vehicle_reid_pytorch.utils.iotools import read_rgb_image
import scipy.stats as st


def get_imagedata_info(data):
    ids, cams = [], []
    for item in data:
        ids.append(item["id"])
        cams.append(item["cam"])

    ids = [item["id"] for item in data]
    cams = [item["cam"] for item in data]
    pids = set(ids)
    cams = set(cams)
    num_pids = len(pids)
    num_cams = len(cams)
    num_imgs = len(data)
    return num_pids, num_imgs, num_cams


def relabel(data):
    """
    :param list data:
    :return:
    """
    raw_ids = set()
    data = data.copy()
    for item in data:
        raw_ids.add(item['id'])
    raw_ids = sorted(list(raw_ids))
    rawid2label = {raw_vid: i for i, raw_vid in enumerate(raw_ids)}
    label2rawid = {i: raw_vid for i, raw_vid in enumerate(raw_ids)}
    for item in data:
        item["id"] = rawid2label[item["id"]]
        item["cam"] = int(item["cam"])
    return data, rawid2label, label2rawid


class ReIDMetaDataset:
    """
    定义了ReID数据集的元信息。必须包含train, query, gallery属性。
    A list of dict. Dict contains meta infomation, which is
    {
        "image_path": str, required
        "id": int, required

        "cam"(optional): int,
        "keypoints"(optional): extra information
        "kp_vis"(optional): 每个keypoint是否可见
        "mask"(optional): extra information
        "box"(optional): extra information
        "color"(optional): extra information
        "type"(optional): extra information
        "view"(optional): extra information
    }
    """
    def __init__(self, pkl_path, verbose=True, **kwargs):
        with open(pkl_path, 'rb') as f:
            metas = pkl.load(f)

        self.train = metas["train"]
        self.query = metas["query"]
        self.gallery = metas["gallery"]
        self.relabel()
        self._calc_meta_info()

        if verbose:
            print("=> Dataset loaded")
            self.print_dataset_statistics()

    def relabel(self):
        self.train, self.train_rawid2label, self.train_label2rawid = relabel(self.train)
        eval_set, self.eval_rawid2label, self.eval_label2rawid = relabel(self.query + self.gallery)
        self.query = eval_set[:len(self.query)]
        self.gallery = eval_set[len(self.query):]

    def print_dataset_statistics(self):
        num_train_pids, num_train_imgs, num_train_cams = get_imagedata_info(self.train)
        num_query_pids, num_query_imgs, num_query_cams = get_imagedata_info(self.query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = get_imagedata_info(self.gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")

    def _calc_meta_info(self):
        self.num_train_ids, self.num_train_imgs, self.num_train_cams = get_imagedata_info(self.train)
        self.num_query_ids, self.num_query_imgs, self.num_query_cams = get_imagedata_info(self.query)
        self.num_gallery_ids, self.num_gallery_imgs, self.num_gallery_cams = get_imagedata_info(self.gallery)


class ReIDDataset(Dataset):
    def __init__(self, meta_dataset, *, with_mask=False, mask_num=5, transform=None, preprocessing=None):
        """将元数据集转化为图片数据集，并进行预处理

        Arguments:
            Dataset {ReIDMetaDataset} -- self
            meta_dataset {ReIDMetaDataset} -- 元数据集

        Keyword Arguments:
            with_box {bool} -- [是否使用检测框做crop。从box属性中读取检测框信息] (default: {False})
            with_mask {bool} -- [是否读取mask。为True时从mask_nori_id读取mask] (default: {False})
            mask_num {int} -- [mask数量] (default: {5})
            sub_bg {bool} -- [是否删除背景。with_mask为True时才会生效。将利用第一个mask对图片做背景减除] (default: {False})
            transform {[type]} -- [数据增强] (default: {None})
            preprocessing {[type]} -- [normalize, to tensor等预处理] (default: {None})
        """
        self.meta_dataset = meta_dataset
        self.transform = transform
        self.preprocessing = preprocessing
        self.with_mask = with_mask
        self.mask_num = mask_num

    def read_mask(self, sample):
        # 读入mask
        mask = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
        mask = [mask == v for v in range(self.mask_num)]
        mask = np.stack(mask, axis=-1).astype('float32')
        sample["mask"] = mask


    def __getitem__(self, item):
        meta: dict = self.meta_dataset[item]
        sample = meta.copy()
        # 读入图片
        sample["image"] = read_rgb_image(meta["image_path"])

        # 读入mask
        if self.with_mask:
            self.read_mask(sample)

        # 数据增强
        if self.transform:
            sample = self.transform(**sample)

        # preprocessing
        if self.preprocessing:
            sample = self.preprocessing(**sample)
        
        return sample

    def __len__(self):
        return len(self.meta_dataset)


if __name__ == "__main__":
    from vehicle_reid_pytorch.data.datasets import AICity
    from vehicle_reid_pytorch.utils.visualize import visualize_img
    from vehicle_reid_pytorch.data.demo_transforms import get_training_albumentations 
    import matplotlib.pyplot as plt
    meta_dataset = ReIDMetaDataset(pkl_path="")
    dataset = ReIDDataset(meta_dataset.train, transform=get_training_albumentations(with_keypoints=True))
    images = []
    for idx in np.random.randint(0, len(dataset), 10):
        sample = dataset[idx]
        image = sample['image']
        image = image[:, :, :3] * 0.5 + sample['kp_heatmap'].reshape(256, 256, 1) * 50
        images.append(image.astype('uint8'))

    visualize_img(*images, cols=2, show=False)
    plt.savefig('aaver.png')
    print('finish')
