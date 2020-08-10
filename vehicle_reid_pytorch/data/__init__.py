from . import datasets
from . import demo_transforms as demo_trans


def make_basic_dataset(pkl_path, train_size, val_size, pad, *, test_ext='', re_prob=0.5, with_mask=False, for_vis=False):
    """
    构建基础数据集。
    """

    meta_dataset = datasets.CommonReIDDataset(pkl_path=pkl_path, test_ext=test_ext)
    train_transform = demo_trans.get_training_albumentations(train_size, pad, re_prob)
    val_transform = demo_trans.get_validation_augmentations(val_size)
    if for_vis:
        preprocessing = None
    else:
        preprocessing = demo_trans.get_preprocessing()

    train_dataset = datasets.ReIDDataset(
        meta_dataset.train, with_mask=with_mask, transform=train_transform, preprocessing=preprocessing)

    val_dataset = datasets.ReIDDataset(meta_dataset.query + meta_dataset.gallery, with_mask=with_mask, transform=val_transform,
                                       preprocessing=preprocessing)

    return train_dataset, val_dataset, meta_dataset
