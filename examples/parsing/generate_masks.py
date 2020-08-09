"""
Generate masks for ReID dataset.
"""
import torch
import os
import sys
import pickle
import tqdm
from dataset import *
from vehicle_reid_pytorch.data.datasets import ReIDMetaDataset
from vehicle_reid_pytorch.utils import mkdir_p
import numpy as np
from pathlib import Path
import time
import torch
import segmentation_models_pytorch as smp
import argparse

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


def predict(model, test_dataset, test_dataset_vis, output_path):
    mkdir_p(output_path)
    for i in tqdm.tqdm(range(len(test_dataset))):
        image = test_dataset[i]
        image_vis, extra = test_dataset_vis[i]

        x_tensor = torch.from_numpy(image).to("cuda").unsqueeze(0)
        with torch.no_grad():
            pr_mask = model.predict(x_tensor)
        pr_map = pr_mask.squeeze().cpu().numpy().round()
        pr_map = np.argmax(pr_map, axis=0)[
            :image_vis.shape[0], :image_vis.shape[1]]
        image_path = Path(extra["image_path"])
        mask_path = output_path / f"{image_path.name.split('.')[0]}.png"
        cv2.imwrite(str(mask_path), pr_map.astype(np.uint8))
        extra["mask_path"] = str(mask_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="best_model_trainval.pth")
    parser.add_argument("--reid-pkl-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()
    model = torch.load(args.model_path)
    model = model.cuda()
    model.eval()

    with open(args.reid_pkl_path, "rb") as f:
        metas = pickle.load(f)
    output_path = Path(args.output_path).absolute()

    for phase in metas.keys():
        sub_path = output_path / phase
        mkdir_p(str(sub_path))
        dataset = VehicleReIDParsingDataset(metas[phase], augmentation=get_validation_augmentation(),
                                            preprocessing=get_preprocessing(preprocessing_fn))
        dataset_vis = VehicleReIDParsingDataset(metas[phase], with_extra=True)
        print('Predict mask to {}'.format(sub_path))
        predict(model, dataset, dataset_vis, sub_path)

    # Write mask path to pkl
    with open(args.reid_pkl_path, "wb") as f:
        pickle.dump(metas, f)
