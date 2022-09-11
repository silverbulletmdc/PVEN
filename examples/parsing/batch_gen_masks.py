"""
Generate masks for ReID dataset.
"""
import math
import torch
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool
import click
import albumentations as albu
from mdc_tools.timer import Timer
import cv2
import pandas as pd
# import debugpy; debugpy.connect(('100.64.158.205', 5678))

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

IMG2MASK = {}

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def travel_imgs(root, exts=('.png', '.jpg', '.jpeg')):
    outputs = []
    for dir_, dirs, files in os.walk(root):
        for file in files:
            file: str
            if os.path.splitext(file)[1] in exts:
                outputs.append(os.path.join(dir_, file))
    return outputs


class ParsingTestDataset(Dataset):
    def __init__(self, root_path) -> None:
        self.transforms = albu.Compose([
            albu.LongestMaxSize(244),
            albu.PadIfNeeded(min_height=256, min_width=256, always_apply=True,
                             border_mode=cv2.BORDER_CONSTANT, position='top_left'),
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=to_tensor, mask=to_tensor)
        ])
        self.root_path = root_path
        with Timer(f"Scanning images in {root_path}..."):
            self.image_paths = travel_imgs(root_path)
        print(f"Total {len(self.image_paths)} images.")

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_shape = image.shape[:2]

        image = self.transforms(image=image)['image']
        return image, image_path, torch.tensor(image_shape)

    def __len__(self):
        return len(self.image_paths)


def write_worker(args):
    mask, image_path, image_shape, input_path, output_path = args
    # 处理路径
    image_rel_path = os.path.relpath(image_path, input_path)
    save_rel_path = os.path.splitext(image_rel_path)[0] + '.png'
    image_output_path = os.path.join(output_path, save_rel_path)
    os.makedirs(os.path.split(image_output_path)[0], exist_ok=True)

    # 处理形状
    image_shape = np.array(image_shape)
    target_shape = (244 / max(image_shape) * image_shape).astype(np.int)
    mask = mask[:target_shape[0], :target_shape[1]].astype(np.uint8)
    mask = cv2.resize(mask, image_shape, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(image_output_path, mask)


@click.group()
def main():
    pass


@main.command()
@click.option("--model-path", default="parsing_model.pth")
@click.option("--input-path", type=str, required=True)
@click.option("--output-path", type=str, default='')
@click.option("--batch-size", type=int, default=128)
@click.option("--num-workers", type=int, default=16)
@click.option("--device", type=str, default='cuda:0')
def parse_folder(model_path, input_path, output_path, batch_size, num_workers, device):
    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()

    output_path = output_path

    dataset = ParsingTestDataset(input_path)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers)

    pool = Pool(16)
    for batch in tqdm(dataloader, total=math.ceil(len(dataset) / batch_size)):
        images, image_paths, image_shapes = batch
        images = images.to(device)
        with torch.no_grad():
            # [B, C, H, W]
            pr_map = model.predict(images)

        masks: torch.Tensor = pr_map.round().argmax(dim=1).detach().cpu().numpy()
        args = [(mask, image_path, image_shape, input_path, output_path)
                for mask, image_path, image_shape in zip(masks, image_paths, image_shapes)]
        iters = pool.imap(write_worker, args)
    
    for iter in iters:
        pass
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
