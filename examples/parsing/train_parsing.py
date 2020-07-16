from argparse import ArgumentParser
from dataset import VeRi3kParsingDataset, get_preprocessing, get_training_albumentations, get_validation_augmentation
from torch.utils.data import DataLoader
from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import argparse
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


parser = ArgumentParser()
parser.add_argument("--train-set", default="trainval")
parser.add_argument("--masks-path", default="veri776_parsing3165")
parser.add_argument("--image-path", default="/data/datasets/VeRi/VeRi/image_train")
args = parser.parse_args()

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'

CLASSES = VeRi3kParsingDataset.CLASSES
ACTIVATION = 'sigmoid'

model = smp.Unet(encoder_name=ENCODER,
                 encoder_weights=ENCODER_WEIGHTS,
                 classes=len(CLASSES),
                 activation=ACTIVATION)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# train_dataset = PascalParsingDataset(augmentation=get_training_albumentations(),
#                                      preprocessing=get_preprocessing(preprocessing_fn),
#                                      subset='training')
# valid_dataset = PascalParsingDataset(augmentation=get_validation_augmentation(),
#                                      preprocessing=get_preprocessing(preprocessing_fn),
#                                      subset='validation')

train_dataset = VeRi3kParsingDataset(image_path=args.image_path,
                                     masks_path=args.masks_path,
                                     augmentation=get_training_albumentations(),
                                     preprocessing=get_preprocessing(
                                         preprocessing_fn),
                                     subset=args.train_set)

valid_dataset = VeRi3kParsingDataset(image_path=args.image_path,
                                     masks_path=args.masks_path,
                                     augmentation=get_validation_augmentation(),
                                     preprocessing=get_preprocessing(
                                         preprocessing_fn),
                                     subset='validation')

train_loader = DataLoader(train_dataset, batch_size=8,
                          shuffle=True, num_workers=12)
valid_loader = DataLoader(valid_dataset, batch_size=1,
                          shuffle=False, num_workers=4)
# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

# loss = smp.utils.losses.BCEDiceLoss(eps=1.)
class BCEDiceLoss(smp.utils.losses.DiceLoss):
    __name__ = 'bce_dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__(eps=eps, activation=activation)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return dice + bce

loss = BCEDiceLoss(eps=1.)
metrics = [
    # smp.utils.metrics.IoUMetric(eps=1.),
    smp.utils.metrics.IoU(eps=1.),
]

optimizer = torch.optim.Adam([
    {'params': model.decoder.parameters(), 'lr': 1e-4},
    # decrease lr for encoder in order not to permute
    # pre-trained weights with large gradients on training start
    {'params': model.encoder.parameters(), 'lr': 1e-6},
])

train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)
max_score = 0

for i in range(0, 40):

    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './best_model_{}.pth'.format(args.train_set))
        print('Model saved!')

    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')
