import time
import os
import click
from tqdm import tqdm
import logzero
import logging
from logzero import logger
import torch
import time
from torch import nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from vehicle_reid_pytorch.data import samplers
from vehicle_reid_pytorch.models import Baseline
from vehicle_reid_pytorch.utils import mkdir_p, load_checkpoint, save_checkpoint, merge_configs, get_host_ip
from vehicle_reid_pytorch.metrics import R1_mAP
from vehicle_reid_pytorch import loss as vr_loss
from vehicle_reid_pytorch.utils.pytorch_tools import make_optimizer, make_warmup_scheduler
from yacs.config import CfgNode
from vehicle_reid_pytorch.models import Baseline
from vehicle_reid_pytorch.data import make_basic_dataset
from model import ParsingReidModel, ParsingTripletLoss
from math_tools import Clck_R1_mAP
import numpy as np


torch.set_printoptions(precision=2, linewidth=200, sci_mode=False)


def make_config():
    cfg = CfgNode()
    cfg.desc = ""  # 对本次实验的简单描述，用于为tensorboard命名
    cfg.stage = "train"  # train or eval or test
    cfg.device = "cpu"  # cpu or cuda
    cfg.device_ids = ""  # if not set, use all gpus
    cfg.output_dir = "/data/vehicle_reid/perspective_transform_feature/debug"
    cfg.debug = False

    cfg.train = CfgNode()
    cfg.train.epochs = 120

    cfg.data = CfgNode()
    cfg.data.name = "VeRi776"
    cfg.data.pkl_path = "../data_processing/veri776.pkl"
    cfg.data.train_size = (256, 256)
    cfg.data.valid_size = (256, 256)
    cfg.data.pad = 10
    cfg.data.re_prob = 0.5
    cfg.data.with_mask = True
    cfg.data.test_ext = ''

    cfg.data.sampler = 'RandomIdentitySampler'
    cfg.data.batch_size = 16
    cfg.data.num_instances = 4

    cfg.data.train_num_workers = 0
    cfg.data.test_num_workers = 0

    cfg.model = CfgNode()
    cfg.model.name = "resnet50"
    # If it is set to empty, we will download it from torchvision official website.
    cfg.model.pretrain_path = ""
    cfg.model.last_stride = 1
    cfg.model.neck = 'bnneck'
    cfg.model.neck_feat = 'after'
    cfg.model.pretrain_choice = 'imagenet'
    cfg.model.ckpt_period = 10

    cfg.optim = CfgNode()
    cfg.optim.name = 'Adam'
    cfg.optim.base_lr = 3.5e-4
    cfg.optim.bias_lr_factor = 1
    cfg.optim.weight_decay = 0.0005
    cfg.optim.momentum = 0.9

    cfg.loss = CfgNode()
    cfg.loss.losses = ["triplet", "id", "center", "local-triplet"]
    cfg.loss.triplet_margin = 0.3
    cfg.loss.normalize_feature = True
    cfg.loss.id_epsilon = 0.1

    cfg.loss.center_lr = 0.5
    cfg.loss.center_weight = 0.0005

    cfg.loss.tuplet_s = 64
    cfg.loss.tuplet_beta = 0.1

    cfg.scheduler = CfgNode()
    cfg.scheduler.milestones = [40, 70]
    cfg.scheduler.gamma = 0.1
    cfg.scheduler.warmup_factor = 0.0
    cfg.scheduler.warmup_iters = 10
    cfg.scheduler.warmup_method = "linear"

    cfg.test = CfgNode()
    cfg.test.feat_norm = True
    cfg.test.remove_junk = True
    cfg.test.period = 10
    cfg.test.device = "cuda"
    cfg.test.model_path = "../outputs/veri776.pth"
    cfg.test.max_rank = 50
    cfg.test.rerank = False
    cfg.test.lambda_ = 0.0
    cfg.test.output_html_path = ""
    # split: When the CUDA memory is not sufficient, 
    # we can split the dataset into different parts
    # for the computing of distance.
    cfg.test.split = 0  

    cfg.logging = CfgNode()
    cfg.logging.level = "info"
    cfg.logging.period = 20

    return cfg


def build_model(cfg, num_classes):
    # if cfg.MODEL.NAME == 'resnet50':
    #     model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    model = ParsingReidModel(num_classes, cfg.model.last_stride, cfg.model.pretrain_path, cfg.model.neck,
                             cfg.model.neck_feat, cfg.model.name, cfg.model.pretrain_choice)
    return model


@click.group()
def clk():
    pass


# global
@clk.command()
@click.option("-c", "--config-files", type=str, default="")
@click.argument("cmd-config", nargs=-1)
def train(config_files, cmd_config):
    """
    Training models.
    """
    cfg = make_config()
    cfg = merge_configs(cfg, config_files, cmd_config)

    mkdir_p(cfg.output_dir)
    logzero.logfile(f"{cfg.output_dir}/train.log")
    logzero.loglevel(getattr(logging, cfg.logging.level.upper()))
    logger.info(cfg)
    logger.info(f"worker ip is {get_host_ip()}")

    writter = SummaryWriter(
        comment=f"{cfg.data.name}_{cfg.model.name}__{cfg.data.batch_size}")

    logger.info(f"Loading {cfg.data.name} dataset")

    train_dataset, valid_dataset, meta_dataset = make_basic_dataset(cfg.data.pkl_path,
                                                                    cfg.data.train_size,
                                                                    cfg.data.valid_size,
                                                                    cfg.data.pad,
                                                                    test_ext=cfg.data.test_ext,
                                                                    re_prob=cfg.data.re_prob,
                                                                    with_mask=cfg.data.with_mask,
                                                                    )
    num_class = meta_dataset.num_train_ids
    sampler = getattr(samplers, cfg.data.sampler)(train_dataset.meta_dataset, cfg.data.batch_size, cfg.data.num_instances)
    train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=cfg.data.batch_size,
                              num_workers=cfg.data.train_num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.test_num_workers,
                              pin_memory=True, shuffle=False)
    logger.info(f"Successfully load {cfg.data.name}!")

    logger.info(f"Building {cfg.model.name} model, "
                f"num class is {num_class}")
    model = build_model(cfg, num_class).to(cfg.device)

    logger.info(f"Building {cfg.optim.name} optimizer...")

    optimizer = make_optimizer(cfg.optim.name,
                               model,
                               cfg.optim.base_lr,
                               cfg.optim.weight_decay,
                               cfg.optim.bias_lr_factor,
                               cfg.optim.momentum)

    logger.info(f"Building losses {cfg.loss.losses}")

    triplet_loss = None
    id_loss = None
    center_loss = None
    optimizer_center = None
    tuplet_loss = None
    if 'local-triplet' in cfg.loss.losses:
        pt_loss = ParsingTripletLoss(margin=0.3)
    if 'triplet' in cfg.loss.losses:
        triplet_loss = vr_loss.TripletLoss(margin=cfg.loss.triplet_margin)
    if 'id' in cfg.loss.losses:
        id_loss = vr_loss.CrossEntropyLabelSmooth(num_class, cfg.loss.id_epsilon)
        # id_loss = vr_losses.CrossEntropyLabelSmooth(num_class, cfg.loss.id_epsilon, keep_dim=False)
    if 'center' in cfg.loss.losses:
        center_loss = vr_loss.CenterLoss(num_class, feat_dim=model.in_planes).to(cfg.device)
        optimizer_center = torch.optim.SGD(center_loss.parameters(), cfg.loss.center_lr)
    if 'tuplet' in cfg.loss.losses:
        tuplet_loss = vr_loss.TupletLoss(cfg.data.num_instances, cfg.data.batch_size // cfg.data.num_instances,
                                         cfg.loss.tuplet_s, cfg.loss.tuplet_beta)

    start_epoch = 1
    if cfg.model.pretrain_choice == "self":
        logger.info(f"Loading checkpoint from {cfg.output_dir}")
        if "center_loss" in cfg.loss.losses:
            start_epoch = load_checkpoint(cfg.output_dir, cfg.device, model=model, optimizer=optimizer,
                                          optimizer_center=optimizer_center, center_loss=center_loss)
        else:
            start_epoch = load_checkpoint(cfg.output_dir, cfg.device, model=model, optimizer=optimizer)
        logger.info(f"Loaded checkpoint successfully! Start epoch is {start_epoch}")

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    scheduler = make_warmup_scheduler(optimizer,
                                      cfg.scheduler.milestones,
                                      cfg.scheduler.gamma,
                                      cfg.scheduler.warmup_factor,
                                      cfg.scheduler.warmup_iters,
                                      cfg.scheduler.warmup_method,
                                      last_epoch=start_epoch - 1)

    logger.info("Start training!")
    for epoch in range(start_epoch, cfg.train.epochs + 1):
        t_begin = time.time()
        scheduler.step()
        running_loss = 0
        running_acc = 0
        gpu_time = 0
        data_time = 0

        t0 = time.time()
        for iter, batch in enumerate(train_loader):
            t1 = time.time()
            data_time += t1 - t0
            global_steps = (epoch - 1) * len(train_loader) + iter
            model.train()
            optimizer.zero_grad()

            if 'center' in cfg.loss.losses:
                optimizer_center.zero_grad()

            for name, item in batch.items():
                if isinstance(item, torch.Tensor):
                    batch[name] = item.to(cfg.device)

            output = model(**batch)
            global_feat = output["global_feat"]
            global_score = output["cls_score"]
            local_feat = output["local_feat"]
            vis_score = output["vis_score"]

            # losses
            loss = 0
            if "id" in cfg.loss.losses:
                g_xent_loss = id_loss(global_score, batch["id"]).mean()
                loss += g_xent_loss
                logger.debug(f'ID Loss: {g_xent_loss.item()}')
                writter.add_scalar("global_loss/id_loss",
                                   g_xent_loss.item(), global_steps)

            if "triplet" in cfg.loss.losses:
                t_loss, _, _ = triplet_loss(global_feat, batch["id"], normalize_feature=False)
                logger.debug(f'Triplet Loss: {t_loss.item()}')
                loss += t_loss
                writter.add_scalar("global_loss/triplet_loss", t_loss.item(), global_steps)

            if "center" in cfg.loss.losses:
                g_center_loss = center_loss(global_feat, batch["id"])
                logger.debug(g_center_loss.item())
                loss += cfg.loss.center_weight * g_center_loss
                writter.add_scalar("global_loss/center_loss", g_center_loss.item(), global_steps)

            if "tuplet" in cfg.loss.losses:
                g_tuplet_loss = tuplet_loss(global_feat)
                loss += g_tuplet_loss
                writter.add_scalar("global_loss/tuplet_loss", g_tuplet_loss.item(), global_steps)

            if "local-triplet" in cfg.loss.losses:
                l_triplet_loss, _, _ = pt_loss(
                    local_feat, vis_score, batch["id"], True)
                writter.add_scalar("local_loss/triplet_loss", l_triplet_loss.item(), global_steps)
                loss += l_triplet_loss

            loss.backward()
            optimizer.step()

            # centerloss单独优化
            if 'center' in cfg.loss.losses:
                for param in center_loss.parameters():
                    param.grad.data *= (1. / cfg.loss.center_weight)
                optimizer_center.step()

            acc = (global_score.max(1)[1] == batch["id"]).float().mean()

            # running mean
            if iter == 0:
                running_acc = acc.item()
                running_loss = loss.item()
            else:
                running_acc = 0.98 * running_acc + 0.02 * acc.item()
                running_loss = 0.98 * running_loss + 0.02 * loss.item()

            if iter % cfg.logging.period == 0:
                logger.info(
                    f"Epoch[{epoch:3d}] Iteration[{iter:4d}/{len(train_loader):4d}] "
                    f"Loss: {running_loss:.3f}, Acc: {running_acc:.3f}, Base Lr: {scheduler.get_lr()[0]:.2e}")
                if cfg.debug:
                    break
            t0 = time.time()
            gpu_time += t0 - t1
            logger.debug(f"GPU Time: {gpu_time}, Data Time: {data_time}")

        t_end = time.time()

        logger.info(
            f"Epoch {epoch} done. Time per epoch: {t_end - t_begin:.1f}[s] "
            f"Speed:{(t_end - t_begin) / len(train_loader.dataset):.1f}[samples/s] ")
        logger.info('-' * 10)

        # 测试模型, veriwild在训练时测试会导致显存溢出,训练后单独测试。 vehicleid使用不同的测试策略，也训练后单独测试
        if (epoch == 1 or epoch % cfg.test.period == 0) and cfg.data.name.lower() != 'veriwild' and cfg.data.name.lower() != 'vehicleid':
            query_length = meta_dataset.num_query_imgs
            if query_length != 0:  # Private没有测试集
                eval_(model,
                      device=cfg.device,
                      valid_loader=valid_loader,
                      query_length=query_length,
                      feat_norm=cfg.test.feat_norm,
                      remove_junk=cfg.test.remove_junk,
                      lambda_=cfg.test.lambda_,
                      output_dir=cfg.output_dir,
                      output_html_path=cfg.test.output_html_path)

        # save checkpoint
        if epoch % cfg.model.ckpt_period == 0 or epoch == 1:
            logger.info(f"Saving models in epoch {epoch}")
            if 'center' in cfg.loss.losses:
                save_checkpoint(epoch, cfg.output_dir, model=model, optimizer=optimizer,
                                center_loss=center_loss, optimizer_center=optimizer_center)
            else:
                save_checkpoint(epoch, cfg.output_dir,
                                model=model, optimizer=optimizer)


@clk.command()
@click.option("-c", "--config-files", type=str, default="")
@click.argument("cmd-config", nargs=-1)
def eval(config_files, cmd_config):
    cfg = make_config()
    cfg = merge_configs(cfg, config_files, cmd_config)

    os.makedirs(cfg.output_dir, exist_ok=True)

    model = build_model(cfg, 1).to(cfg.device)
    # start_epoch = load_checkpoint(cfg.output_dir, device=cfg.device, epoch=cfg.test.epoch, exclude="classifier", model=model)
    state_dict = torch.load(cfg.test.model_path, map_location=cfg.device)

    # Remove the classifier
    remove_keys = []
    for key, value in state_dict.items():
        if 'classifier' in key:
            remove_keys.append(key)
    for key in remove_keys:
        del state_dict[key]

    model.load_state_dict(state_dict, strict=False)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    logger.info(f"Load model {cfg.test.model_path}")
    train_dataset, valid_dataset, meta_dataset = make_basic_dataset(cfg.data.pkl_path,
                                                                    cfg.data.train_size,
                                                                    cfg.data.valid_size,
                                                                    cfg.data.pad,
                                                                    test_ext=cfg.data.test_ext,
                                                                    re_prob=cfg.data.re_prob,
                                                                    with_mask=cfg.data.with_mask,
                                                                    )
    valid_loader = DataLoader(valid_dataset, 
                              batch_size=cfg.data.batch_size, 
                              num_workers=cfg.data.test_num_workers, 
                              pin_memory=True, 
                              shuffle=False)

    query_length = meta_dataset.num_query_imgs

    if cfg.data.name.lower() == "vehicleid":
        eval_vehicle_id_(model, valid_loader, query_length, cfg)
    else:
        eval_(model, cfg.test.device, valid_loader, query_length, 
            feat_norm=cfg.test.feat_norm,
            remove_junk=cfg.test.remove_junk, 
            max_rank=cfg.test.max_rank, 
            output_dir=cfg.output_dir, 
            lambda_=cfg.test.lambda_,
            rerank=cfg.test.rerank, 
            split=cfg.test.split,
            output_html_path=cfg.test.output_html_path)


def eval_(model, 
          device, 
          valid_loader, 
          query_length, 
          feat_norm=True, 
          remove_junk=True, 
          max_rank=50, 
          output_dir='', 
          rerank=False, 
          lambda_=0.5,
          split=0,
          output_html_path=''):
    """实际测试函数

    Arguments:
        model {nn.Module}} -- 模型
        device {string} -- 设备
        valid_loader {DataLoader} -- 测试集
        query_length {int} -- 测试集长度

    Keyword Arguments:
        remove_junk {bool} -- 是否删除垃圾图片 (default: {True})
        max_rank {int} -- [description] (default: {50})
        output_dir {str} -- 输出目录。若为空则不输出。 (default: {''})

    Returns:
        [type] -- [description]
    """
    metric = Clck_R1_mAP(query_length, max_rank=max_rank, rerank=rerank, remove_junk=remove_junk, feat_norm=feat_norm, output_path=output_dir, lambda_=lambda_)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            for name, item in batch.items():
                if isinstance(item, torch.Tensor):
                    batch[name] = item.to("cuda")
            output = model(**batch)
            global_feat = output["global_feat"]
            local_feat = output["local_feat"]
            vis_score = output["vis_score"]
            metric.update((global_feat.detach().cpu(), local_feat.detach().cpu(), vis_score.cpu(), batch["id"].cpu(), batch["cam"].cpu(), batch["image_path"]))

    metric_output = metric.compute(split=split)
    cmc = metric_output['cmc'] 
    mAP = metric_output['mAP']
    distmat = metric_output['distmat'] 
    all_AP = metric_output['all_AP']

    if output_html_path != '':
        from vehicle_reid_pytorch.utils.visualize import reid_html_table 
        query = valid_loader.dataset.meta_dataset[:query_length]
        gallery = valid_loader.dataset.meta_dataset[query_length:]
        # distmat = np.random.rand(query_length, len(valid_loader.dataset.meta_dataset)-query_length)
        reid_html_table(query, gallery, distmat, output_html_path, all_AP, topk=15)


    # distmat = metric.distmat
    # gallery_idxs = np.argsort(distmat, axis=-1)
    # meta_dataset = valid_loader.dataset.meta_dataset

    # f = open('submit.txt', 'w')
    # for query_idx in range(len(meta_dataset.query)):
    #     names = []
    #     for gallery_idx in gallery_idxs[query_idx][:100]:
    #         name = meta_dataset.gallery[gallery_idx]['name'].split('.')[0]
    #         names.append(name)
    #     names_str = ' '.join(names) + '\n'
    #     f.write(names_str)
    # f.close()

    metric.reset()
    logger.info(f"mAP: {mAP:.1%}")
    for r in [1, 5, 10]:
        logger.info(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.2%}")
    return cmc, mAP


def eval_vehicle_id_(model, valid_loader, query_length, cfg):
    metric = Clck_R1_mAP(query_length, 
                         max_rank=cfg.test.max_rank, 
                         rerank=cfg.test.rerank, 
                         remove_junk=cfg.test.remove_junk, 
                         feat_norm=cfg.test.feat_norm, 
                         output_path=cfg.output_dir, 
                         lambda_=cfg.test.lambda_)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            for name, item in batch.items():
                if isinstance(item, torch.Tensor):
                    batch[name] = item.to("cuda")
            output = model(**batch)
            global_feat = output["global_feat"]
            local_feat = output["local_feat"]
            vis_score = output["vis_score"]
            metric.update((global_feat.detach().cpu(), local_feat.detach().cpu(), vis_score.cpu(), batch["id"].cpu(), batch["cam"].cpu(), ""))
    
    mAPs = []
    cmcs = []
    for i in range(10):
        metric.resplit_for_vehicleid()
        metric_output = metric.compute()
        cmc = metric_output['cmc'] 
        mAP = metric_output['mAP']
        mAPs.append(mAP)
        cmcs.append(cmc)

    mAP = np.mean(mAPs)
    cmc = np.mean(cmcs, axis=0)
    logger.info(f"mAP: {mAP:.2%}")
    for r in [1, 5, 10]:
        logger.info(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.2%}")
    return cmc, mAP

@clk.command()
@click.option('-i', '--model-path')
@click.option('-o', '--output-path')
def drop_linear(model_path, output_path):
    model = torch.load(model_path)
    for key in model.keys():
        if 'classifier' in key:
            model[key] = None
    torch.save(model, output_path)

if __name__ == '__main__':
    clk()
