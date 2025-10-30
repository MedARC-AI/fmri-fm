import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path
import math

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data.distributed import DistributedSampler

import flat_mae.utils as ut
import flat_mae.models_mae as models_mae

import data.flat_data as flat_data

from simclr.data import SimCLRTransform, simclr_collate
from simclr.models import ContrastiveModel
from simclr.loss import nt_xent_loss, simsiam_loss

import flat_mae.models_mae as models_mae

PROJECT = "fMRI-foundation-model"


BACKBONE_MODELS_DICT = models_mae.__dict__

def main(args: DictConfig):
    ut.init_distributed_mode(args)
    global_rank = ut.get_rank()
    is_master = global_rank == 0
    world_size = ut.get_world_size()
    device = torch.device(args.device)
    ut.random_seed(args.seed, rank=global_rank)

    if args.name and not args.output_dir.endswith(args.name):
        args.output_dir = f"{args.output_dir}/{args.name}"
    output_dir = Path(args.output_dir)

    if is_master:
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(args, output_dir / "config.yaml")

    ut.setup_for_distributed(log_path=output_dir / "log.txt")
    print(f"pretraining with {args.model.contrastive_mode}")
    print("config:", OmegaConf.to_yaml(args), sep="\n")

    train_loader, eval_loaders = create_data_loaders(args)

    print(f"Creating backbone: {args.model.backbone_name}")
    backbone = BACKBONE_MODELS_DICT[args.model.backbone_name](
        img_size=args.data.img_size,
        in_chans=args.data.in_chans,
        **args.model.get("backbone_kwargs", {}),
    )
    model = ContrastiveModel(
        backbone=backbone,
        mode=args.model.contrastive_mode,
        embed_dim=args.model.backbone_kwargs.embed_dim,
        model_kwargs=args.model.get("head_kwargs"),
    ).to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    print("model:", model, sep="\n")

    total_batch_size = args.optim.batch_size * args.optim.accum_iter * world_size
    if not args.optim.get("lr"): args.optim.lr = args.optim.base_lr * total_batch_size / 256

    param_groups = ut.get_param_groups(model)
    ut.update_lr(param_groups, args.optim.lr)
    ut.update_wd(param_groups, args.optim.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, betas=tuple(args.optim.betas))

    epoch_num_batches = len(train_loader)
    steps_per_epoch = epoch_num_batches // args.optim.accum_iter
    total_steps = args.optim.epochs * steps_per_epoch
    warmup_steps = args.optim.warmup_epochs * steps_per_epoch
    lr_schedule = ut.WarmupThenCosine(
        base_value=args.optim.lr, final_value=args.optim.min_lr,
        total_iters=total_steps, warmup_iters=warmup_steps
    )

    loss_scaler = ut.GradScaler() if args.amp and args.amp_dtype != 'bfloat16' else None

    ut.load_model(args, model_without_ddp, optimizer, loss_scaler)

    print(f"start training for {args.optim.epochs} epochs")
    for epoch in range(args.start_epoch, args.optim.epochs):
        if args.distributed: train_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(args, model, train_loader, optimizer, loss_scaler, lr_schedule, epoch, device)


        if args.output_dir:
            ut.save_model(args, epoch, model_without_ddp, optimizer, loss_scaler)


def create_data_loaders(args: DictConfig):
    base_transform = flat_data.make_flat_transform(
        img_size=args.data.img_size,
        clip_vmax=args.data.get("clip_vmax"),
        normalize=args.data.get("normalize"),
        random_crop=args.data.get("random_crop", False),
        crop_kwargs=args.data.get("crop_kwargs"),
    )

    transform = SimCLRTransform(base_transform)

    data_loaders = {}
    dataset_names = [args.train_dataset] + args.eval_datasets
    for name in dataset_names:
        config = args.datasets[name]
        dataset = flat_data.FlatClipsDataset(root=config.root, transform=transform)
        sampler = DistributedSampler(dataset, shuffle=config.shuffle) if args.distributed else None

        loader = flat_data.DataLoader(
            dataset, batch_size=args.optim.batch_size,
            collate_fn=simclr_collate, sampler=sampler,
            shuffle=sampler is None and config.shuffle,
            num_workers=args.num_workers, pin_memory=True, drop_last=True
        )
        data_loaders[name] = loader
    
    train_loader = data_loaders.pop(args.train_dataset)
    return train_loader, data_loaders


def train_one_epoch(args, model, data_loader, optimizer, loss_scaler, lr_schedule, epoch, device):
    # --- This is the training engine, adapted for SimCLR/SimSiam ---
    model.train()
    metric_logger = ut.MetricLogger(delimiter="  ")
    header = f'Train: [{epoch}]'

    epoch_num_batches = len(data_loader)
    steps_per_epoch = epoch_num_batches // args.optim.accum_iter

    optimizer.zero_grad()

    for batch_idx, (batch_view_1, batch_view_2) in enumerate(metric_logger.log_every(data_loader, 100, header)):

        global_step = epoch * steps_per_epoch + (batch_idx + 1) // args.optim.accum_iter
        lr = lr_schedule[global_step]
        need_update = (batch_idx + 1) % args.optim.accum_iter == 0
        if need_update: ut.update_lr(optimizer.param_groups, lr)

        view_1 = batch_view_1['image'].to(device, non_blocking=True)
        view_2 = batch_view_2['image'].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=getattr(torch, args.amp_dtype), enabled=args.amp):

            outputs = model(view_1, view_2, mask_ratio=args.model.mask_ratio)

            if args.model.contrastive_mode == "simclr":
                z1, z2 = outputs
                loss = nt_xent_loss(z1, z2, temperature=args.model.get("temperature", 0.5), distributed=args.distributed)
            elif args.model.contrastive_mode == "simsiam":
                p1, z2, p2, z1 = outputs
                loss = simsiam_loss(p1, z2, p2, z1)
            else:
                raise ValueError(f"Unknown contrastive mode: {args.model.contrastive_mode}")

        loss_value = loss.item()
        if not math.isfinite(loss_value): raise RuntimeError(f"Loss is {loss_value}, stopping training")

        ut.backward_step(loss / args.optim.accum_iter, optimizer, scaler=loss_scaler, 
                         need_update=need_update, max_norm=args.optim.get("clip_grad"))

        metric_logger.update(loss=loss_value)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, data_loader, name, device, args):
    model.eval()
    print(f"--- Running evaluation on {name} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, required=True)
    parser.add_argument("--overrides", type=str, default=None, nargs="+")
    cli_args = parser.parse_args()

    cfg = OmegaConf.load(cli_args.cfg_path)
    if cli_args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(cli_args.overrides))

    main(cfg)