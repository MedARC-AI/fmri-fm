# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
#
# Forked from MAE-st:
# https://github.com/facebookresearch/mae_st
# --------------------------------------------------------
import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path

import util.misc as misc

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import models_mae
import models_mae_linear
import wandb
from iopath.common.file_io import g_pathmgr as pathmgr
from omegaconf import DictConfig, OmegaConf
from torch.utils.data.distributed import DistributedSampler
from webdataset import WebLoader as DataLoader
from engine_pretrain import train_one_epoch, evaluate
from flat_data import FlatClipsDataset, make_flat_wds_dataset, make_flat_transform
from util.misc import NativeScalerWithGradNormCount as NativeScaler

PROJECT = "fMRI-foundation-model"

DEFAULT_CONFIG = Path(__file__).parent / "config/default_pretrain.yaml"

MODELS_DICT = {**models_mae.__dict__, **models_mae_linear.__dict__}


def main(args: DictConfig):
    misc.init_distributed_mode(args)

    global_rank = misc.get_rank()
    if global_rank == 0 and args.wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=PROJECT,
            name=args.name,
            notes=args.notes,
            config=OmegaConf.to_container(args),
        )

    print("pretraining fmri-fm")
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print(misc.get_sha())
    print("config:", OmegaConf.to_yaml(args), sep="\n")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    data_loaders, samplers, total_num_batches = make_data_loaders(args)

    data_loader_train = data_loaders["train"]
    data_loaders_eval = data_loaders.copy()
    data_loaders_eval.pop("train")

    num_batches_train = total_num_batches["train"]

    if global_rank == 0 and args.output_dir:
        if args.name:
            args.output_dir = f"{args.output_dir}/{args.name}"
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        out_cfg_path = output_dir / "config.yaml"
        if out_cfg_path.exists():
            prev_cfg = OmegaConf.load(out_cfg_path)
            assert args == prev_cfg, "current config doesn't match previous config"
        else:
            OmegaConf.save(args, out_cfg_path)

    # define the model
    model = MODELS_DICT[args.model](**args)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    
    num_params = sum(p.numel() for p in model_without_ddp.parameters())
    print(f"Num params = {num_params / 1e6:.1f}M")

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    args.lr = args.base_lr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            # find_unused_parameters=True,
        )
        model_without_ddp = model.module

    if global_rank == 0 and args.wandb:
        wandb.watch(model)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay(
        model_without_ddp,
        args.weight_decay,
    )
    if args.beta is None:
        beta = (0.9, 0.95)
    else:
        beta = tuple(args.beta)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=beta,
    )
    loss_scaler = NativeScaler(fp32=args.fp32)

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    checkpoint_path = ""
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            for sampler in samplers.values():
                if sampler is not None:
                    sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args=args,
            fp32=args.fp32,
            num_batches=num_batches_train,
            log_wandb=args.wandb,
        )

        eval_stats = {}
        eval_plots = {}
        for dataset_name, data_loader_eval in data_loaders_eval.items():
            ds_stats, ds_plots = evaluate(
                model,
                data_loader_eval,
                dataset_name,
                device,
                epoch,
                args=args,
                fp32=args.fp32,
                num_batches=total_num_batches[dataset_name],
                log_wandb=args.wandb,
            )
            eval_stats[dataset_name] = ds_stats
            eval_plots[dataset_name] = ds_plots

        if args.output_dir and (
            epoch % args.checkpoint_period == 0 or epoch + 1 == args.epochs or args.debug
        ):
            checkpoint_path = misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )
            misc.cleanup_checkpoints(args)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{
                f"test_{ds_name}_{k}": v for ds_name, ds_stats in eval_stats.items()
                for k, v in ds_stats.items()
            },
            "epoch": epoch,
        }

        if args.output_dir and misc.is_main_process():
            with pathmgr.open(
                f"{args.output_dir}/log.txt",
                "a",
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

        log_plots = {
            f"test_{ds_name}_{k}": img for ds_name, ds_plots in eval_plots.items()
            for k, img in ds_plots.items()
        }

        if log_plots and args.output_dir and misc.is_main_process():
            for plot_name, img in log_plots.items():
                img.save(f"{args.output_dir}/{plot_name}__{epoch:05}.png")

        if args.debug:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    print(torch.cuda.memory_allocated())
    return [checkpoint_path]


def make_data_loaders(args: DictConfig):

    transform = make_flat_transform(
        clip_vmax=args.clip_vmax,
        normalize=args.normalize,
        masking=args.masking,
        masking_kwargs=args.masking_kwargs,
    )

    data_loaders = {}
    samplers = {}
    total_num_batches = {}

    for dataset_name, dataset_config in args.datasets.items():
        print(f"dataset: {dataset_name}\n\n{OmegaConf.to_yaml(dataset_config)}")

        dataset_type = dataset_config.pop("type")

        if dataset_type == "flat-wds":
            samples_per_epoch = dataset_config.pop("samples_per_epoch")
            dataset = make_flat_wds_dataset(num_frames=args.num_frames, **dataset_config)
            dataset = dataset.map(transform)
            sampler = None
            # the shuffle happens inside the dataset with a buffer.
            shuffle = False
        elif dataset_type == "flat-clips":
            dataset = FlatClipsDataset(dataset_config.root, transform=transform)
            if args.distributed:
                sampler = DistributedSampler(dataset, shuffle=dataset_config.shuffle)
            else:
                sampler = None
            samples_per_epoch = len(dataset)
            shuffle = sampler is None and dataset_config.shuffle
        else:
            raise ValueError(f"Unknown dataset type {dataset_type}.")

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # setting the epoch length is needed for infinite wds loaders
        num_batches = samples_per_epoch // (misc.get_world_size() * args.batch_size)
        loader = loader.with_epoch(num_batches)

        data_loaders[dataset_name] = loader
        samplers[dataset_name] = sampler
        total_num_batches[dataset_name] = num_batches

    return data_loaders, samplers, total_num_batches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default=None)
    parser.add_argument("--overrides", type=str, default=None, nargs="+")
    args = parser.parse_args()
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    if args.cfg_path:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(args.cfg_path))
    if args.overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(args.overrides))
    main(cfg)
