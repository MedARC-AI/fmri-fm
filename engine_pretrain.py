# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
from typing import Iterable

import numpy as np
import util.lr_sched as lr_sched
import util.misc as misc
import util.visualization as vis
import torch
import wandb
from iopath.common.file_io import g_pathmgr as pathmgr


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    args=None,
    fp32=False,
    num_batches=None,
    log_wandb=False,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "cpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "cpu_mem_all", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "gpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Train: Epoch: [{}]".format(epoch)
    print_freq = 1 if args.debug else 20
    debug_steps = 10 * args.accum_iter
    log_wandb = misc.is_main_process() and log_wandb

    accum_iter = args.accum_iter
    if num_batches is None:
        num_batches = len(data_loader)

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(
            data_loader, print_freq, header, total_steps=num_batches
        )
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / num_batches + epoch, args
            )

        samples = batch["image"]
        img_mask = batch["mask"]
        visible_mask = batch.get("visible_mask")

        samples = samples.to(device, non_blocking=True)
        img_mask = img_mask.to(device, non_blocking=True)
        if visible_mask is not None:
            visible_mask = visible_mask.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=not fp32):
            loss, _, _ = model(
                samples,
                mask_ratio=args.mask_ratio,
                img_mask=img_mask,
                visible_mask=visible_mask,
            )

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            for _ in range(args.num_checkpoint_del):
                try:
                    path = misc.get_last_checkpoint(args)
                    pathmgr.rm(path)
                    print(f"remove checkpoint {path}")
                except Exception as _:
                    pass
            raise Exception("Loss is {}, stopping training".format(loss_value))

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
            clip_grad=args.clip_grad,
        )

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(cpu_mem=misc.cpu_mem_usage()[0])
        metric_logger.update(cpu_mem_all=misc.cpu_mem_usage()[1])
        metric_logger.update(gpu_mem=misc.gpu_mem_usage())
        # TODO: log mask stats?

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_wandb and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(
                (data_iter_step / num_batches + epoch) * 1000
            )
            wandb.log({"train/loss": loss_value_reduce, "train/lr": lr}, step=epoch_1000x)

        if args.debug and (data_iter_step + 1) >= debug_steps:
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: Iterable,
    eval_name: str,
    device: torch.device,
    epoch: int,
    args=None,
    fp32=False,
    num_batches=None,
    log_wandb=False,
):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f"Eval ({eval_name}): [{epoch}]"
    print_freq = 1 if args.debug else 20
    debug_steps = 10
    log_wandb = misc.is_main_process() and log_wandb

    model_without_ddp = model.module if args.distributed else model

    if num_batches is None:
        num_batches = len(data_loader)

    example_iter = np.random.randint(0, debug_steps if args.debug else num_batches)

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(
            data_loader, print_freq, header, total_steps=num_batches
        )
    ):
        samples = batch["image"]
        img_mask = batch["mask"]
        visible_mask = batch.get("visible_mask")

        samples = samples.to(device, non_blocking=True)
        img_mask = img_mask.to(device, non_blocking=True)
        if visible_mask is not None:
            visible_mask = visible_mask.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=not fp32):
            loss, pred, mask = model(
                samples,
                mask_ratio=args.mask_ratio,
                img_mask=img_mask,
                visible_mask=visible_mask,
            )

        loss_value = loss.item()
        metric_logger.update(loss=loss_value)

        if data_iter_step == example_iter:
            example_data = {
                "samples": samples,
                "pred": pred,
                "mask": mask,
                "img_mask": img_mask,
                "visible_mask": visible_mask,
            }

        if args.debug and (data_iter_step + 1) >= debug_steps:
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(f"Averaged stats ({eval_name}):", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # make plots
    print(f"Making plots ({eval_name}, example={example_iter})")
    samples = example_data["samples"]
    pred = example_data["pred"]
    mask = example_data["mask"]
    img_mask = example_data["img_mask"]
    visible_mask = example_data["visible_mask"]

    target, _, _, im_masked, im_paste, img_mask = model_without_ddp.forward_masked_recon(
        samples, pred, mask, img_mask=img_mask
    )
    mask_pred_fig = vis.plot_mask_pred(target, im_masked, im_paste, img_mask=img_mask)
    mask_pred_img = vis.fig2pil(mask_pred_fig)
    plots = {"mask_pred": mask_pred_img}

    if log_wandb:
        # eval at the end of training, so epoch + 1
        epoch_1000x = int((epoch + 1) * 1000)
        wandb.log({f"test/{eval_name}/{k}": v for k, v in stats.items()}, step=epoch_1000x)
        wandb.log(
            {f"test/{eval_name}/{k}": wandb.Image(img) for k, img in plots.items()},
            step=epoch_1000x,
        )

    return stats, plots
