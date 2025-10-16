# This source code is licensed under the Apache License, Version 2.0
#
# References:
# deit: https://github.com/facebookresearch/deit/blob/main/main.py
# capi: https://github.com/facebookresearch/capi/blob/main/eval_classification.py

import argparse
import datetime
import json
import math
import time
from collections import defaultdict
from functools import partial
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from timm.utils import accuracy

import data.flat_data as flat_data
import flat_mae.models_probe as models_probe
import flat_mae.utils as ut
import flat_mae.masking as masking
import flat_mae.models_mae as models_mae

DEFAULT_CONFIG = Path(__file__).parent / "config/default_probe.yaml"

MODELS_DICT = models_mae.__dict__


def main(args: DictConfig):
    # setup
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
        out_cfg_path = output_dir / "config.yaml"
        if out_cfg_path.exists():
            prev_cfg = OmegaConf.load(out_cfg_path)
            assert args == prev_cfg, "current config doesn't match previous config"
        else:
            OmegaConf.save(args, out_cfg_path)

        if args.wandb:
            wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=args.name,
                notes=args.notes,
                config=OmegaConf.to_container(args),
            )

    ut.setup_for_distributed(log_path=output_dir / "log.txt")

    print("probe eval flat map mae")
    print(f"start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"cwd: {Path.cwd()}")
    print(ut.get_sha())
    print("config:", OmegaConf.to_yaml(args), sep="\n")

    # data loaders
    train_loader, test_loader, eval_loaders = create_data_loaders(args)

    # backbone embedding model
    if args.pretrain_ckpt:
        print(f"creating backbone model from checkpoint: {args.pretrain_ckpt}")
        ckpt = torch.load(args.pretrain_ckpt, map_location="cpu", weights_only=True)
        ckpt_args = OmegaConf.create(ckpt["args"])
        backbone = MODELS_DICT[ckpt_args.model](
            img_size=ckpt_args.img_size,
            in_chans=ckpt_args.in_chans,
            patch_size=ckpt_args.patch_size,
            num_frames=ckpt_args.num_frames,
            t_patch_size=ckpt_args.t_patch_size,
            **ckpt_args.model_kwargs,
        )
        backbone.load_state_dict(ckpt["model"])
    else:
        print(f"creating backbone model from scratch: {args.model}")
        backbone = MODELS_DICT[args.model](
            img_size=args.img_size,
            in_chans=args.in_chans,
            patch_size=args.patch_size,
            num_frames=args.num_frames,
            t_patch_size=args.t_patch_size,
            **args.model_kwargs,
        )

    if not args.finetune:
        print("freezing backbone model")
        backbone.requires_grad_(False)
        backbone_param_groups = []
    else:
        print("finetuning backbone model")
        backbone_param_groups = ut.get_param_groups(backbone)

    backbone.to(device)
    embedding_shapes = get_embedding_shapes(backbone, args.representations, train_loader, device)
    print(f"embedding feature shapes:\n{embedding_shapes}")

    print("initializing sweep of classifier heads")
    classifiers, classifier_param_groups = make_classifiers(args, embedding_shapes)

    model = models_probe.ClassificationWrapper(backbone, classifiers)
    model.to(device)
    print(f"model:\n{model}")
    num_params = sum(p.numel() for p in model.parameters())
    num_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"num params (train): {num_params / 1e6:.1f}M ({num_params_train / 1e6:.1f}M)")

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # todo: compile?

    # optimizer
    total_batch_size = args.batch_size * args.accum_iter * world_size
    print(
        f"total batch size: {total_batch_size} = "
        f"{args.batch_size} bs per gpu x {args.accum_iter} accum x {world_size} gpus"
    )

    if not args.get("lr"):
        args.lr = args.base_lr * total_batch_size / 256
        print(f"lr: {args.lr:.2e} = {args.base_lr:.2e} x {total_batch_size} / 256")
    else:
        print(f"lr: {args.lr:.2e}")

    param_groups = backbone_param_groups + classifier_param_groups
    ut.update_lr(param_groups, args.lr)
    ut.update_wd(param_groups, args.weight_decay)
    # cast or else it corrupts the checkpoint
    betas = tuple(args.betas) if args.betas is not None else None
    optimizer = torch.optim.AdamW(param_groups, betas=betas)

    epoch_num_batches = len(train_loader)
    steps_per_epoch = epoch_num_batches // args.accum_iter
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    lr_schedule = ut.WarmupThenCosine(
        base_value=args.lr,
        final_value=args.min_lr,
        total_iters=total_steps,
        warmup_iters=warmup_steps,
    )
    print(f"full schedule: epochs = {args.epochs} (steps = {total_steps})")
    print(f"warmup: epochs = {args.warmup_epochs} (steps = {warmup_steps})")

    # loss scaling not needed for bfloat16 (according to timm)
    if args.amp and args.amp_dtype != "bfloat16":
        loss_scaler = torch.GradScaler(device.type)
    else:
        loss_scaler = None

    # load checkpoint/resume training
    ut.load_model(args, model_without_ddp, optimizer, loss_scaler)

    # training loss
    if args.task == "classification":
        criterion = nn.CrossEntropyLoss(reduction="none")
    elif args.task == "regression":
        if args.loss_function == "mse":
            criterion = nn.MSELoss(reduction="none")
        elif args.loss_function == "cosine":
            # predictions are shape [batch_size, num_targets, num_classifiers]
            # normalize over the target dimension
            criterion = models_probe.CosineLoss(dim=-2, reduction="none")
        else:
            raise ValueError(f"unknown loss_function {args.loss_function}")
    else:
        raise ValueError(f"unknown task {args.task}")

    # watch middle classifiers on first epoch
    # for later epochs will update to the best
    # a classifier "key" is a tuple of (feature_source, (lr_scale, weight_decay))
    mid_lr_scale = args.lr_scale_grid[len(args.lr_scale_grid) // 2]
    mid_weight_decay = args.weight_decay_grid[len(args.weight_decay_grid) // 2]
    log_classifier_keys = [
        (feature_source, (mid_lr_scale, mid_weight_decay))
        for feature_source in args.representations
    ]

    print(f"start training for {args.epochs} epochs")
    start_time = time.monotonic()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and hasattr(train_loader, "sampler"):
            train_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            args,
            model,
            criterion,
            train_loader,
            optimizer,
            loss_scaler,
            lr_schedule,
            epoch,
            device,
            log_classifier_keys=log_classifier_keys,
        )

        eval_stats = {}
        for name, loader in eval_loaders.items():
            stats = evaluate(
                args,
                model,
                criterion,
                loader,
                epoch,
                device,
                eval_name=name,
                log_classifier_keys=log_classifier_keys,
            )
            eval_stats.update(stats)

        merged_stats = {"epoch": epoch, **train_stats, **eval_stats}
        if is_master:
            with (output_dir / "log.json").open("a") as f:
                print(json.dumps(merged_stats), file=f)

        best_scores, best_hparams = get_best_classifiers(args, model_without_ddp, eval_stats)

        # log the best classifiers
        log_classifier_keys = [
            (feature_source, hparam) for feature_source, hparam in best_hparams.items()
        ]

        print(f"Epoch: [{epoch}] Best validation scores:\n{json.dumps(best_scores)}")
        print(f"Epoch: [{epoch}] Best validation hparams:\n{json.dumps(best_hparams)}")

        ut.save_model(args, epoch, model_without_ddp, optimizer, loss_scaler)

    best_classifiers = {
        (feature_source, hparam): classifiers[feature_source, hparam]
        for feature_source, hparam in best_hparams.items()
    }
    best_model = models_probe.ClassificationWrapper(backbone, best_classifiers)

    if test_loader is not None:
        print("Evaluating best models on test set")

        if args.distributed:
            best_model = torch.nn.parallel.DistributedDataParallel(
                best_model, device_ids=[args.gpu]
            )

        test_stats = evaluate(
            args,
            best_model,
            criterion,
            test_loader,
            epoch,
            device,
            args.test_dataset,
            log_classifier_keys=log_classifier_keys,
        )
        print(f"Best models test stats:\n{json.dumps(test_stats)}")

        if is_master:
            with (output_dir / "test_log.json").open("a") as f:
                print(json.dumps(test_stats), file=f)

    if is_master:
        best_checkpoint_path = output_dir / "checkpoint-best.pth"
        to_save = {
            "model": best_model.state_dict(),
            "epoch": epoch,
            "hparams": best_hparams,
            "scores": best_scores,
            "test_stats": test_stats,
            "args": OmegaConf.to_container(args),
        }
        torch.save(to_save, best_checkpoint_path)

    if args.distributed:
        torch.distributed.destroy_process_group()

    total_time = time.monotonic() - start_time
    print(f"done! training time: {datetime.timedelta(seconds=int(total_time))}")


def create_data_loaders(args: DictConfig):
    data_loaders = {}
    dataset_names = [args.train_dataset, args.val_dataset, args.test_dataset] + args.eval_datasets

    transform = flat_data.make_flat_transform(
        img_size=args.img_size,
        clip_vmax=args.clip_vmax,
        normalize=args.normalize,
        target_id_map=args.target_id_map,
        target_key=args.target_key,
    )

    for dataset_name in dataset_names:
        if dataset_name is None or dataset_name in data_loaders:
            continue

        dataset_config = args.datasets[dataset_name].copy()
        print(f"loading dataset: {dataset_name}\n\n{OmegaConf.to_yaml(dataset_config)}")

        # only support pre-clipped datasets
        dataset = flat_data.FlatClipsDataset(dataset_config.root, transform=transform)

        # subset split
        split_range = dataset_config.get("split_range")
        if split_range is not None:
            split_start, split_stop = split_range
            if isinstance(split_stop, float):
                split_start = int(split_start * len(dataset))
                split_stop = int(split_stop * len(dataset))
            shuffle_seed = dataset_config.get("shuffle_seed", 42)
            rng = np.random.default_rng(shuffle_seed)
            sample_order = rng.permutation(len(dataset))
            split_indices = sample_order[split_start:split_stop]
            print(f"split indices: {split_indices[:10].tolist()}")
            dataset = Subset(dataset, split_indices)

        if args.distributed:
            sampler = DistributedSampler(dataset, shuffle=dataset_config.shuffle)
        else:
            sampler = None
        shuffle = sampler is None and dataset_config.shuffle

        # The pretrain script uses WebLoader, but for our pre-clipped data,
        # the standard PyTorch DataLoader is more direct and easier to debug.
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=args.num_workers,
            collate_fn=masking.mask_collate,  # needed to pad img mask
            pin_memory=True,
            drop_last=True,  # nb, changes the dataset, prob shouldn't do
        )

        data_loaders[dataset_name] = loader

    train_loader = data_loaders.pop(args.train_dataset)
    test_loader = data_loaders.pop(args.test_dataset) if args.test_dataset else None
    return train_loader, test_loader, data_loaders


@torch.no_grad()
def get_embedding_shapes(
    backbone: nn.Module,
    representations: list[str],
    loader: Iterable,
    device: torch.device,
):
    print("running backbone on example batch to get embedding shapes")
    example_batch = next(iter(loader))
    example_batch = ut.send_data(example_batch, device)
    images = example_batch["image"]
    img_mask = example_batch.get("img_mask")

    cls_token, object_tokens, patch_tokens = backbone.forward_embedding(images, mask=img_mask)
    backbone_out = models_probe.pool_representations(
        cls_token, object_tokens, patch_tokens, representations
    )

    embedding_shapes = {k: tuple(v.shape[1:]) for k, v in backbone_out.items()}
    return embedding_shapes


def make_classifiers(args: DictConfig, embedding_shapes: dict[str, tuple[int, ...]]):
    # create sweep of classifier heads with varying input features,
    # lr scales, weight decays.
    all_classifiers = {}
    param_groups = {}

    base_weight_decay = args.weight_decay

    for feature_source in args.representations:
        embed_shape = embedding_shapes[feature_source]
        assert len(embed_shape) in {1, 2}

        if len(embed_shape) == 1:
            clf_fn = partial(models_probe.LinearClassifier, embed_shape[-1], args.num_classes)
        else:
            clf_fn = partial(
                models_probe.AttnPoolClassifier,
                embed_shape[-1],
                args.num_classes,
                embed_dim=args.get("attn_pool_embed_dim"),
            )

        for lr_scale, weight_decay in product(args.lr_scale_grid, args.weight_decay_grid):
            clf = clf_fn()
            all_classifiers[(feature_source, (lr_scale, weight_decay))] = clf

            for name, param in clf.named_parameters():
                param_weight_decay = 0.0 if "bias" in name else weight_decay

                if (lr_scale, param_weight_decay) not in param_groups:
                    param_groups[lr_scale, param_weight_decay] = {
                        "params": [],
                        "lr_multiplier": lr_scale,
                        "wd_multiplier": param_weight_decay / base_weight_decay,
                    }

                param_groups[lr_scale, param_weight_decay]["params"].append(param)

    param_groups = list(param_groups.values())
    return all_classifiers, param_groups


def train_one_epoch(
    args: DictConfig,
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    loss_scaler: torch.GradScaler | None,
    lr_schedule: Sequence[float],
    epoch: int,
    device: torch.device,
    log_classifier_keys: list[tuple[str, tuple[float, float]]],
):
    model.train()

    metric_logger = ut.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", ut.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("grad", ut.SmoothedValue())
    header = f"Train: [{epoch}]"
    log_wandb = args.wandb and ut.is_main_process()

    epoch_num_batches = len(data_loader)
    steps_per_epoch = epoch_num_batches // args.accum_iter

    use_cuda = device.type == "cuda"
    if use_cuda and args.presend_cuda:
        data_loader = ut.pre_send_to_cuda_wrapper(data_loader, device)

    print_freq = args.get("print_freq", 20) if not args.debug else 1
    num_batches = epoch_num_batches if not args.debug else 10

    model_without_ddp = model.module if args.distributed else model
    classifier_keys = model_without_ddp.classifier_keys
    clf_key_to_idx = {key: ii for ii, key in enumerate(classifier_keys)}
    num_classifiers = len(classifier_keys)

    amp_dtype = getattr(torch, args.amp_dtype)

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, total_steps=num_batches)
    ):
        if use_cuda and not args.presend_cuda:
            batch = ut.send_data(batch, device)

        batch_step = batch_idx + 1
        global_step = epoch * steps_per_epoch + batch_step // args.accum_iter
        lr = lr_schedule[global_step]
        need_update = batch_step % args.accum_iter == 0

        if need_update:
            ut.update_lr(optimizer.param_groups, lr)

        images = batch["image"]
        img_mask = batch.get("img_mask")
        target = batch["target"]

        # handle single target regression
        # predictions are always shape [batch_size, num_targets, num_classifiers], so
        # need to match second dimension.
        if args.task == "regression" and target.ndim == 1:
            target = target.unsqueeze(-1)

        # expand last dimension of target to match prediction
        # note that the num_classifiers dimension has to go at the end bc this is
        # what nn.CrossEntropyLoss expects.
        expand_shape = target.ndim * (-1,) + (num_classifiers,)
        target = target.unsqueeze(-1).expand(*expand_shape)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp):
            pred = model(images, mask=img_mask)
            # [batch, num_classifiers] or [batch, num_targets, num_classifiers]
            all_loss = criterion(pred, target)
            all_loss = all_loss.reshape(-1, num_classifiers).mean(dim=0)
            loss = all_loss.mean()

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Loss is {loss_value}, stopping training")

        grad_norm = ut.backward_step(
            loss / args.accum_iter,
            optimizer,
            scaler=loss_scaler,
            need_update=need_update,
            max_norm=args.clip_grad,
        )

        all_loss_values = all_loss.detach().cpu().numpy()
        log_loss_dict = {
            f"loss_{key[0]}": all_loss_values[clf_key_to_idx[key]] for key in log_classifier_keys
        }
        metric_logger.update(loss=loss_value)
        metric_logger.update(**log_loss_dict)
        if need_update:
            metric_logger.update(lr=lr)
            grad_norm_value = grad_norm.item()
            metric_logger.update(grad=grad_norm_value)

        if log_wandb:
            log_stats = {
                "train/loss": loss_value,
                **{f"train/{k}": v for k, v in log_loss_dict.items()},
            }
            if need_update:
                log_stats.update({"train/lr": lr, "train/grad": grad_norm_value})
            wandb.log(log_stats, step=int(1000 * (epoch + batch_step / epoch_num_batches)))

        if use_cuda:
            torch.cuda.synchronize()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {f"train/{k}": meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.inference_mode()
def evaluate(
    args: DictConfig,
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    epoch: int,
    device: torch.device,
    eval_name: str,
    log_classifier_keys: list[tuple[str, tuple[float, float]]],
):
    """
    log_classifier_keys: list of (feature_source, hparams) keys for the classifiers to log.
    """
    model.eval()

    metric_logger = ut.MetricLogger(delimiter="  ")
    header = f"Eval ({eval_name}): [{epoch}]"
    log_wandb = args.wandb and ut.is_main_process()

    epoch_num_batches = len(data_loader)

    use_cuda = device.type == "cuda"
    if use_cuda and args.presend_cuda:
        data_loader = ut.pre_send_to_cuda_wrapper(data_loader, device)

    print_freq = args.get("print_freq", 20) if not args.debug else 1
    num_batches = epoch_num_batches if not args.debug else 10

    model_without_ddp = model.module if args.distributed else model
    classifier_keys = model_without_ddp.classifier_keys
    clf_key_to_idx = {key: ii for ii, key in enumerate(classifier_keys)}
    num_classifiers = len(classifier_keys)

    all_meters = defaultdict(ut.SmoothedValue)

    amp_dtype = getattr(torch, args.amp_dtype)

    for batch_idx, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, total_steps=num_batches)
    ):
        if use_cuda and not args.presend_cuda:
            batch = ut.send_data(batch, device)

        images = batch["image"]
        img_mask = batch.get("img_mask")
        target = batch["target"]

        if args.task == "regression" and target.ndim == 1:
            target = target.unsqueeze(-1)
        expand_shape = target.ndim * (-1,) + (num_classifiers,)
        target = target.unsqueeze(-1).expand(*expand_shape)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp):
            pred = model(images, mask=img_mask)
            # [batch, num_classifiers] or [batch, num_targets, num_classifiers]
            all_loss = criterion(pred, target)
            all_loss = all_loss.reshape(-1, num_classifiers).mean(dim=0)
            loss = all_loss.mean()

        loss_value = loss.item()
        metric_logger.update(loss=loss_value)

        all_loss_values = all_loss.detach().cpu().numpy()
        if args.task == "classification":
            all_acc1_values = [
                accuracy(pred[:, :, ii], target[:, ii])[0].item() for ii in range(num_classifiers)
            ]

        for ii, key in enumerate(classifier_keys):
            fmt_key = format_clf_key(key)
            all_meters[f"loss_{fmt_key}"].update(all_loss_values[ii])
            if args.task == "classification":
                all_meters[f"acc1_{fmt_key}"].update(all_acc1_values[ii])

        log_metric_dict = {}
        for feature_source, hparam in log_classifier_keys:
            idx = clf_key_to_idx[(feature_source, hparam)]
            log_metric_dict[f"loss_{feature_source}"] = all_loss_values[idx]
            if args.task == "classification":
                log_metric_dict[f"acc1_{feature_source}"] = all_acc1_values[idx]

        metric_logger.update(**log_metric_dict)

        if use_cuda:
            torch.cuda.synchronize()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    for meter in all_meters.values():
        meter.synchronize_between_processes()
    print(f"Averaged stats ({eval_name}):", metric_logger)
    stats = {f"eval/{eval_name}/{k}": meter.global_avg for k, meter in metric_logger.meters.items()}

    if log_wandb:
        wandb.log(stats, step=1000 * (epoch + 1))

    stats.update({f"eval/{eval_name}/{k}": meter.global_avg for k, meter in all_meters.items()})
    return stats


def format_clf_key(key: tuple[str, tuple[float, float]]) -> str:
    feature_source, (lr, weight_decay) = key
    return f"{feature_source}_{lr:.1e}_{weight_decay:.1e}"


def get_best_classifiers(
    args: DictConfig,
    model: models_probe.ClassificationWrapper,
    eval_stats: dict[str, float],
):
    val_scores = defaultdict(list)
    clf_hparams = defaultdict(list)
    prefix = f"eval/{args.val_dataset}"
    for key in model.classifier_keys:
        feature_source, hparam = key
        if args.task == "classification":
            score = eval_stats[f"{prefix}/acc1_{format_clf_key(key)}"]
        else:
            score = 1 - eval_stats[f"{prefix}/loss_{format_clf_key(key)}"]
        val_scores[feature_source].append(score)
        clf_hparams[feature_source].append(hparam)

    best_scores = {}
    best_hparams = {}
    for feature_source in val_scores:
        best_idx = np.argmax(val_scores[feature_source])
        best_score = val_scores[feature_source][best_idx]
        best_hparam = clf_hparams[feature_source][best_idx]
        best_scores[feature_source] = best_score
        best_hparams[feature_source] = best_hparam

    return best_scores, best_hparams


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
