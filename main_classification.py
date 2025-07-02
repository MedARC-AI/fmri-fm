# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE-st: https://github.com/facebookresearch/mae_st
# CAPI: https://github.com/facebookresearch/capi
# --------------------------------------------------------
import argparse
import datetime
import json
import math
import os
import random
import time
from collections import defaultdict
from functools import partial
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import wandb
from omegaconf import DictConfig, OmegaConf
from timm.utils import accuracy
from torch import Tensor
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from webdataset import WebLoader as DataLoader

import models_mae
import models_mae_linear
import util.lr_sched as lr_sched
import util.misc as misc
from flat_data import FlatClipsDataset, make_flat_wds_dataset, make_flat_transform
from util.misc import NativeScalerWithGradNormCount as NativeScaler

PROJECT = "fMRI-foundation-model"

DEFAULT_CONFIG = Path(__file__).parent / "config/default_classification.yaml"

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

    print("classification train/eval fmri-fm")
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print(misc.get_sha())
    print("config:", OmegaConf.to_yaml(args), sep="\n")

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

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # loading datasets
    data_loaders, samplers, total_num_batches = make_data_loaders(args)

    train_ds_name = args.get("train_dataset_name", "train")
    val_ds_name = args.get("val_dataset_name", "val")
    test_ds_name = args.get("test_dataset_name", "test")

    data_loader_train = data_loaders[train_ds_name]
    num_batches_train = total_num_batches[train_ds_name]

    if test_ds_name in data_loaders:
        data_loader_test = data_loaders[test_ds_name]
        num_batches_test = total_num_batches[test_ds_name]
    else:
        data_loader_test = None

    data_loaders_eval = data_loaders.copy()
    data_loaders_eval.pop(train_ds_name)
    data_loaders_eval.pop(test_ds_name, None)

    # define the backbone model
    print(f"loading backbone model: {args.model}")
    backbone = MODELS_DICT[args.model](**args)

    if args.checkpoint:
        print(f"loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        backbone.load_state_dict(checkpoint["model"])

    if not args.finetune:
        print("freezing backbone model")
        backbone.requires_grad_(False)
        backbone_param_groups = []
    else:
        backbone_param_groups = misc.add_weight_decay(backbone, args.weight_decay)

    backbone.to(device)
    embedding_shapes = get_embedding_shapes(
        backbone, args.representations, data_loader_train, device
    )
    print(f"embedding feature shapes:\n{embedding_shapes}")

    print("initializing sweep of classifier heads")
    classifiers, classifier_param_groups = make_classifiers(args, embedding_shapes)

    model = ClassificationWrapper(backbone, classifiers)
    model.to(device)

    print(f"Model:\n{model}")

    num_params = sum(p.numel() for p in model.parameters())
    num_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Num params (train): {num_params / 1e6:.1f}M ({num_params_train / 1e6:.1f}M)")

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    args.lr = args.base_lr * eff_batch_size / 256
    print("base lr: %.2e" % args.base_lr)
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()],
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    param_groups = backbone_param_groups + classifier_param_groups
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=(0.9, 0.95) if not args.get("beta") else tuple(args.beta),
    )
    loss_scaler = NativeScaler(fp32=args.fp32)

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    # watch middle classifiers on first epoch
    # for later epochs will update to the best
    mid_lr_scale = args.lr_scale_grid[len(args.lr_scale_grid) // 2]
    mid_weight_decay = args.weight_decay_grid[len(args.weight_decay_grid) // 2]
    log_classifier_keys = {
        feature_source: (mid_lr_scale, mid_weight_decay)
        for feature_source in args.representations
    }

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
            log_classifier_keys=log_classifier_keys,
        )

        eval_stats = {}
        for dataset_name, data_loader_eval in data_loaders_eval.items():
            ds_stats = evaluate(
                model,
                data_loader_eval,
                dataset_name,
                device,
                epoch,
                args=args,
                fp32=args.fp32,
                num_batches=total_num_batches[dataset_name],
                log_wandb=args.wandb,
                log_classifier_keys=log_classifier_keys,
            )
            eval_stats[dataset_name] = ds_stats

        if args.output_dir and (
            epoch % args.checkpoint_period == 0 or epoch + 1 == args.epochs or args.debug
        ):
            checkpoint_path = misc.save_model(
                args=args,
                model=None,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )
            misc.cleanup_checkpoints(args)

        log_stats = {
            **{f"train__{k}": v for k, v in train_stats.items()},
            **{
                f"eval__{ds_name}__{k}": v for ds_name, ds_stats in eval_stats.items()
                for k, v in ds_stats.items()
            },
            "epoch": epoch,
        }

        if args.output_dir and misc.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                print(json.dumps(log_stats), file=f)

        best_scores, best_hparams = get_best_classifiers(
            model_without_ddp, eval_stats[val_ds_name]
        )

        # log the best classifiers
        log_classifier_keys = best_hparams
        
        print(f"Epoch: [{epoch}] Best validation scores:\n{json.dumps(best_scores)}")
        print(f"Epoch: [{epoch}] Best validation hparams:\n{json.dumps(best_hparams)}")

        if args.debug:
            break

    best_classifiers = {
        (feature_source, hparam): classifiers[feature_source, hparam]
        for feature_source, hparam in best_hparams.items()
    }
    best_model = ClassificationWrapper(backbone, best_classifiers)

    if args.output_dir and misc.is_main_process():
        best_checkpoint_path = f"{args.output_dir}/checkpoint-best.pth"
        to_save = {
            "model": best_model.state_dict(),
            "epoch": epoch,
            "hparams": best_hparams,
            "scores": best_scores,
            "args": OmegaConf.to_container(args),
        }
        misc.save_on_master(to_save, best_checkpoint_path)

    if data_loader_test is not None:
        print("Evaluating best models on test set")

        if args.distributed:
            best_model = torch.nn.parallel.DistributedDataParallel(
                best_model, device_ids=[torch.cuda.current_device()],
            )

        test_stats = evaluate(
            best_model,
            data_loader_test,
            test_ds_name,
            device,
            epoch,
            args=args,
            fp32=args.fp32,
            num_batches=num_batches_test,
            log_wandb=args.wandb,
            log_classifier_keys=log_classifier_keys,
        )
        print(f"Best models test stats:\n{json.dumps(test_stats)}")

        if args.output_dir and misc.is_main_process():
            with (Path(args.output_dir) / "test_log.txt").open("a") as f:
                print(json.dumps(test_stats), file=f)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    print(torch.cuda.memory_allocated())
    return [checkpoint_path]


def make_data_loaders(args: DictConfig):
    transform = make_flat_transform(clip_vmax=args.clip_vmax, normalize=args.normalize)

    data_loaders = {}
    samplers = {}
    total_num_batches = {}

    for dataset_name, dataset_config in args.datasets.items():
        print(f"dataset: {dataset_name}\n\n{OmegaConf.to_yaml(dataset_config)}")

        # only support pre-clipped datasets
        dataset = FlatClipsDataset(dataset_config.root, transform=transform)

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
            split_indices = sample_order[split_start: split_stop]
            print(f"split indices: {split_indices[:10].tolist()}")
            dataset = Subset(dataset, split_indices)

        if args.distributed:
            sampler = DistributedSampler(dataset, shuffle=dataset_config.shuffle)
        else:
            sampler = None
        samples_per_epoch = len(dataset)
        shuffle = sampler is None and dataset_config.shuffle

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


@torch.no_grad()
def get_embedding_shapes(
    backbone: nn.Module,
    representations: list[str],
    loader: Iterable,
    device: torch.device
):
    print("running backbone on example batch to get embedding shapes")
    example_batch = next(iter(loader))
    samples = example_batch["image"].to(device)
    img_mask = example_batch["mask"].to(device)

    cls_token, object_tokens, patch_tokens = backbone.forward_embedding(
        samples, img_mask=img_mask
    )
    backbone_out = pool_representations(
        cls_token, object_tokens, patch_tokens, representations
    )

    embedding_shapes = {
        k: tuple(v.shape[1:]) for k, v in backbone_out.items()
    }
    return embedding_shapes


def make_classifiers(
    args: DictConfig, embedding_shapes: dict[str, tuple[int, ...]]
):
    # create sweep of classifier heads with varying input features,
    # lr scales, weight decays.
    all_classifiers = {}
    param_groups = {}

    for feature_source in args.representations:
        embed_shape = embedding_shapes[feature_source]
        assert len(embed_shape) in {1, 2}

        if len(embed_shape) == 1:
            clf_fn = partial(LinearClassifier, embed_shape[-1], args.num_classes)
        else:
            clf_fn = partial(AttnPoolClassifier, embed_shape[-1], args.num_classes)

        for lr_scale, weight_decay in product(args.lr_scale_grid, args.weight_decay_grid):
            # TODO: should they all get the same init?
            clf = clf_fn()
            all_classifiers[(feature_source, (lr_scale, weight_decay))] = clf

            for name, param in clf.named_parameters():
                param_weight_decay = 0.0 if "bias" in name else weight_decay

                if (lr_scale, param_weight_decay) not in param_groups:
                    param_groups[lr_scale, param_weight_decay] = {
                        "params": [], "lr_scale": lr_scale, "weight_decay": weight_decay
                    }

                param_groups[lr_scale, param_weight_decay]["params"].append(param)

    param_groups = list(param_groups.values())
    return all_classifiers, param_groups


def train_one_epoch(
    model: "ClassificationWrapper",
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    args=None,
    fp32=False,
    num_batches=None,
    log_wandb=False,
    log_classifier_keys=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "gpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Train: Epoch: [{}]".format(epoch)
    print_freq = 1 if args.debug else 20
    debug_steps = 10 * args.accum_iter
    log_wandb = misc.is_main_process() and log_wandb

    model_without_ddp = model.module if args.distributed else model
    classifier_keys = model_without_ddp.classifier_keys
    clf_key_to_idx = {key: ii for ii, key in enumerate(classifier_keys)}

    if log_classifier_keys is None:
        log_classifier_keys = {}

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
        # TODO: classifiers need to set lr_scale, and should be set appropriately to
        # match capi param grid
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / num_batches + epoch, args
            )

        samples = batch["image"]
        img_mask = batch["mask"]
        target = batch["target"]

        samples = samples.to(device, non_blocking=True)
        img_mask = img_mask.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=not fp32):
            # TODO: capi does not do amp. is there a reason? is amp usually not done
            # during linear probe?
            all_logit = model(samples, img_mask=img_mask)
            all_loss = F.cross_entropy(
                all_logit,
                target.unsqueeze(-1).expand(-1, all_logit.shape[-1]),  # [B, num_classifiers]
                reduction="none",
            ).mean(dim=0)  # [num_classifiers]
            loss = all_loss.mean()

        loss_value = misc.all_reduce_mean(loss.detach()).item()
        all_loss_values = misc.all_reduce_mean(all_loss.detach()).tolist()

        if not math.isfinite(loss_value):
            raise Exception("Loss is {}, stopping training".format(loss_value))

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
            clip_grad=args.clip_grad,
        )

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        log_loss_dict = {
            f"loss_{key[0]}": all_loss_values[clf_key_to_idx[key]]
            for key in log_classifier_keys.items()
        }
        metric_logger.update(**log_loss_dict)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(gpu_mem=misc.gpu_mem_usage())

        # TODO: do we want to log the sweep of lr values?
        # it could be useful for debugging. but it is a lot of values.

        if log_wandb and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(
                (data_iter_step / num_batches + epoch) * 1000
            )
            log_stats = {
                "train/loss": loss_value,
                **{f"train/{k}": v for k, v in log_loss_dict.items()}
            }
            wandb.log(log_stats, step=epoch_1000x)

        if args.debug and (data_iter_step + 1) >= debug_steps:
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: "ClassificationWrapper",
    data_loader: Iterable,
    eval_name: str,
    device: torch.device,
    epoch: int,
    args=None,
    fp32=False,
    num_batches=None,
    log_wandb=False,
    log_classifier_keys=None,
):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f"Eval ({eval_name}): [{epoch}]"
    print_freq = 1 if args.debug else 50
    debug_steps = 10
    log_wandb = misc.is_main_process() and log_wandb

    model_without_ddp = model.module if args.distributed else model
    classifier_keys = model_without_ddp.classifier_keys
    clf_key_to_idx = {key: ii for ii, key in enumerate(classifier_keys)}

    all_meters = defaultdict(misc.SmoothedValue)

    if log_classifier_keys is None:
        log_classifier_keys = {}

    if num_batches is None:
        num_batches = len(data_loader)

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(
            data_loader, print_freq, header, total_steps=num_batches
        )
    ):
        samples = batch["image"]
        img_mask = batch["mask"]
        target = batch["target"]

        samples = samples.to(device, non_blocking=True)
        img_mask = img_mask.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=not fp32):
            all_logit = model(samples, img_mask=img_mask)
            all_loss = F.cross_entropy(
                all_logit,
                target.unsqueeze(-1).expand(-1, all_logit.shape[-1]),  # [B, num_classifiers]
                reduction="none",
            ).mean(dim=0)  # [num_classifiers]
            loss = all_loss.mean()

        loss_value = loss.item()
        metric_logger.update(loss=loss_value)

        all_loss_values = all_loss.tolist()
        all_acc1_values = [
            accuracy(all_logit.detach()[..., ii], target)[0].item()
            for ii in range(all_logit.shape[-1])
        ]

        for ii, key in enumerate(classifier_keys):
            fmt_key = format_clf_key(key)
            all_meters[f"loss_{fmt_key}"].update(all_loss_values[ii])
            all_meters[f"acc1_{fmt_key}"].update(all_acc1_values[ii])
        
        for feature_source, hparam in log_classifier_keys.items():
            idx = clf_key_to_idx[(feature_source, hparam)]
            metric_logger.update(
                **{
                    f"loss_{feature_source}": all_loss_values[idx],
                    f"acc1_{feature_source}": all_acc1_values[idx],
                }
            )

        if args.debug and (data_iter_step + 1) >= debug_steps:
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    for meter in all_meters.values():
        meter.synchronize_between_processes()
    print(f"Averaged stats ({eval_name}):", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if log_wandb:
        # eval at the end of training, so epoch + 1
        epoch_1000x = int((epoch + 1) * 1000)
        wandb.log({f"eval/{eval_name}/{k}": v for k, v in stats.items()}, step=epoch_1000x)
    
    stats.update({k: meter.global_avg for k, meter in all_meters.items()})

    return stats


def format_clf_key(key: tuple[str, tuple[float, float]]) -> str:
    feature_source, (lr, weight_decay) = key
    return f"{feature_source}_{lr:.1e}_{weight_decay:.1e}"


def get_best_classifiers(
    model: "ClassificationWrapper",
    val_stats: dict[str, float],
    metric: str = "acc1",
):
    val_scores = defaultdict(list)
    clf_hparams = defaultdict(list)
    for key in model.classifier_keys:
        feature_source, hparam = key
        score = val_stats[f"{metric}_{format_clf_key(key)}"]
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


class ClassificationWrapper(nn.Module):
    """
    Wrap a backbone embedding model together with a grid of classifier heads.

    backbone: backbone model implementing forward_embedding
    classifiers: map of (feature_source, (lr_scale, weight_decay)) -> classifier
    """
    def __init__(
        self,
        backbone: nn.Module,
        classifiers: dict[tuple[str, tuple[int, int]], nn.Module],
    ):
        super().__init__()
        self.representations = {key[0] for key in classifiers}
        self.backbone = backbone

        # can't use ModuleDict bc of restrictions of keys (must be strings, no dots).
        self.classifier_keys = list(classifiers)
        self.classifiers = nn.ModuleList(list(classifiers.values()))

    def forward(self, *args, **kwargs) -> Tensor:
        cls_token, object_tokens, patch_tokens = self.backbone.forward_embedding(
            *args, **kwargs
        )
        backbone_out = pool_representations(
            cls_token, object_tokens, patch_tokens, self.representations
        )

        all_logit = []
        for ii, (feature_source, _) in enumerate(self.classifier_keys):
            clf = self.classifiers[ii]
            all_logit.append(clf(backbone_out[feature_source]))

        # [B, num_classes, num_classifiers]
        all_logit = torch.stack(all_logit, dim=-1)
        return all_logit


class LinearClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cls_token):
        return self.linear(cls_token)


class AttnPoolClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        assert in_dim % 64 == 0
        self.query_token = nn.Parameter(torch.empty(in_dim))
        self.num_heads = in_dim // 64
        self.kv = nn.Linear(in_dim, in_dim * 2)
        self.linear = nn.Linear(in_dim, out_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.query_token, std=0.02)
        nn.init.trunc_normal_(self.kv.weight, std=0.02)
        nn.init.zeros_(self.kv.bias)
        nn.init.trunc_normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, feat_tokens):
        B, N, D = feat_tokens.shape

        q = self.query_token.expand(B, 1, -1)
        q = q.reshape(B, 1, self.num_heads, D // self.num_heads)  # [B, 1, head, D_head]
        q = q.permute(0, 2, 1, 3)  # [B, head, 1, D_head]

        kv = self.kv(feat_tokens).reshape(B, N, 2, self.num_heads, D // self.num_heads)  # [B, N, 2, head, D_head]
        kv = kv.permute(2, 0, 3, 1, 4)  # [2, B, head, N, D_head]
        k, v = torch.unbind(kv, dim=0)  # 2 * [B, head, N, D_head]

        x = F.scaled_dot_product_attention(q, k, v)  # [B, head, 1, D_head]
        x = x.reshape(B, D)  # [B, D]
        return self.linear(x)


def pool_representations(
    cls_token: Tensor | None,
    object_tokens: Tensor | None,
    patch_tokens: Tensor,
    representations: list[str],
):
    B, N, D = patch_tokens.shape

    if cls_token is not None:
        assert cls_token.shape == (B, 1, D)
        cls_token = cls_token.squeeze(1)

    if object_tokens is not None:
        R = object_tokens.shape[1]
        assert object_tokens.shape == (B, R, D)

    # Global features for the linear classifiers
    out: dict[str, Tensor] = {}
    if "cls" in representations:
        out["cls"] = cls_token  # [B, D]
    if "avg_patch" in representations:
        out["avg_patch"] = patch_tokens.mean(1)  # [B, D]
    if "cls_avg_patch" in representations:
        out["cls_avg_patch"] = torch.cat([cls_token, patch_tokens.mean(1)], dim=-1)  # [B, 2 * D]
    if "avg_objects" in representations:
        out["avg_objects"] = object_tokens.mean(1)  # [B, D]
    if "concat_objects" in representations:
        out["concat_objects"] = object_tokens.flatten(1, 2)  # [B, R * D]
    # Object features (registers) for the attention pooling classifiers
    if "objects" in representations:
        out["reg"] = object_tokens
    # Patch features for the attention pooling classifiers
    if "patch" in representations:
        out["patch"] = patch_tokens  # [B, h * w, D]
    return out


class ImageFlatten(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward_embedding(
        self,
        imgs: torch.Tensor,
        img_mask: torch.Tensor | None = None,
    ):
        N, C, T, H, W = imgs.shape
        assert C == 1
        latent = imgs.reshape(N, T, H*W)  # [N, T, D]
        return None, None, latent


MODELS_DICT["image_flatten"] = ImageFlatten


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
