import argparse
import datetime
import json
import math
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data.distributed import DistributedSampler
from webdataset import WebLoader

import data.flat_data as flat_data
import flat_mae.models_mae as models_mae
import flat_mae.utils as ut

from simclr.data import SimCLRTransform, simclr_collate
from simclr.loss import nt_xent_loss, simsiam_loss
from simclr.models import ContrastiveModel

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
        if args.wandb:
            wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=args.name,
                notes=args.notes,
                config=OmegaConf.to_container(args),
            )

    ut.setup_for_distributed(log_path=output_dir / "log.txt")
    print(f"Pre-training with contrastive mode: {args.model.contrastive_mode}")
    print("Config:", OmegaConf.to_yaml(args), sep="\n")

    train_loader, eval_loaders, samplers = make_data_loaders(args)

    print(f"Creating backbone: {args.model.backbone_name}")
    model_constructor_args = {
        "img_size": args.data.img_size,
        "in_chans": args.data.in_chans,
        "patch_size": args.data.patch_size,
        "num_frames": args.data.num_frames,
        "t_patch_size": args.data.t_patch_size,
        **args.model.backbone_kwargs,
    }
    backbone = BACKBONE_MODELS_DICT[args.model.backbone_name](**model_constructor_args)
    backbone_embed_dim = backbone.encoder.pos_embed.embed_dim
    print(f"Backbone created with embedding dimension: {backbone_embed_dim}")

    model = ContrastiveModel(
        backbone=backbone,
        mode=args.model.contrastive_mode,
        embed_dim=backbone_embed_dim,
        model_kwargs=args.model.get("head_kwargs"),
    ).to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    print("Model:", model_without_ddp, sep="\n")

    total_batch_size = args.optim.batch_size * args.optim.accum_iter * world_size
    if not args.optim.get("lr"):
        args.optim.lr = args.optim.base_lr * total_batch_size / 256
    print(f"Effective batch size: {total_batch_size}")
    print(f"Learning rate: {args.optim.lr:.2e}")

    param_groups = ut.get_param_groups(model)
    optimizer = torch.optim.AdamW(param_groups, lr=args.optim.lr, betas=tuple(args.optim.betas))
    loss_scaler = ut.GradScaler() if args.amp and args.amp_dtype != 'bfloat16' else None

    epoch_num_batches = len(train_loader)
    steps_per_epoch = epoch_num_batches // args.optim.accum_iter
    total_steps = args.optim.epochs * steps_per_epoch
    warmup_steps = args.optim.warmup_epochs * steps_per_epoch
    start_lr_val = args.optim.get("start_warmup_lr", 0.0)
    lr_schedule = ut.WarmupThenCosine(
        base_value=args.optim.lr,
        final_value=args.optim.min_lr,
        total_iters=total_steps,
        warmup_iters=warmup_steps,
        start_warmup_value=start_lr_val
    )

    ut.load_model(args, model_without_ddp, optimizer, loss_scaler)

    print(f"Start training for {args.optim.epochs} epochs")
    start_time = time.monotonic()
    for epoch in range(args.start_epoch, args.optim.epochs):
        if args.distributed and samplers.get(args.train_dataset) is not None:
            samplers[args.train_dataset].set_epoch(epoch)

        train_stats = train_one_epoch(args, model, train_loader, optimizer, loss_scaler, lr_schedule, epoch, device)

        eval_stats = {}
        for name, loader in eval_loaders.items():
            if args.distributed and samplers.get(name) is not None:
                samplers[name].set_epoch(epoch)
            stats = evaluate(args, model, loader, epoch, device, eval_name=name)
            eval_stats.update(stats)

        merged_stats = {"epoch": epoch, **train_stats, **eval_stats}
        if is_master:
            with (output_dir / "log.json").open("a") as f:
                f.write(json.dumps(merged_stats) + "\n")

            ut.save_model(args, epoch, model_without_ddp, optimizer, loss_scaler)

    total_time = time.monotonic() - start_time
    print(f"Done! Training time: {datetime.timedelta(seconds=int(total_time))}")


def make_data_loaders(args: DictConfig):
    base_transform = flat_data.make_flat_transform(
        img_size=args.data.img_size,
        clip_vmax=args.data.get("clip_vmax"),
        normalize=args.data.get("normalize"),
        random_crop=False,
        crop_kwargs=args.data.get("crop_kwargs"),
    )

    train_base_transform = flat_data.make_flat_transform(
        img_size=args.data.img_size,
        clip_vmax=args.data.get("clip_vmax"),
        normalize=args.data.get("normalize"),
        random_crop=args.data.get("random_crop", False),
        crop_kwargs=args.data.get("crop_kwargs"),
    )

    train_transform = SimCLRTransform(train_base_transform)
    eval_transform = SimCLRTransform(base_transform)

    data_loaders = {}
    samplers = {}
    world_size = ut.get_world_size()
    all_dataset_names = [args.train_dataset] + args.eval_datasets

    for dataset_name in all_dataset_names:
        if not dataset_name: continue

        is_train = dataset_name == args.train_dataset
        transform = train_transform if is_train else eval_transform
        dataset_config = args.datasets[dataset_name].copy()
        print(f"Loading dataset: {dataset_name} (is_train={is_train})\n\n{OmegaConf.to_yaml(dataset_config)}")
        dataset_type = dataset_config.pop("type")

        if dataset_type == "flat-wds":
            samples_per_epoch = dataset_config.pop("samples_per_epoch")
            dataset = flat_data.make_flat_wds_dataset(num_frames=args.data.num_frames, **dataset_config).map(transform)
            sampler = None
            shuffle = False
        elif dataset_type == "flat-clips":
            dataset = flat_data.FlatClipsDataset(dataset_config.root, transform=transform)
            samples_per_epoch = len(dataset)
            sampler = DistributedSampler(dataset, shuffle=dataset_config.shuffle, drop_last=True) if args.distributed else None
            shuffle = sampler is None and dataset_config.shuffle
        else:
            raise ValueError(f"Unknown dataset type {dataset_type}.")

        loader = WebLoader(dataset, batch_size=args.optim.batch_size, collate_fn=simclr_collate, sampler=sampler,
                           shuffle=shuffle, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        num_batches = samples_per_epoch // (world_size * args.optim.batch_size)
        loader = loader.with_epoch(num_batches).with_length(num_batches)
        data_loaders[dataset_name] = loader
        samplers[dataset_name] = sampler

    train_loader = data_loaders.pop(args.train_dataset)
    eval_loaders = data_loaders
    return train_loader, eval_loaders, samplers


def train_one_epoch(args, model, data_loader, optimizer, loss_scaler, lr_schedule, epoch, device):
    model.train()
    metric_logger = ut.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", ut.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("grad", ut.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    header = f'Train: [{epoch}]'
    log_wandb = args.wandb and ut.is_main_process()

    epoch_num_batches = len(data_loader)
    steps_per_epoch = epoch_num_batches // args.optim.accum_iter

    print_freq = args.get("print_freq", 100) if not args.debug else 1
    num_batches_to_log = epoch_num_batches if not args.debug else 10

    amp_dtype = getattr(torch, args.amp_dtype)
    optimizer.zero_grad()

    for batch_idx, (batch_view_1, batch_view_2) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, total_steps=num_batches_to_log)
    ):
        if batch_idx >= num_batches_to_log:
            break

        batch_step = batch_idx + 1
        global_step = epoch * steps_per_epoch + batch_step // args.optim.accum_iter
        lr = lr_schedule[global_step]
        need_update = batch_step % args.optim.accum_iter == 0

        if need_update:
            ut.update_lr(optimizer.param_groups, lr)

        view_1 = batch_view_1['image'].to(device, non_blocking=True)
        view_2 = batch_view_2['image'].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp):
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
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Loss is {loss_value}, stopping training")

        grad_norm = ut.backward_step(
            loss / args.optim.accum_iter,
            optimizer,
            scaler=loss_scaler,
            need_update=need_update,
            max_norm=args.optim.get("clip_grad")
        )

        metric_logger.update(loss=loss_value)
        if need_update:
            metric_logger.update(lr=lr)
            if grad_norm is not None:
                grad_norm_value = grad_norm.item()
                metric_logger.update(grad=grad_norm_value)

        if log_wandb:
            log_stats = {"train/loss": loss_value}
            if need_update:
                log_stats.update({"train/lr": lr})
                if grad_norm is not None:
                    log_stats.update({"train/grad": grad_norm_value})
            wandb.log(log_stats, step=int(1000 * (epoch + batch_step / epoch_num_batches)))

        if device.type == "cuda":
            torch.cuda.synchronize()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {f"train/{k}": meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(args, model, data_loader, epoch, device, eval_name):
    model.eval()
    metric_logger = ut.MetricLogger(delimiter="  ")
    header = f"Eval ({eval_name}): [{epoch}]"
    log_wandb = args.wandb and ut.is_main_process()

    amp_dtype = getattr(torch, args.amp_dtype)

    for batch_view_1, batch_view_2 in metric_logger.log_every(data_loader, 100, header):
        view_1 = batch_view_1['image'].to(device, non_blocking=True)
        view_2 = batch_view_2['image'].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp):
            outputs = model(view_1, view_2, mask_ratio=args.model.mask_ratio)
            if args.model.contrastive_mode == "simclr":
                z1, z2 = outputs
                loss = nt_xent_loss(z1, z2, temperature=args.model.get("temperature", 0.5), distributed=args.distributed)
            elif args.model.contrastive_mode == "simsiam":
                p1, z2, p2, z1 = outputs
                loss = simsiam_loss(p1, z2, p2, z1)
            else:
                raise ValueError(f"Unknown contrastive mode: {args.model.contrastive_mode}")

        metric_logger.update(loss=loss.item())

    metric_logger.synchronize_between_processes()
    print(f"Averaged stats for {eval_name}:", metric_logger)
    stats = {f"eval/{eval_name}/{k}": meter.global_avg for k, meter in metric_logger.meters.items()}

    if log_wandb:
        wandb.log(stats, step=int(1000 * (epoch + 1)))

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SimCLR/SimSiam Pre-training", add_help=False)
    parser.add_argument("--cfg-path", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--overrides", type=str, default=None, nargs="+", help="Modify config options from command line.")
    cli_args = parser.parse_args()

    cfg = OmegaConf.load(cli_args.cfg_path)
    if cli_args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(cli_args.overrides))

    main(cfg)