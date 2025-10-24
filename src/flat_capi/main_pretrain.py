import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data.distributed import DistributedSampler
from webdataset import WebLoader as WDSLoader
from torch.utils.data import DataLoader as TorchLoader

import data.flat_data as flat_data
import flat_capi.models_capi as models_capi
from flat_capi.engine_pretrain import train_one_epoch, do_student, do_teacher
from flat_capi.schedules import WarmupThenCosine
from flat_capi.utils.misc import (
    NativeScalerWithGradNormCount as NativeScaler,
)
from flat_capi.utils import misc
from flat_capi.data import collate_data_capi


DEFAULT_CONFIG = Path(__file__).parent / "config/default_pretrain.yaml"
MODELS_DICT = {**models_capi.__dict__}

_AMP_DTYPE = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def main(args: DictConfig):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()

    if global_rank == 0 and args.wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.name,
            notes=args.notes,
            config=OmegaConf.to_container(args),
        )

    print("pretraining CAPI")
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
    seed = args.seed + misc.get_rank()
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    if getattr(args, "time_as_channels", None) is None:
        try:
            args.time_as_channels = bool(int(args.in_chans) == int(args.num_frames) and int(args.num_frames) > 1)
        except Exception:
            args.time_as_channels = False

    if getattr(args, "precision", None) is None:
        args.precision = "bf16"
    args.precision = str(args.precision).lower()
    assert args.precision in ("fp16", "bf16", "fp32"), f"Unsupported precision: {args.precision}"

    data_loader_train, sampler_train, num_batches_train = make_data_loader_train(args)

    if bool(getattr(args, "time_as_channels", False)):
        if args.in_chans != args.num_frames:
            print(
                f"[CAPI] Adjusting in_chans from {args.in_chans} to num_frames={args.num_frames} for time_as_channels=true"
            )
            args.in_chans = args.num_frames

    backbone = MODELS_DICT[args.model](**args)
    model = build_student_teacher(backbone, args).to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    num_params = sum(p.numel() for p in model_without_ddp.parameters())
    print(f"Num params = {num_params / 1e6:.1f}M")

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    args.lr = args.base_lr * eff_batch_size / 256
    print("base lr: %.2e" % args.base_lr)
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
        )
        model_without_ddp = model.module

    def get_student_param_groups(student_module: nn.Module):
        params = []
        patch_embed_lr_mult = float(getattr(args, "patch_embed_lr_mult", 0.2))
        rope_lr_mult = float(getattr(args, "rope_lr_mult", 0.0))
        layernorm_wd_mult = float(getattr(args, "layernorm_wd_mult", 0.0))
        for name, p in student_module.named_parameters():
            if not p.requires_grad:
                continue
            d = {
                "params": [p],
                "name": name,
                "lr_multiplier": 1.0,
                "wd_multiplier": 1.0,
            }
            if name.endswith(".bias") or ("gamma" in name):
                d["wd_multiplier"] = 0.0
            if ("norm" in name) and ("weight" in name):
                d["wd_multiplier"] = layernorm_wd_mult
            if "patch_embed" in name:
                d["lr_multiplier"] *= patch_embed_lr_mult
            if "rope" in name:
                d["lr_multiplier"] *= rope_lr_mult
            params.append(d)
        return params

    student_param_groups = get_student_param_groups(model_without_ddp.student)
    beta = (0.9, 0.95) if args.beta is None else tuple(args.beta)
    optimizer = torch.optim.AdamW(student_param_groups, lr=0.0, betas=beta, weight_decay=0.0)
    clustering_params = [p for n, p in model_without_ddp.named_parameters() if ("teacher.head" in n and p.requires_grad)]
    clustering_opt_cfg = getattr(getattr(args, "capi", None), "clustering_optimizer", None)
    if clustering_opt_cfg is not None:
        opt_name = getattr(clustering_opt_cfg, "name", "AdamW")
        opt_kwargs = OmegaConf.to_container(getattr(clustering_opt_cfg, "kwargs", {}), resolve=True)
        if not isinstance(opt_kwargs, dict):
            opt_kwargs = {}
        opt_kwargs.setdefault("weight_decay", 0.05)
        Cls = getattr(torch.optim, str(opt_name), torch.optim.AdamW)
        clustering_optimizer = Cls(clustering_params, **opt_kwargs)
    else:
        clustering_optimizer = torch.optim.AdamW(clustering_params, lr=0.0, betas=beta, weight_decay=0.05)

    use_grad_scaler = args.precision == "fp16"
    loss_scaler = NativeScaler(fp32=not use_grad_scaler)

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        clustering_optimizer=clustering_optimizer,
    )

    checkpoint_path = ""
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    total_iters = args.epochs * (num_batches_train)
    lr_sched_cfg = getattr(args, "lr_schedule", {})
    try:
        lr_sched_cfg = OmegaConf.to_container(lr_sched_cfg, resolve=True)
    except Exception:
        pass
    if not isinstance(lr_sched_cfg, dict):
        lr_sched_cfg = {}
    student_lr_sched = WarmupThenCosine(
        base_value=float(lr_sched_cfg.get("base_value", args.lr)),
        final_value=float(lr_sched_cfg.get("final_value", args.min_lr)),
        total_iters=total_iters,
        warmup_iters=int(lr_sched_cfg.get("warmup_iters", int(args.warmup_epochs * num_batches_train))),
        freeze_iters=int(lr_sched_cfg.get("freeze_iters", 0)),
        truncate_cos=float(lr_sched_cfg.get("truncate_cos", 1.0)),
    )
    _ms_cfg = getattr(args, "momentum_schedule", {})
    try:
        _ms_cfg = OmegaConf.to_container(_ms_cfg, resolve=True)
    except Exception:
        pass
    if not isinstance(_ms_cfg, dict):
        _ms_cfg = {}
    momentum_sched = WarmupThenCosine(
        base_value=float(_ms_cfg.get("base_value", 0.999)),
        final_value=float(_ms_cfg.get("final_value", 1.0)),
        total_iters=total_iters,
        warmup_iters=int(_ms_cfg.get("warmup_iters", int(args.warmup_epochs * num_batches_train))),
        start_warmup_value=float(_ms_cfg.get("start_warmup_value", 1.0)),
        freeze_iters=int(_ms_cfg.get("freeze_iters", 0)),
        truncate_cos=float(_ms_cfg.get("truncate_cos", 1.0)),
    )
    co_sched_cfg = getattr(getattr(getattr(args, "capi", None), "clustering_optimizer", None), "lr_schedule", None)
    cluster_lr_sched = WarmupThenCosine(
        base_value=float(getattr(co_sched_cfg, "base_value", args.lr * 0.5)),
        final_value=float(getattr(co_sched_cfg, "final_value", 0.0)),
        total_iters=total_iters,
        warmup_iters=int(getattr(co_sched_cfg, "warmup_iters", int(args.warmup_epochs * num_batches_train))),
        start_warmup_value=float(getattr(co_sched_cfg, "start_warmup_value", 0.0)),
        freeze_iters=int(getattr(co_sched_cfg, "freeze_iters", 0)),
        truncate_cos=float(getattr(co_sched_cfg, "truncate_cos", 1.0)),
    )

    st_sched_cfg = getattr(getattr(args, "capi", None), "student_temp_schedule", None)
    if st_sched_cfg is not None:
        try:
            st_sched_cfg = OmegaConf.to_container(st_sched_cfg, resolve=True)
        except Exception:
            pass
    student_temp_sched = (
        WarmupThenCosine(
            base_value=float(st_sched_cfg.get("base_value", args.capi.student_temp)),
            final_value=float(st_sched_cfg.get("final_value", args.capi.student_temp)),
            total_iters=total_iters,
            warmup_iters=int(st_sched_cfg.get("warmup_iters", 0)),
            start_warmup_value=float(st_sched_cfg.get("start_warmup_value", 0.0)),
            freeze_iters=int(st_sched_cfg.get("freeze_iters", 0)),
            truncate_cos=float(st_sched_cfg.get("truncate_cos", 1.0)),
        )
        if isinstance(st_sched_cfg, dict)
        else None
    )

    ck = getattr(getattr(args, "capi", None), "clustering_kwargs", None)
    tgt_sched_cfg = getattr(ck, "target_temp_schedule", None) if ck is not None else None
    prd_sched_cfg = getattr(ck, "pred_temp_schedule", None) if ck is not None else None

    def _build_temp_sched(cfg, default_val):
        if cfg is None:
            return None
        try:
            cfg = OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            pass
        if not isinstance(cfg, dict):
            return None
        return WarmupThenCosine(
            base_value=float(cfg.get("base_value", default_val)),
            final_value=float(cfg.get("final_value", default_val)),
            total_iters=total_iters,
            warmup_iters=int(cfg.get("warmup_iters", 0)),
            start_warmup_value=float(cfg.get("start_warmup_value", 0.0)),
            freeze_iters=int(cfg.get("freeze_iters", 0)),
            truncate_cos=float(cfg.get("truncate_cos", 1.0)),
        )

    target_temp_sched = _build_temp_sched(tgt_sched_cfg, getattr(getattr(args, "capi", None).clustering_kwargs, "target_temp", 0.1)) if ck is not None else None
    pred_temp_sched = _build_temp_sched(prd_sched_cfg, getattr(getattr(args, "capi", None).clustering_kwargs, "pred_temp", 0.2)) if ck is not None else None

    it_global = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and sampler_train is not None:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args=args,
            num_batches=num_batches_train,
            log_wandb=args.wandb,
            clustering_optimizer=clustering_optimizer,
            student_lr_sched=student_lr_sched,
            momentum_sched=momentum_sched,
            cluster_lr_sched=cluster_lr_sched,
            it_start=it_global,
            student_temp_sched=student_temp_sched,
            target_temp_sched=target_temp_sched,
            pred_temp_sched=pred_temp_sched,
        )
        it_global += num_batches_train

        if args.output_dir and (epoch % args.checkpoint_period == 0 or epoch + 1 == args.epochs):
            checkpoint_path = misc.save_model(
                args=args,
                epoch=epoch,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                clustering_optimizer=clustering_optimizer,
            )
            misc.cleanup_checkpoints(args)

        log_stats = {**{f"train__{k}": v for k, v in train_stats.items()}, "epoch": epoch}
        if misc.is_main_process():
            if args.output_dir:
                with (Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            unit = str(getattr(args, "log_metrics_unit", "step")).lower()
            cadence = int(getattr(args, "log_metrics_cadence", 1))
            if args.wandb and unit == "epoch" and ((epoch + 1) % max(cadence, 1) == 0):
                try:
                    payload = {k.replace("train__", "train/"): v for k, v in log_stats.items() if k.startswith("train__")}
                    global_step_end_epoch = int((epoch + 1) * num_batches_train)
                    wandb.log(payload, step=global_step_end_epoch, commit=True)
                except Exception:
                    pass

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    print(torch.cuda.memory_allocated())
    return [checkpoint_path]


def make_data_loader_train(args: DictConfig):
    transform = flat_data.make_flat_transform(
        img_size=args.img_size,
        clip_vmax=args.clip_vmax,
        normalize=args.normalize,
        random_crop=args.random_crop,
        crop_kwargs=args.crop_kwargs,
    )
    dataset_cfg = OmegaConf.to_container(args.datasets.train, resolve=True)
    dataset_type = dataset_cfg.pop("type")
    if dataset_type == "flat-wds":
        samples_per_epoch = int(dataset_cfg.pop("samples_per_epoch"))
        if samples_per_epoch <= 0:
            raise RuntimeError("datasets.train.samples_per_epoch must be > 0 for flat-wds")
        dataset = flat_data.make_flat_wds_dataset(num_frames=args.num_frames, **dataset_cfg)
        dataset = dataset.map(transform)
        sampler = None
        shuffle = False
    elif dataset_type == "flat-clips":
        root = dataset_cfg["root"]
        dataset = flat_data.FlatClipsDataset(root, transform=transform)
        samples_per_epoch = len(dataset)
        if samples_per_epoch == 0:
            raise RuntimeError(
                f"No .pt samples found in flat-clips root: {root}.\n"
                "- Fix the path to an existing flat-clips directory, or\n"
                "- Switch to type=flat-wds with a valid 'url' and 'samples_per_epoch'."
            )
        sampler = DistributedSampler(dataset, shuffle=dataset_cfg.get("shuffle", True)) if args.distributed else None
        shuffle = sampler is None and dataset_cfg.get("shuffle", True)
    else:
        raise ValueError(f"Unknown dataset type {dataset_type}.")

    img_h, img_w = args.img_size
    n_tokens = (img_h // args.patch_size) * (img_w // args.patch_size)

    def collate_fn(batch):
        return collate_data_capi(
            batch,
            img_h=img_h,
            img_w=img_w,
            patch_size=args.patch_size,
            mask_ratio=args.mask_ratio,
            prediction_subsampling=args.prediction_subsampling,
            dtype=_AMP_DTYPE[args.precision],
            time_as_channels=bool(getattr(args, "time_as_channels", False)),
            select_frame_index=int(getattr(args, "select_frame_index", 0)),
        )

    if dataset_type == "flat-wds":
        loader = WDSLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        num_batches = samples_per_epoch // (misc.get_world_size() * args.batch_size)
        if num_batches <= 0:
            raise RuntimeError(
                f"Computed num_batches=0 (samples_per_epoch={samples_per_epoch}, world_size={misc.get_world_size()}, batch_size={args.batch_size})."
            )
        loader = loader.with_epoch(num_batches)
        try:
            loader = loader.with_length(num_batches, silent=True)
        except Exception:
            pass
    else:
        # flat-clips: use standard DataLoader (finite dataset)
        loader = TorchLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        num_batches = samples_per_epoch // (misc.get_world_size() * args.batch_size)
        if num_batches <= 0:
            raise RuntimeError(
                f"Computed num_batches=0 (samples_per_epoch={samples_per_epoch}, world_size={misc.get_world_size()}, batch_size={args.batch_size})."
            )
    return loader, sampler, num_batches


class StudentTeacher(nn.Module):
    def __init__(self, backbone: nn.Module, args: DictConfig) -> None:
        super().__init__()
        self.student = nn.ModuleDict({})
        self.student_ema = nn.ModuleDict({})
        self.teacher = nn.ModuleDict({})
        self.student["backbone"] = backbone
        if hasattr(self.student["backbone"], "init_weights"):
            self.student["backbone"].init_weights()
        self.student_ema["backbone"] = MODELS_DICT[args.model](**args)
        self.student_ema["backbone"].load_state_dict(self.student["backbone"].state_dict(), strict=True)
        self.student["head"] = models_capi.L2NormLinear(backbone.pred_dim, args.capi.num_clusters)
        self.student_ema["head"] = models_capi.L2NormLinear(backbone.pred_dim, args.capi.num_clusters)
        self.student_ema["head"].load_state_dict(self.student["head"].state_dict(), strict=True)
        with torch.no_grad():
            for p_ema, p in zip(self.student_ema.parameters(), self.student.parameters(), strict=False):
                p_ema.copy_(p)
        self.teacher["backbone"] = self.student_ema["backbone"]
        self.teacher["head"] = models_capi.OnlineClustering(backbone.embed_dim, args.capi.num_clusters, **args.capi.clustering_kwargs)
        self.student_ema.requires_grad_(False)
        self.teacher["backbone"].requires_grad_(False)
        self.teacher["head"].requires_grad_(True)


def build_student_teacher(backbone: nn.Module, args: DictConfig) -> nn.Module:
    model = StudentTeacher(backbone, args)
    if hasattr(torch, "compile") and not getattr(args, "disable_compile", False):
        dynamic = bool(getattr(args, "compile_dynamic", False))
        compiled_do_teacher = torch.compile(do_teacher, dynamic=dynamic)  # type: ignore
        compiled_do_student = torch.compile(do_student, dynamic=dynamic)  # type: ignore
        model.do_teacher_compiled = compiled_do_teacher  # type: ignore[attr-defined]
        model.do_student_compiled = compiled_do_student  # type: ignore[attr-defined]
    return model


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


