# This source code is licensed under the Apache License, Version 2.0
#
# References:
# deit: https://github.com/facebookresearch/deit/blob/main/utils.py
# beit3: https://github.com/microsoft/unilm/blob/master/beit3/utils.py
# capi: https://github.com/facebookresearch/capi/blob/main/utils.py
# dinov2: https://github.com/facebookresearch/dinov2/blob/main/dinov2/utils/param_groups.py
# timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/cuda.py
# dino: https://github.com/facebookresearch/dino/blob/main/utils.py

import datetime
import inspect
import os
import random
import subprocess
import time
from collections import defaultdict, deque
from omegaconf import OmegaConf
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.amp import GradScaler
from torch.optim import Optimizer


# these very useful utils copied from deit with only minor changes
# thanks to the original authors, wherever you are


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(float(value))
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        if not self.count:
            return float("nan")
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        if not self.count:
            return float("nan")
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        if not self.count:
            return float("nan")
        return self.total / self.count

    @property
    def max(self):
        if not self.count:
            return float("nan")
        return max(self.deque)

    @property
    def value(self):
        if not self.count:
            return float("nan")
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, (torch.Tensor, np.generic)):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, total_steps=None):
        i = 0
        total_steps = total_steps or len(iterable)
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(total_steps))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            if i >= total_steps:
                break
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == total_steps - 1:
                eta_seconds = iter_time.global_avg * (total_steps - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            total_steps,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )

                else:
                    print(
                        log_msg.format(
                            i,
                            total_steps,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / total_steps
            )
        )


def setup_for_distributed(log_path=None):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    is_master = is_main_process()

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)
            # tee to log file
            if log_path and "file" not in kwargs:
                with open(log_path, "a") as f:
                    builtin_print(*args, file=f, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    # removed slurm block, can add if we use slurm
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    else:
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank})")
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        world_size=args.world_size,
        rank=args.rank,
        device_id=args.gpu,
    )
    torch.distributed.barrier()


# checkpoint saving utils adapted from beit3


def save_model(args, epoch, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / f"checkpoint-{epoch:05d}.pth"
    last_checkpoint_path = output_dir / "checkpoint-last.pth"

    to_save = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "scaler": None if loss_scaler is None else loss_scaler.state_dict(),
        "args": OmegaConf.to_container(args),
    }

    print(f"saving checkpoint {last_checkpoint_path}")
    save_on_master(to_save, last_checkpoint_path)
    if args.checkpoint_period and epoch % args.checkpoint_period == 0:
        print(f"saving checkpoint {checkpoint_path}")
        save_on_master(to_save, checkpoint_path)

    if args.max_checkpoints and is_main_process():
        all_checkpoints = sorted(output_dir.glob("checkpoint-[0-9]*.pth"))
        del_count = max(0, len(all_checkpoints) - args.max_checkpoints)
        for checkpoint_path in all_checkpoints[:del_count]:
            print(f"removing checkpoint {checkpoint_path}")
            checkpoint_path.unlink()


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    auto_resume = getattr(args, "auto_resume", True)
    output_dir = Path(args.output_dir)

    last_checkpoint_path = output_dir / "checkpoint-last.pth"
    if auto_resume and last_checkpoint_path.exists():
        args.ckpt = str(last_checkpoint_path)
        args.resume = True

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        model_without_ddp.load_state_dict(ckpt["model"])
        print(f"loaded model state from checkpoint {args.ckpt}")

        if args.resume:
            optimizer.load_state_dict(ckpt["optimizer"])
            if loss_scaler is not None:
                loss_scaler.load_state_dict(ckpt["scaler"])
            args.start_epoch = ckpt["epoch"] + 1
            print(f"loaded optimizer state, resuming training from {args.start_epoch}")


# optimization utils


# from capi
class WarmupThenCosine:
    def __init__(
        self,
        base_value: float,
        final_value: float,
        total_iters: int,
        warmup_iters: int = 0,
        start_warmup_value: float = 0.0,
        freeze_iters: int = 0,
        truncate_cos: float = 1.0,
    ):
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros(freeze_iters)

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * truncate_cos * iters / len(iters))
        )
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))
        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it: int) -> float:
        if it >= self.total_iters:
            return self.final_value
        # cast to float or else it can corrupt the checkpoint
        return float(self.schedule[it])


# adapted from timm backward logic
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/cuda.py
def backward_step(
    loss: Tensor,
    optimizer: Optimizer,
    scaler: GradScaler = None,
    need_update: bool = True,
    max_norm: float | None = None,
) -> Tensor | None:
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    if need_update:
        if scaler is not None:
            scaler.unscale_(optimizer)

        total_norm = clip_grad(optimizer, max_norm)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    else:
        total_norm = None
    return total_norm


def clip_grad(optimizer: Optimizer, max_norm: float | None = None) -> Tensor:
    params = [p for group in optimizer.param_groups for p in group["params"]]
    if max_norm:
        total_norm = nn.utils.clip_grad_norm_(params, max_norm=max_norm)
    else:
        total_norm = nn.utils.get_total_norm(params)
    return total_norm


# from dinov2 with some minor changes
def get_param_groups(model, patch_embed_lr_mult=1.0):
    # no lr decay, we could add this later if needed
    all_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        d = {"param": param, "lr_multiplier": 1.0, "wd_multiplier": 1.0, "name": name}

        if name.endswith(".bias") or "norm" in name or "gamma" in name:
            d["wd_multiplier"] = 0.0

        if "patch_embed" in name:
            d["lr_multiplier"] = d["lr_multiplier"] * patch_embed_lr_mult

        all_params.append(d)

    param_groups = _fuse_param_groups(all_params)
    return param_groups


def _fuse_param_groups(all_param_groups):
    fused_param_groups = defaultdict(lambda: {"params": []})
    for d in all_param_groups:
        keys = sorted(set(d.keys()) - {"param", "name"})
        identifier = "_".join(f"{k}{d[k]}" for k in keys)
        for k in keys:
            fused_param_groups[identifier][k] = d[k]
        fused_param_groups[identifier]["params"].append(d["param"])

    param_groups = list(fused_param_groups.values())
    return param_groups


def update_lr(param_groups, lr: float):
    for group in param_groups:
        group["lr"] = lr * group["lr_multiplier"]


def update_wd(param_groups, weight_decay: float | None = None):
    for group in param_groups:
        group["weight_decay"] = weight_decay * group["wd_multiplier"]


# moving data to cuda utils copied from capi
# added device argument


def send_data(x, device=None):
    if device is None:
        device = torch.device("cuda")
    else:
        device = torch.device(device)

    if isinstance(x, torch.Tensor):
        x = x.to(device=device, non_blocking=True)
        if device.type == "cuda":
            x.record_stream(torch.cuda.current_stream(device))
        return x
    if isinstance(x, dict):
        return {k: send_data(v, device=device) for k, v in x.items()}
    if isinstance(x, list):
        return [send_data(v, device=device) for v in x]
    return x


def pre_send_to_cuda_wrapper(generator, device=None):
    """From apex"""
    data = None
    stream = torch.cuda.Stream(device)
    for next_data in generator:
        # Move to GPU
        with torch.cuda.stream(stream):
            next_data = send_data(next_data, device=device)
        if data is not None:
            yield data
        torch.cuda.current_stream(device).wait_stream(stream)
        data = next_data


# other misc utils


# from dino
def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommitted changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


# from timm
def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


# mine :)
def filter_kwargs(func, kwargs):
    sigature = inspect.signature(func)
    kwargs = {k: v for k, v in kwargs.items() if k in sigature.parameters}
    return kwargs
