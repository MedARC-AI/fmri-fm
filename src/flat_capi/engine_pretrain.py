from __future__ import annotations

from typing import Iterable, Callable

import torch
import wandb
from torch import Tensor, nn

from flat_capi.utils import misc


def _get_do_teacher(model) -> Callable:
    """Get the teacher forward function, preferring compiled version if present."""
    return getattr(model, "do_teacher_compiled", do_teacher)


def _get_do_student(model) -> Callable:
    """Get the student loss function, preferring compiled version if present."""
    return getattr(model, "do_student_compiled", do_student)


def do_student(
    model: nn.Module,
    images: Tensor,
    predict_indices: Tensor,
    visible_indices: Tensor,
    target: Tensor,
    temp: float,
) -> Tensor:
    """Compute the CAPI student loss against teacher targets.

    Args:
        model: Student-teacher wrapper.
        images: Input batch [B, C, H, W] (or channels-as-time).
        predict_indices: Absolute indices of tokens to predict.
        visible_indices: Absolute indices of visible tokens.
        target: Teacher assignments used as soft labels.
        temp: Student softmax temperature.

    Returns:
        Mean loss tensor (double).
    """
    core = model.module if hasattr(model, "module") else model
    amp_dtype = images.dtype if images.dtype in (torch.float16, torch.bfloat16) else torch.float32
    enabled = amp_dtype != torch.float32
    with torch.autocast("cuda", dtype=amp_dtype, enabled=enabled):
        _, backbone_predictions = core.student.backbone.forward_pretrain(
            images,
            visible_indices=visible_indices,
            predict_indices=predict_indices,
            do_prediction=True,
        )
        pred = core.student.head(backbone_predictions)
        loss = -torch.sum(target.float() * torch.log_softmax(pred.float() / temp, dim=-1), dim=-1)
    return loss.double().mean()


def do_teacher(
    model: nn.Module,
    images: Tensor,
    predict_indices: Tensor,
) -> tuple[Tensor, Tensor]:
    """Run teacher backbone and clustering head to produce targets and their loss."""
    core = model.module if hasattr(model, "module") else model
    amp_dtype = images.dtype if images.dtype in (torch.float16, torch.bfloat16) else torch.float32
    enabled = amp_dtype != torch.float32
    with torch.autocast("cuda", dtype=amp_dtype, enabled=enabled):
        with torch.set_grad_enabled(False):
            patch_before_head, _ = core.teacher.backbone.forward_pretrain(images)
    bs, n_visible, dim = patch_before_head.shape
    if amp_dtype == torch.float16:
        with torch.autocast("cuda", enabled=False):
            with torch.set_grad_enabled(True):
                patch_after_head, loss = core.teacher.head(patch_before_head.float().transpose(0, 1))
        with torch.set_grad_enabled(False):
            patch_after_head = patch_after_head.detach().transpose(0, 1)
            selected_patch_after_head = torch.index_select(
                patch_after_head.reshape(bs * n_visible, -1),
                dim=0,
                index=predict_indices,
            )
    else:
        with torch.autocast("cuda", dtype=amp_dtype, enabled=enabled):
            with torch.set_grad_enabled(True):
                patch_after_head, loss = core.teacher.head(patch_before_head.transpose(0, 1))
        with torch.set_grad_enabled(False):
            patch_after_head = patch_after_head.detach().transpose(0, 1)
            selected_patch_after_head = torch.index_select(
                patch_after_head.reshape(bs * n_visible, -1),
                dim=0,
                index=predict_indices,
            )
    return selected_patch_after_head, loss


def train_one_epoch(
    model: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    args=None,
    num_batches=None,
    log_wandb=False,
    clustering_optimizer: torch.optim.Optimizer | None = None,
    student_lr_sched=None,
    momentum_sched=None,
    cluster_lr_sched=None,
    it_start: int = 0,
    student_temp_sched=None,
    target_temp_sched=None,
    pred_temp_sched=None,
):
    """One training epoch of CAPI pretraining with EMA and clustering head.

    Supports gradient accumulation, temperature schedules, and optional separate
    optimizer for the clustering head.
    """
    model.train(True)
    core = model.module if hasattr(model, "module") else model
    student_param_list = list(core.student.parameters())
    ema_param_list = list(core.student_ema.parameters())
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Train: Epoch: [{}]".format(epoch)
    print_freq = 1 if args.debug else 100
    debug_steps = 10 * args.accum_iter
    log_wandb = misc.is_main_process() and log_wandb

    accum_iter = args.accum_iter
    if num_batches is None:
        num_batches = len(data_loader)

    optimizer.zero_grad(set_to_none=True)
    if clustering_optimizer is not None:
        clustering_optimizer.zero_grad(set_to_none=True)

    _accum_capi_loss = 0.0
    _accum_cluster_loss = 0.0
    _accum_count = 0

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, total_steps=num_batches)
    ):
        it = it_start + data_iter_step
        base_lr = student_lr_sched[it] if student_lr_sched is not None else optimizer.param_groups[0]["lr"]
        for pg in optimizer.param_groups:
            lr_mult = float(pg.get("lr_multiplier", 1.0))
            wd_mult = float(pg.get("wd_multiplier", 1.0))
            pg["lr"] = base_lr * lr_mult
            pg["weight_decay"] = getattr(args, "weight_decay", 0.0) * wd_mult
        if clustering_optimizer is not None:
            clr = cluster_lr_sched[it] if cluster_lr_sched is not None else optimizer.param_groups[0]["lr"]
            clustering_optimizer.param_groups[0]["lr"] = clr

        images = batch["image"].to(device, non_blocking=True)
        visible_indices = batch["visible_indices"].to(device, non_blocking=True)
        predict_indices = batch["predict_indices"].to(device, non_blocking=True)

        teacher_fn = _get_do_teacher(model)
        targets, clustering_loss = teacher_fn(model, images, predict_indices)
        (clustering_loss / accum_iter).backward()

        student_fn = _get_do_student(model)
        curr_it = it
        if student_temp_sched is not None:
            args.capi.student_temp = float(student_temp_sched[curr_it])
        if target_temp_sched is not None:
            args.capi.clustering_kwargs.target_temp = float(target_temp_sched[curr_it])
        if pred_temp_sched is not None:
            args.capi.clustering_kwargs.pred_temp = float(pred_temp_sched[curr_it])

        capi_loss = student_fn(model, images, predict_indices, visible_indices, targets, args.capi.student_temp)
        with torch.no_grad():
            target_entropy = -torch.xlogy(targets, targets).sum(dim=-1).mean()
        grad_norm = loss_scaler(
            capi_loss / accum_iter,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
            clip_grad=args.clip_grad,
        )

        if (data_iter_step + 1) % accum_iter == 0:
            with torch.no_grad():
                m = momentum_sched[it] if momentum_sched is not None else 0.999
                torch._foreach_mul_(ema_param_list, m)
                torch._foreach_add_(ema_param_list, student_param_list, alpha=1 - m)
            if clustering_optimizer is not None:
                clustering_optimizer.step()
                clustering_optimizer.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        _accum_capi_loss += float(capi_loss.item())
        _accum_cluster_loss += float(clustering_loss.item())
        _accum_count += 1
        metric_logger.update(capi_loss=capi_loss.item(), clustering_loss=clustering_loss.item())
        curr_lr = optimizer.param_groups[0]["lr"]
        curr_clr = clustering_optimizer.param_groups[0]["lr"] if clustering_optimizer is not None else curr_lr
        metric_logger.update(lr=curr_lr)
        metric_logger.update(clustering_lr=curr_clr)
        metric_logger.update(target_entropy=float(target_entropy))
        if (data_iter_step + 1) % accum_iter == 0:
            _avg_capi_local = _accum_capi_loss / max(_accum_count, 1)
            _avg_cluster_local = _accum_cluster_loss / max(_accum_count, 1)
            total_loss_value = float(_avg_capi_local + _avg_cluster_local)
            _capi_loss_local = float(_avg_capi_local)
            _clustering_loss_local = float(_avg_cluster_local)
            _target_entropy_local = float(target_entropy)
            loss_value_reduce = misc.all_reduce_mean(total_loss_value)
            capi_loss_reduce = misc.all_reduce_mean(_capi_loss_local)
            clustering_loss_reduce = misc.all_reduce_mean(_clustering_loss_local)
            target_entropy_reduce = misc.all_reduce_mean(_target_entropy_local)

            unit = str(getattr(args, "log_metrics_unit", "step")).lower()
            cadence = int(getattr(args, "log_metrics_cadence", 1))
            should_log = False
            if unit == "step":
                global_step = int(it + 1)
                should_log = (global_step % max(cadence, 1) == 0)
            else:
                global_step = int(it + 1)
                should_log = ((epoch + 1) % max(cadence, 1) == 0)

            if log_wandb and should_log:
                wandb.log(
                    {
                        "train/loss": loss_value_reduce,
                        "train/lr": float(curr_lr),
                        "train/capi_loss": capi_loss_reduce,
                        "train/clustering_loss": clustering_loss_reduce,
                        "train/clustering_lr": float(curr_clr),
                        "train/target_entropy": target_entropy_reduce,
                        "train/grad_norm": float(grad_norm.item()) if grad_norm is not None else None,
                    },
                    step=global_step,
                )
            _accum_capi_loss = 0.0
            _accum_cluster_loss = 0.0
            _accum_count = 0

        if args.debug and (data_iter_step + 1) >= debug_steps:
            break

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


