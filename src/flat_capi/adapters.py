from __future__ import annotations

from typing import Optional, Tuple

import torch
from omegaconf import OmegaConf
from torch import Tensor, nn

import flat_capi.models_capi as models_capi


class CapiBackboneAdapter(nn.Module):
    """
    Adapter that exposes a MAE-compatible forward_embedding API for the CAPI EncoderDecoder.

    The MAE probe expects:
      - forward_embedding(images, mask=None) -> (cls_embeds [B,1,D] | None,
                                                reg_embeds [B,R,D] | None,
                                                patch_embeds [B,N,D])
    The CAPI EncoderDecoder forward() returns:
      - global_repr [B, D]
      - registers [B, R, D]
      - feature_map [B, H', W', D]
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        time_as_channels: bool = False,
        select_frame_index: int = 0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.time_as_channels = bool(time_as_channels)
        self.select_frame_index = int(select_frame_index)

    @torch.no_grad()
    def _prepare_inputs(self, images: Tensor) -> Tensor:
        """
        Convert [B, C, T, H, W] to the 2D CAPI input as needed.
        If time_as_channels, merge time into channels: [B, T, H, W].
        Otherwise, select a single temporal frame: [B, C, H, W].
        """
        if images.ndim == 5:  # [B, C, T, H, W]
            if self.time_as_channels:
                # Expect C=1 for fMRI clips; squeeze channel then use time as channels
                if images.shape[1] == 1:
                    images = images.squeeze(1)
                else:
                    # If channels > 1, assume they already encode time in channels
                    pass
                # Now [B, T, H, W]
            else:
                # Select one frame along time dimension (index clamped to valid range)
                _, _, T, _, _ = images.shape
                idx = min(max(self.select_frame_index, 0), T - 1)
                images = images[:, :, idx]  # [B, C, H, W]
        # else: already [B, C, H, W] or [B, T, H, W]
        return images

    def forward_embedding(
        self,
        images: Tensor,
        mask: Optional[Tensor] = None,
        mask_ratio: Optional[float] = None,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Tensor]:
        """
        Produce MAE-style token outputs from a CAPI backbone.

        Args:
            images: Input tensor of shape [B, C, T, H, W] or [B, C, H, W].
            mask: Optional spatial mask; ignored by this adapter (kept for API parity).
            mask_ratio: Optional mask ratio; ignored by this adapter (kept for API parity).

        Returns:
            Tuple of (cls_token, reg_tokens, patch_tokens) where:
              - cls_token: [B, 1, D] or None if not produced by the backbone
              - reg_tokens: [B, R, D] or None if not produced/empty
              - patch_tokens: [B, N, D] flattened from feature map tokens
        """
        x = self._prepare_inputs(images)
        global_repr, registers, feature_map = self.backbone.forward(x)
        # Shapes:
        # global_repr: [B, D], registers: [B, R, D], feature_map: [B, H', W', D]
        cls_token = global_repr.unsqueeze(1) if global_repr is not None else None
        reg_tokens = registers if registers is not None and registers.numel() > 0 else None
        B, Hp, Wp, D = feature_map.shape
        patch_tokens = feature_map.reshape(B, Hp * Wp, D)
        return cls_token, reg_tokens, patch_tokens


def _extract_backbone_state_dict(full_state: dict, prefer_ema: bool = True) -> dict:
    """
    Extract the sub-state-dict corresponding to the student_ema (or student) backbone
    from a CAPI checkpoint state dict.
    """
    prefixes = []
    if prefer_ema:
        prefixes.append("student_ema.backbone.")
    prefixes.append("student.backbone.")

    chosen_prefix = None
    for p in prefixes:
        if any(k.startswith(p) for k in full_state.keys()):
            chosen_prefix = p
            break
    if chosen_prefix is None:
        raise KeyError("Could not find backbone weights under student_ema.backbone or student.backbone")

    out = {}
    for k, v in full_state.items():
        if k.startswith(chosen_prefix):
            out[k[len(chosen_prefix) :]] = v
    return out


def create_capi_backbone_from_ckpt(
    ckpt_path: str,
    *,
    device: torch.device | str = "cpu",
    prefer_ema: bool = True,
) -> tuple[nn.Module, OmegaConf]:
    """
    Build a fresh CAPI EncoderDecoder and load the EMA (or student) backbone weights from a checkpoint.
    Returns the model (on CPU by default) and the saved config (OmegaConf).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    saved_args = OmegaConf.create(ckpt.get("args", {}))

    model_name = getattr(saved_args, "model", "vit_l14_capi")
    model_ctor = getattr(models_capi, model_name, models_capi.vit_l14_capi)
    # Pass-through saved_args as kwargs; ctor is permissive
    backbone: nn.Module = model_ctor(**saved_args)

    full_state = ckpt["model"]
    bb_state = _extract_backbone_state_dict(full_state, prefer_ema=prefer_ema)
    missing, unexpected = backbone.load_state_dict(bb_state, strict=False)
    if missing or unexpected:
        # Strict=True could be enforced, but allow minor non-critical deltas
        pass

    backbone.to(torch.device(device))
    return backbone, saved_args


def create_adapter_from_ckpt(
    ckpt_path: str,
    *,
    device: torch.device | str = "cpu",
    prefer_ema: bool = True,
    time_as_channels: Optional[bool] = None,
    select_frame_index: Optional[int] = None,
) -> tuple[CapiBackboneAdapter, OmegaConf]:
    backbone, saved_args = create_capi_backbone_from_ckpt(ckpt_path, device=device, prefer_ema=prefer_ema)
    if time_as_channels is None:
        time_as_channels = bool(getattr(saved_args, "time_as_channels", False))
    if select_frame_index is None:
        select_frame_index = int(getattr(saved_args, "select_frame_index", 0))
    adapter = CapiBackboneAdapter(
        backbone,
        time_as_channels=time_as_channels,
        select_frame_index=select_frame_index,
    )
    adapter.to(torch.device(device))
    return adapter, saved_args


