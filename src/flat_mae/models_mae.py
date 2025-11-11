# Copyright (c) Sophont, Inc
#
# This source code is licensed under the Apache License, Version 2.0
#
# References:
# capi: https://github.com/facebookresearch/capi/blob/main/model.py
# timm: https://github.com/huggingface/pytorch-image-models/blob/v1.0.20/timm/models/vision_transformer.py

"""
From-scratch re-implementation of the original MAE model.

MaskedEncoder: standard ViT with masking
MaskedDecoder: MAE decoder transformer supporting multiple decoding modes:
    - self-attention (classic MAE)
    - cross-attention (CrossMAE)
    - cross-register attention (MAETok)
MaskedAutoEncoderViT: full MAE model supporting image and video
"""

from typing import Literal, Type

import torch
import torch.nn as nn
from torch import Tensor
from huggingface_hub import PyTorchModelHubMixin
from jaxtyping import Float, Int
from timm.layers import to_2tuple, to_ntuple
from einops import rearrange

from .modules import (
    Block,
    LayerNorm,
    Patchify2D,
    Patchify3D,
    StridedPatchify3D,
    AbsolutePosEmbed,
    SeparablePosEmbed,
    SinCosPosEmbed2D,
    SinCosPosEmbed3D,
    Normalize,
)
from .masking import trim_patch_mask, pad_image_mask
from .utils import filter_kwargs


Layer = Type[nn.Module]


class MaskedEncoder(nn.Module):
    """
    Masked transformer encoder.

    Following timm ViT but with abstracted patch embed for more generality.
    """

    def __init__(
        self,
        patchify: nn.Module,
        patch_embed: nn.Module,
        pos_embed: nn.Module,
        depth: int = 12,
        embed_dim: int = 768,
        num_heads: int = 12,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        mlp_ratio: int | float = 4,
        class_token: bool = True,
        reg_tokens: int = 0,
        no_embed_class: bool = False,
        final_norm: bool = True,
        drop_path_rate: float = 0.0,
        mask_drop_scale: bool = False,
    ):
        super().__init__()
        self.num_prefix_tokens = int(class_token) + reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class

        # scale inputs by 1 / observed rate (like dropout)
        self.mask_drop_scale = mask_drop_scale

        # inject tokenization modules, so that the encoder doesn't specifically need to
        # know how the data are tokenized, while still implementing a complete
        # self-contained model.
        self.patchify = patchify
        self.patch_embed = patch_embed
        self.pos_embed = pos_embed

        R = reg_tokens
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.empty(1, R, embed_dim)) if reg_tokens else None

        if not no_embed_class:
            self.cls_token_pos = nn.Parameter(torch.empty(1, 1, embed_dim)) if class_token else None
            self.reg_token_pos = nn.Parameter(torch.empty(1, R, embed_dim)) if reg_tokens else None
        else:
            self.cls_token_pos = self.reg_token_pos = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[ii],
                )
                for ii in range(depth)
            ]
        )

        self.norm = LayerNorm(embed_dim) if final_norm else nn.Identity()

        self.reset_parameters()

    def extra_repr(self):
        return (
            f"class_token={self.has_class_token}, reg_tokens={self.num_reg_tokens}, "
            f"no_embed_class={self.no_embed_class}, mask_drop_scale={self.mask_drop_scale}"
        )

    def reset_parameters(self) -> None:
        for p in [self.cls_token, self.cls_token_pos, self.reg_token, self.reg_token_pos]:
            if p is not None:
                nn.init.trunc_normal_(p, std=0.02)

    def cat_tokens(self, x: Tensor) -> Tensor:
        # prepend cls and reg tokens with optional learned position embedding
        # the cls and reg pos embedding is ofc redundant, but included in many other
        # implementations.
        B, _, _ = x.shape

        to_cat = []
        if self.has_class_token:
            cls_token = self.cls_token
            if not self.no_embed_class:
                cls_token = cls_token + self.cls_token_pos
            to_cat.append(cls_token.expand(B, -1, -1))

        if self.num_reg_tokens:
            reg_token = self.reg_token
            if not self.no_embed_class:
                reg_token = reg_token + self.reg_token_pos
            to_cat.append(reg_token.expand(B, -1, -1))

        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)
        return x

    def chunk_tokens(self, x: Tensor) -> tuple[Tensor | None, Tensor | None, Tensor]:
        cls_offset = int(self.has_class_token)
        cls = x[:, :cls_offset] if self.has_class_token else None
        if self.num_reg_tokens:
            reg = x[:, cls_offset : self.num_prefix_tokens, :]
        else:
            reg = None
        patch = x[:, self.num_prefix_tokens :, :]
        return cls, reg, patch

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        mask_ratio: float | None = None,
    ) -> tuple[
        Float[Tensor, "B 1 D"] | None,
        Float[Tensor, "B R D"] | None,
        Float[Tensor, "B L D"],
        Tensor | None,
        Int[Tensor, "B L"] | None,
    ]:
        """
        x: input data, e.g. shape [B, C, H, W] for image, [B, C, T, H, W] for video
        mask: visible mask, 1 = visible, 0 = invisible. broadcastable shape
        mask_ratio: mask ratio for uniform random masking

        returns:
        - cls_embeds: [B, 1, D]
        - reg_embeds: [B, R, D]
        - patch_embeds: [B, L, D], where L is the number of visible patches
        - mask: observed mask, 1 = observed, 0 = unobserved. same shape as input
        - mask_ids: indices of visible patches [B L]
        """
        dtype = x.dtype
        device = x.device

        # apply mask to the input
        if mask is not None:
            mask = mask.to(dtype).expand_as(x)
            x = mask * x

        # patchify input
        x = self.patchify(x)
        B, N, P = x.shape

        # patchify mask and apply dropout style scaling
        if mask is not None:
            mask_patches = self.patchify(mask)
            patch_num_obs = mask_patches.sum(dim=-1)
            patch_mask = (patch_num_obs > 0).to(dtype)
            # rescale input to compensate for number of observed values
            # similar to dropout scaling
            if self.mask_drop_scale:
                x = x * (P / patch_num_obs.unsqueeze(-1).clamp(min=1.0))
        elif mask_ratio is not None:
            patch_mask = torch.ones((B, N), dtype=dtype, device=device)
            mask_patches = patch_mask.unsqueeze(-1).expand(-1, -1, P)
        else:
            patch_mask = mask_patches = None

        # patch and position embed
        x = self.patch_embed(x)
        x = self.pos_embed(x)

        if mask is not None or mask_ratio is not None:
            # trim mask to get equal number of visible patches per sample in batch.
            # patches sampled randomly if mask_ratio is not None, and otherwise in raster
            # grid order. (nb as a hack you can set mask_ratio=0.0 to get random instead of
            # trailing trim.)
            patch_mask, mask_ids = trim_patch_mask(
                patch_mask, mask_ratio=mask_ratio, shuffle=mask_ratio is not None
            )
            mask_patches = mask_patches * patch_mask.unsqueeze(-1)
            # nb, unnecessary computation for convenience
            mask = self.patchify.unpatchify(mask_patches)

            # keep only visible patches
            x = x.gather(1, mask_ids.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        else:
            mask_ids = None

        # transformer
        x = self.cat_tokens(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        cls_embeds, reg_embeds, patch_embeds = self.chunk_tokens(x)
        return cls_embeds, reg_embeds, patch_embeds, mask, mask_ids

    def forward_embedding(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        mask_ratio: float | None = None,
    ):
        cls_embeds, reg_embeds, patch_embeds, *_ = self.forward(x, mask=mask, mask_ratio=mask_ratio)
        return cls_embeds, reg_embeds, patch_embeds


class MaskedDecoder(nn.Module):
    """
    MAE decoder supporting:

    - Standard MAE decoding
    - VideoMAE v2 sparse subset decoding (via pred_ids)
    - CrossMAE cross-attention decoding (via cross_decode=True)
    - MAE-Tok cross-register decoding (pass reg_embeds with cross_decode=True)
    """

    def __init__(
        self,
        pos_embed: nn.Module,
        head: nn.Module | None = None,
        cross_decode: bool = False,
        context_dim: int | None = None,
        depth: int = 12,
        embed_dim: int = 768,
        num_heads: int = 12,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        mlp_ratio: int | float = 4,
        class_token: bool = True,
        no_embed_class: bool = False,
        final_norm: bool = True,
        no_context_proj: bool = False,
    ):
        super().__init__()
        assert not context_dim or not no_context_proj or context_dim == embed_dim or cross_decode, (
            "context projection required except for cross decoding"
        )

        self.cross_decode = cross_decode
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim)) if class_token else None
        if not no_embed_class:
            self.cls_token_pos = nn.Parameter(torch.empty(1, 1, embed_dim)) if class_token else None

        self.mask_token = nn.Parameter(torch.empty(1, 1, embed_dim))

        # decoder position embedding, encodes query position information into masks
        self.pos_embed = pos_embed

        if context_dim and not no_context_proj:
            self.proj = nn.Linear(context_dim, embed_dim)
        else:
            self.proj = nn.Identity()

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    mlp_ratio=mlp_ratio,
                    context_dim=context_dim if no_context_proj else None,
                )
                for ii in range(depth)
            ]
        )

        self.norm = LayerNorm(embed_dim) if final_norm else nn.Identity()

        # optional injected prediction head
        self.head = nn.Identity() if head is None else head

        self.reset_parameters()

    def extra_repr(self):
        return (
            f"cross_decode={self.cross_decode}, "
            f"class_token={self.has_class_token}, no_embed_class={self.no_embed_class}"
        )

    def reset_parameters(self) -> None:
        # official mae initializes decoder cls token to zeros
        # although perhaps this was an oversight
        if self.cls_token is not None:
            nn.init.zeros_(self.cls_token)
        if self.cls_token_pos is not None:
            nn.init.trunc_normal_(self.cls_token_pos, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def cat_tokens(self, x: Tensor) -> Tensor:
        B, _, _ = x.shape
        to_cat = []
        if self.has_class_token:
            cls_token = self.cls_token
            if not self.no_embed_class:
                cls_token = cls_token + self.cls_token_pos
            to_cat.append(cls_token.expand(B, -1, -1))
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)
        return x

    def chunk_tokens(self, x: Tensor) -> tuple[Tensor | None, Tensor]:
        cls_offset = int(self.has_class_token)
        cls = x[:, :cls_offset] if self.has_class_token else None
        patch = x[:, cls_offset:, :]
        return cls, patch

    def forward(
        self,
        embeds: Float[Tensor, "B L D"],
        embed_ids: Int[Tensor, "B L"] | None = None,
        pred_ids: Int[Tensor, "B Q"] | None = None,
    ) -> Float[Tensor, "B Q P"]:
        """
        embeds: input embeddings, can be patch or register embeddings, which will be fed
            into decoder transformer input or context stream depending on whether
            cross_decode is True.
        embed_ids: optional patch indices for input embeddings. If not provided, no
            position will be added to the embeddings. Not used for cross decoding.
        pred_ids: patch indices of query mask positions. If None, decode *all* patches.

        returns:
        - pred [B, Q, P] where Q is the number of prediction patches and P is the output
            dimension
        """
        B, L, D = embeds.shape

        Q = self.pos_embed.num_patches if pred_ids is None else pred_ids.shape[1]
        mask = self.mask_token.expand(B, Q, -1)
        mask = self.pos_embed(mask, pos_ids=pred_ids)

        embeds = self.proj(embeds)

        if self.cross_decode:
            # cross attention decoding (crossmae)
            x = mask
            context = embeds
            pred_offset = 0
        else:
            # standard self attention decoding (mae)
            # position is needed for mask, but maybe not needed for the visible patches,
            # since they already have position information.
            if embed_ids is not None:
                embeds = self.pos_embed(embeds, pos_ids=embed_ids)
            x = torch.cat([embeds, mask], dim=1)
            context = None
            pred_offset = L

        x = self.cat_tokens(x)
        for block in self.blocks:
            x = block(x, context=context)
        _, x = self.chunk_tokens(x)

        pred = x[:, pred_offset:]
        pred = self.norm(pred)
        pred = self.head(pred)
        return pred


class MaskedAutoencoderViT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size: int | tuple[int, ...] = 224,
        patch_size: int | tuple[int, ...] = 16,
        in_chans: int = 3,
        num_frames: int | None = None,
        t_patch_size: int | None = None,
        depth: int = 12,
        embed_dim: int = 768,
        num_heads: int = 12,
        decoder_depth: int = 4,
        decoder_embed_dim: int | None = 512,
        decoder_num_heads: int | None = 16,  # default from mae, head dim = 32
        qkv_bias: bool = True,
        proj_bias: bool = True,
        mlp_ratio: int | float = 4,
        class_token: bool = True,
        reg_tokens: int = 0,
        no_embed_class: bool = False,
        drop_path_rate: float = 0.0,
        mask_drop_scale: bool = False,
        t_pred_stride: int = 1,
        pred_edge_pad: int = 0,
        no_decode_pos: bool = False,
        pos_embed: Literal["abs", "sep", "sincos"] = "abs",
        decoding: Literal["attn", "cross", "crossreg"] = "attn",
        target_norm: Literal["none", "global", "frame", "patch"] | None = None,
    ):
        super().__init__()
        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        patch_size = to_ntuple(len(img_size))(patch_size)
        if num_frames:
            img_size = (num_frames, *img_size)
            patch_size = (t_patch_size, *patch_size)

        assert not decoding == "crossreg" or reg_tokens > 0, "crossreg decoding requires registers"
        assert pos_embed != "sep" or len(img_size) == 3, "separable pos embed requires 3D inputs"
        assert t_pred_stride == 1 or len(img_size) == 3, "t_pred_stride > 1 requires 3D inputs"

        self.decoding = decoding
        self.t_pred_stride = t_pred_stride  # predict subset of temporal frames
        self.pred_edge_pad = pred_edge_pad  # don't predict edges of visible patches
        self.no_decode_pos = no_decode_pos  # don't pos encode embeddings in decoder

        # patchify reshapes input into sequence of flattened patches, shape [B, N, P]
        ndim = len(img_size)
        patchify_layer = {2: Patchify2D, 3: Patchify3D}[ndim]
        patchify = patchify_layer(img_size, patch_size, in_chans=in_chans)

        # linear patch embedding P -> D
        patch_embed = nn.Linear(patchify.patch_dim, embed_dim)

        # position embedding
        # separable position embedding decouples the pos for the first dim (i.e. time)
        # from the others, following mae-st.
        if pos_embed == "sincos":
            pos_embed_layer = {2: SinCosPosEmbed2D, 3: SinCosPosEmbed3D}[ndim]
        else:
            pos_embed_layer = {"abs": AbsolutePosEmbed, "sep": SeparablePosEmbed}[pos_embed]
        pos_embed = pos_embed_layer(embed_dim, patchify.grid_size)

        # encoder. for inference, this model can be extracted and used like a regular vit
        self.encoder = MaskedEncoder(
            patchify=patchify,
            patch_embed=patch_embed,
            pos_embed=pos_embed,
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            mlp_ratio=mlp_ratio,
            class_token=class_token,
            reg_tokens=reg_tokens,
            no_embed_class=no_embed_class,
            drop_path_rate=drop_path_rate,
            mask_drop_scale=mask_drop_scale,
        )

        # strided prediction setup from mae-st
        if t_pred_stride > 1:
            self.pred_patchify = StridedPatchify3D(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                t_stride=t_pred_stride,
            )
        else:
            self.pred_patchify = patchify

        # fall back to encoder architecture width
        decoder_embed_dim = decoder_embed_dim or embed_dim
        decoder_num_heads = decoder_num_heads or num_heads

        decoder_pos_embed = pos_embed_layer(decoder_embed_dim, self.pred_patchify.grid_size)
        # we might want to try tying the weights of the prediction head to the patch
        # embedding at some point.
        decoder_head = nn.Linear(decoder_embed_dim, self.pred_patchify.patch_dim)

        cross_decode = decoding in {"cross", "crossreg"}
        self.decoder = MaskedDecoder(
            pos_embed=decoder_pos_embed,
            head=decoder_head,
            cross_decode=cross_decode,
            context_dim=embed_dim,
            depth=decoder_depth,
            embed_dim=decoder_embed_dim,
            num_heads=decoder_num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            mlp_ratio=mlp_ratio,
            class_token=class_token and not cross_decode,  # cls not active for cross-decode
            no_embed_class=no_embed_class,
            no_context_proj=cross_decode,  # don't project embeds for cross-decode
        )

        # mae style target normalization
        # dim is relative to an unflattened embedding tensor of shape [B, *grid_size, D]
        if target_norm not in {"none", None}:
            norm_dim = {
                "global": tuple(range(1, ndim + 2)),  # full sequence
                "frame": tuple(range(2, ndim + 2)),  # each "frame" (slice along first dim)
                "patch": -1,  # normalize each patch independently (mae pix norm loss)
            }[target_norm]
            self.target_norm = Normalize(self.pred_patchify.grid_size, dim=norm_dim)
        else:
            self.target_norm = None

        self.init_weights()

    def extra_repr(self):
        return (
            f"decoding={self.decoding}, t_pred_stride={self.t_pred_stride}, "
            f"pred_edge_pad={self.pred_edge_pad}, no_decode_pos={self.no_decode_pos}"
        )

    def init_weights(self):
        self.apply(_init_weights)

    def prepare_masks(
        self,
        img_mask: Tensor | None,
        visible_mask: Tensor | None,
        pred_mask: Tensor | None,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ):
        # expand all masks to the input shape and cast to the input dtype
        # intersect visible and pred mask with image data mask
        if img_mask is not None:
            img_mask = img_mask.to(dtype).expand(shape)

        if visible_mask is None:
            visible_mask = img_mask
        elif img_mask is not None:
            visible_mask = img_mask * visible_mask
        else:
            visible_mask = visible_mask.to(dtype).expand(shape)

        if pred_mask is None:
            pred_mask = img_mask
        elif img_mask is not None:
            pred_mask = img_mask * pred_mask
        else:
            pred_mask = pred_mask.to(dtype).expand(shape)

        return img_mask, visible_mask, pred_mask

    def prepare_targets(self, images: Tensor, img_mask: Tensor | None):
        """
        images: [B, C, H, W] or [B, C, T, H, W]
        img_mask: mask of valid data. only used for computing correct normalization
            stats. same shape and type as images.
        """
        targets_patches = self.pred_patchify(images)  # [B, N, P]

        # target normalization
        if self.target_norm is not None:
            # full image data mask used for normalization stats only
            if img_mask is not None:
                img_mask_patches = self.pred_patchify(img_mask)
            else:
                img_mask_patches = None
            targets_patches, *targets_stats = self.target_norm(targets_patches, img_mask_patches)
        else:
            targets_stats = None

        return targets_patches, targets_stats

    def prepare_pred_mask(
        self,
        visible_mask: Tensor,
        pred_mask: Tensor | None = None,
        pred_mask_ratio: float | None = None,
    ):
        """
        prepare prediction mask by removing visible content
        visible_mask: [B, C, H, W] or [B, C, T, H, W], 1 = visible, 0 = invisible
        pred_mask: same shape, 1 = predict, 0 = don't predict
        """
        if pred_mask is None:
            pred_mask = torch.ones_like(visible_mask)

        # pad edges of visible mask to avoid interpolating across patch edges
        if self.pred_edge_pad:
            visible_mask = pad_image_mask(visible_mask, pad=self.pred_edge_pad)

        # don't decode visible pixels (duh)
        pred_mask = pred_mask * (1 - visible_mask)

        # patchify
        pred_mask_patches = self.pred_patchify(pred_mask)  # [B, N, P]

        # trim prediction patches
        pred_patch_mask = pred_mask_patches.any(dim=-1).to(pred_mask.dtype)
        pred_patch_mask, pred_ids = trim_patch_mask(
            pred_patch_mask, mask_ratio=pred_mask_ratio, shuffle=pred_mask_ratio is not None
        )
        pred_mask_patches = pred_mask_patches * pred_patch_mask.unsqueeze(-1)
        B, Q = pred_ids.shape
        assert Q > 0, "empty pred_ids"

        return pred_mask_patches, pred_ids

    def forward_decoder(
        self,
        patch_embeds: Float[Tensor, "B L D"],
        reg_embeds: Float[Tensor, "B R D"] | None,
        visible_ids: Int[Tensor, "B L"],
        pred_ids: Int[Tensor, "B Q"] | None,
    ) -> Float[Tensor, "B Q P"]:
        if self.decoding == "crossreg":
            assert reg_embeds is not None, "reg_embeds required for crossreg decoding"
            embeds = reg_embeds
        else:
            embeds = patch_embeds
        if self.decoding == "attn" and not self.no_decode_pos:
            embed_ids = visible_ids
        else:
            embed_ids = None
        preds = self.decoder.forward(embeds, embed_ids=embed_ids, pred_ids=pred_ids)
        return preds

    def forward_loss(
        self,
        preds: Float[Tensor, "B Q P"],
        targets_patches: Float[Tensor, "B N P"],
        pred_mask_patches: Float[Tensor, "B N P"],
        pred_ids: Int[Tensor, "B Q"],
    ) -> Tensor:
        # select targets corresponding to predictions
        P = self.pred_patchify.patch_dim
        pred_ids = pred_ids.unsqueeze(-1).expand(-1, -1, P)
        targets_patches = targets_patches.gather(1, pred_ids)
        pred_mask_patches = pred_mask_patches.gather(1, pred_ids)

        # loss over predicted patches
        loss = (preds - targets_patches) ** 2
        loss = (pred_mask_patches * loss).sum() / pred_mask_patches.sum()
        return loss

    @torch.no_grad()
    def forward_pred_images(
        self,
        preds: Float[Tensor, "B Q P"],
        pred_ids: Float[Tensor, "B Q"],
        img_mask: Tensor | None = None,
        targets_stats: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        B, Q, P = preds.shape
        N = self.pred_patchify.num_patches

        preds = torch.zeros((B, N, P), dtype=preds.dtype, device=preds.device).scatter_(
            1, pred_ids.unsqueeze(-1).expand(-1, -1, P), preds
        )

        if targets_stats is not None:
            targets_mean, targets_std = targets_stats
            preds = preds * targets_std + targets_mean

        pred_images = self.pred_patchify.unpatchify(preds)
        if img_mask is not None:
            pred_images = img_mask * pred_images
        return pred_images

    def forward(
        self,
        images: Tensor,
        img_mask: Tensor | None = None,
        visible_mask: Tensor | None = None,
        pred_mask: Tensor | None = None,
        mask_ratio: float | None = 0.75,
        pred_mask_ratio: float | None = None,
        with_state: bool = True,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        img_mask, visible_mask, pred_mask = self.prepare_masks(
            img_mask, visible_mask, pred_mask, shape=images.shape, dtype=images.dtype
        )
        targets_patches, targets_stats = self.prepare_targets(images, img_mask)

        cls_embeds, reg_embeds, patch_embeds, visible_mask, visible_ids = self.encoder(
            images, mask=visible_mask, mask_ratio=mask_ratio
        )

        pred_mask_patches, pred_ids = self.prepare_pred_mask(
            visible_mask,
            pred_mask=pred_mask,
            pred_mask_ratio=pred_mask_ratio,
        )

        preds = self.forward_decoder(patch_embeds, reg_embeds, visible_ids, pred_ids)

        loss = self.forward_loss(preds, targets_patches, pred_mask_patches, pred_ids)

        if not with_state:
            return loss

        pred_mask = self.pred_patchify.unpatchify(pred_mask_patches)
        pred_images = self.forward_pred_images(
            preds, pred_ids, img_mask=img_mask, targets_stats=targets_stats
        )

        state = {
            "targets_patches": targets_patches,
            "targets_stats": targets_stats,
            "patch_embeds": patch_embeds,
            "cls_embeds": cls_embeds,
            "reg_embeds": reg_embeds,
            "visible_mask": visible_mask,
            "visible_ids": visible_ids,
            "pred_mask": pred_mask,
            "pred_ids": pred_ids,
            "preds": preds,
            "pred_images": pred_images,
        }
        return loss, state

    def forward_embedding(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        mask_ratio: float | None = None,
    ):
        return self.encoder.forward_embedding(x, mask, mask_ratio)

    def forward_denoise(
        self,
        imgs,
        mask_ratio=0.75,
        n_samples=4,
        img_mask=None,
        generator=None,
    ):
        """
        Explicit denoising via multiple masked reconstructions.

        runs mae n_samples + 1 times with partitioned patch masks, then aggregates
        reconstructions. each patch is reconstructed either n_samples or n_samples + 1 times due to remainder handling.
        structured signal will be consistent across runs, while unstructured noise will be inconsistent.

        args:
            imgs: input fmri volume [N, C, T, H, W]
            mask_ratio: fraction of patches to mask per run (1-mask_ratio are visible)
            n_samples: number of masked reconstructions to run
            img_mask: optional binary mask of valid voxels
            generator: torch.generator for reproducibility

        returns:
            denoised_imgs: [N, C, T, H, W] denoised fmri volume
        """
        # convert input to model's dtype to avoid dtype mismatch
        model_dtype = next(self.parameters()).dtype
        imgs = imgs.to(dtype=model_dtype)
        if img_mask is not None:
            img_mask = img_mask.to(dtype=model_dtype)
        
        N, C, T, H, W = imgs.shape
        # if samples is too high, there won't be enough visible patches.
        assert 1 / (n_samples + 1) >= (1 - mask_ratio), (
            "n_samples too large for mask_ratio"
        )

        # get patch dimensions (extract 2D spatial parts, handling both 2D and 3D cases)
        grid_size = self.encoder.patchify.grid_size
        patch_size = self.encoder.patchify.patch_size
        # for 3D: grid_size is (t, h, w), patch_size is (p_t, p_h, p_w)
        # for 2D: grid_size is (h, w), patch_size is (p_h, p_w)
        # we only need spatial dimensions (last 2)
        h, w = grid_size[-2:]
        ph, pw = patch_size[-2:]

        # we assume a shared img_mask for all samples for simplicity.
        if img_mask is not None:
            # img_mask should be [H, W] or [1, 1, H, W] or similar
            img_mask_squeezed = img_mask.squeeze()
            assert img_mask_squeezed.shape == (H, W), (
                "denoising requires a fixed image mask"
            )
            # use rearrange to patchify the spatial mask directly (like old version)
            # this works with 2D spatial patches regardless of temporal dimension
            patch_mask = rearrange(
                img_mask_squeezed, "(h p) (w q) -> (h w) (p q)", h=h, p=ph, w=w, q=pw
            )
            patch_mask = patch_mask.sum(dim=-1).clip(max=1)
            img_mask = img_mask.expand_as(imgs)
        else:
            patch_mask = torch.ones(h * w, device=imgs.device)

        # generate random patch permutation (using generator for reproducibility)
        valid_ids = patch_mask.flatten().nonzero().flatten()
        patch_permutation = valid_ids[
            torch.randperm(len(valid_ids), generator=generator, device=valid_ids.device)
        ]

        # run multiple masked reconstructions
        reconstructions = []
        decoder_masks = []

        # run n_samples + 1 times
        for run_idx in range(n_samples + 1):
            # create visible mask for this run using sliding window approach
            run_visible_mask = self._generate_visible_mask(
                imgs,
                run_idx,
                n_samples + 1,
                patch_permutation,
            )

            # prepare targets with normalization
            targets_patches, targets_stats = self.prepare_targets(imgs, img_mask)

            # encode with visible mask
            cls_embeds, reg_embeds, patch_embeds, visible_mask_out, visible_ids = self.encoder(
                imgs, mask=run_visible_mask, mask_ratio=None  # use provided mask, no random masking
            )

            # prepare prediction mask (predict non-visible patches)
            pred_mask_patches, pred_ids = self.prepare_pred_mask(
                visible_mask_out,
                pred_mask=None,
                pred_mask_ratio=None,  # predict all non-visible patches
            )

            # decode
            preds = self.forward_decoder(patch_embeds, reg_embeds, visible_ids, pred_ids)

            # unnormalize predictions
            if targets_stats is not None:
                targets_mean, targets_std = targets_stats
                # gather stats for predicted patches
                pred_mean = targets_mean.gather(1, pred_ids.unsqueeze(-1).expand(-1, -1, preds.shape[-1]))
                pred_std = targets_std.gather(1, pred_ids.unsqueeze(-1).expand(-1, -1, preds.shape[-1]))
                pred_unnorm = preds * pred_std + pred_mean
            else:
                pred_unnorm = preds

            # expand predictions to full patch space for aggregation
            # pred_unnorm is [N, Q, P], need to expand to [N, N_patches, P]
            N_patches = self.pred_patchify.num_patches
            P = pred_unnorm.shape[-1]
            pred_expanded = torch.zeros((N, N_patches, P), dtype=pred_unnorm.dtype, device=pred_unnorm.device)
            pred_expanded.scatter_(1, pred_ids.unsqueeze(-1).expand(-1, -1, P), pred_unnorm)
            
            # create decoder_mask to track which patches were predicted
            # use model dtype to ensure compatibility
            model_dtype = next(self.parameters()).dtype
            decoder_mask = torch.zeros((N, N_patches), device=pred_ids.device, dtype=model_dtype)
            decoder_mask.scatter_(1, pred_ids, torch.ones_like(pred_ids, dtype=model_dtype))

            reconstructions.append(pred_expanded)
            decoder_masks.append(decoder_mask)

        # aggregate reconstructions using proper weighted mean
        # each patch may be reconstructed a different number of times
        stacked_reconstructions = torch.stack(reconstructions)  # [n_runs, N, L, D]
        stacked_decoder_masks = torch.stack(decoder_masks)  # [n_runs, N, L]

        # compute weighted mean accounting for how many times each patch was reconstructed
        # decoder_mask: 1 = reconstructed, 0 = not reconstructed
        weights = stacked_decoder_masks.unsqueeze(-1)  # [n_runs, N, L, 1]

        # sum of reconstructions and sum of weights for each patch
        weighted_sum = (stacked_reconstructions * weights).sum(dim=0)  # [N, L, D]
        weight_sum = weights.sum(dim=0)  # [N, L, 1]

        final_patches = weighted_sum / weight_sum.clamp(min=1.0)

        # convert back to image space using unpatchify
        # note: t_pred_stride > 1 is automatically handled by StridedPatchify3D.unpatchify()
        # which repeats temporal frames to fill all time steps
        denoised_imgs = self.pred_patchify.unpatchify(final_patches)

        return denoised_imgs

    def _generate_visible_mask(
        self,
        imgs,
        run_idx,
        total_runs,
        patch_permutation,
    ):
        """
        Partition-based visible mask with correct temporal handling.
        - Randomly partition patch_permutation into total_runs.
        - Select run_idx group as visible.
        - Expand 2D patch-level mask to voxel space using unpatchify without manual repeats.
        """
        N, C, T, H, W = imgs.shape

        # Split shuffled permutation into groups
        groups = torch.tensor_split(patch_permutation, total_runs)
        visible_patch_indices = groups[run_idx]

        # Use patch dimensions from the model configuration
        # extract 2D spatial parts (handling both 2D and 3D cases)
        patch_size = self.encoder.patchify.patch_size
        grid_size = self.encoder.patchify.grid_size
        # for 3D: patch_size is (p_t, p_h, p_w), grid_size is (t, h, w)
        # for 2D: patch_size is (p_h, p_w), grid_size is (h, w)
        # we only need spatial dimensions (last 2 elements work for both cases)
        ph, pw = patch_size[-2:]  # spatial patch size
        h, w = grid_size[-2:]  # spatial grid size
        # for temporal: if 3D, use first dim; if 2D, assume single frame
        u = patch_size[0] if len(patch_size) == 3 else 1
        t = grid_size[0] if len(grid_size) == 3 else 1

        # Create patch_mask in patch space, matching model's parameter dtype
        # use the dtype of the first model parameter to ensure compatibility
        model_dtype = next(self.parameters()).dtype
        patch_mask = torch.zeros((N, h * w), device=imgs.device, dtype=model_dtype)
        patch_mask[:, visible_patch_indices] = 1.0
        patch_mask = patch_mask.unsqueeze(1).expand(-1, t, -1).flatten(1)

        # Expand to include voxel content: (N, t*h*w, u*ph*pw*C)
        patch_mask_expanded = patch_mask.unsqueeze(-1).expand(-1, -1, u * ph * pw * C)

        # use encoder's unpatchify to convert patch mask back to image space
        # patch ordering matches patchify3d: (t h w) flattened to [t*h*w]
        visible_mask = self.encoder.patchify.unpatchify(patch_mask_expanded)
        
        # mask is already in model_dtype from patch_mask creation, just return it
        return visible_mask


class MaskedViT(MaskedEncoder, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        in_chans: int = 3,
        patch_size: int | tuple[int, int] = 16,
        num_frames: int | None = None,
        t_patch_size: int | None = None,
        depth: int = 12,
        embed_dim: int = 768,
        num_heads: int = 12,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        mlp_ratio: int | float = 4,
        class_token: bool = True,
        reg_tokens: int = 0,
        no_embed_class: bool = False,
        final_norm: bool = True,
        drop_path_rate: float = 0.0,
        mask_drop_scale: bool = False,
        pos_embed: Literal["abs", "sep", "sincos"] = "abs",
    ):
        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        patch_size = to_ntuple(len(img_size))(patch_size)
        if num_frames:
            img_size = (num_frames, *img_size)
            patch_size = (t_patch_size, *patch_size)

        ndim = len(img_size)
        patchify_layer = {2: Patchify2D, 3: Patchify3D}[ndim]
        patchify = patchify_layer(img_size, patch_size, in_chans=in_chans)
        patch_embed = nn.Linear(patchify.patch_dim, embed_dim)
        if pos_embed == "sincos":
            pos_embed_layer = {2: SinCosPosEmbed2D, 3: SinCosPosEmbed3D}[ndim]
        else:
            pos_embed_layer = {"abs": AbsolutePosEmbed, "sep": SeparablePosEmbed}[pos_embed]
        pos_embed = pos_embed_layer(embed_dim, patchify.grid_size)

        super().__init__(
            patchify=patchify,
            patch_embed=patch_embed,
            pos_embed=pos_embed,
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            mlp_ratio=mlp_ratio,
            class_token=class_token,
            reg_tokens=reg_tokens,
            no_embed_class=no_embed_class,
            final_norm=final_norm,
            drop_path_rate=drop_path_rate,
            mask_drop_scale=mask_drop_scale,
        )

        self.init_weights()

    def init_weights(self):
        self.apply(_init_weights)


# JAX ViT xavier uniform init
# https://github.com/facebookresearch/capi/blob/main/model.py
def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
        nn.init.constant_(m.weight, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def _create_vit(**kwargs):
    kwargs = filter_kwargs(MaskedViT, kwargs)
    model = MaskedViT(**kwargs)
    return model


def _create_mae_vit(**kwargs):
    kwargs = filter_kwargs(MaskedAutoencoderViT, kwargs)
    model = MaskedAutoencoderViT(**kwargs)
    return model


def _convert_from_timm(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    out_dict = {}
    swaps = [
        ("patch_embed.proj", "patch_embed"),
    ]

    class_token = "cls_token" in state_dict
    reg_tokens = 0 if "reg_token" not in state_dict else state_dict["reg_token"].shape[1]
    num_prefix_tokens = int(class_token) + reg_tokens

    for name, p in state_dict.items():
        for old, new in swaps:
            if name.startswith(old):
                name = name.replace(old, new)

        if name == "patch_embed.weight":
            out_dict[name] = p.flatten(1)
        elif name == "pos_embed":
            out_dict["pos_embed.weight"] = p[0, num_prefix_tokens:, :]
            if class_token:
                out_dict["cls_token_pos"] = p[:, :1, :]
            if reg_tokens:
                out_dict["reg_token_pos"] = p[:, int(class_token) : num_prefix_tokens, :]
        elif "qkv" in name:
            q, k, v = p.chunk(3, dim=0)
            out_dict[name.replace("qkv", "q")] = q
            out_dict[name.replace("qkv", "k")] = k
            out_dict[name.replace("qkv", "v")] = v
        else:
            out_dict[name] = p
    return out_dict


def vit_small(**kwargs):
    model_args = dict(embed_dim=384, depth=12, num_heads=6)
    return _create_vit(**model_args, **kwargs)


def mae_vit_small(**kwargs):
    model_args = dict(embed_dim=384, depth=12, num_heads=6)
    return _create_mae_vit(**model_args, **kwargs)


def mae_vit_base(**kwargs):
    model_args = dict(embed_dim=768, depth=12, num_heads=12)
    return _create_mae_vit(**model_args, **kwargs)


# "patch embed" baseline model, depth 0 ViT (hah)
def patch_embed_small(**kwargs):
    model_args = dict(embed_dim=384, depth=0)
    return _create_vit(**model_args, **kwargs)


def patch_embed_base(**kwargs):
    model_args = dict(embed_dim=768, depth=0)
    return _create_vit(**model_args, **kwargs)
