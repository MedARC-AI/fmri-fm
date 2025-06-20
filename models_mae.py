# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
#
# Forked from MAE-st:
# https://github.com/facebookresearch/mae_st
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from util import video_vit
from util.logging import master_print as print


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        num_frames=16,
        t_patch_size=4,
        t_pred_patch_size=1,
        patch_embed=video_vit.PatchEmbed,
        no_qkv_bias=False,
        sep_pos_embed=False,
        trunc_init=False,
        cls_embed=False,
        **kwargs,
    ):
        super().__init__()
        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.cls_embed = cls_embed
        self.t_pred_patch_size = t_pred_patch_size
        assert t_patch_size % t_pred_patch_size == 0

        self.patch_embed = patch_embed(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            num_frames,
            t_patch_size,
        )
        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], embed_dim)
            )
            if self.cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, embed_dim),
            )

        self.blocks = nn.ModuleList(
            [
                video_vit.Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if sep_pos_embed:
            self.decoder_pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], decoder_embed_dim)
            )
            self.decoder_pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], decoder_embed_dim)
            )
            if self.cls_embed:
                self.decoder_pos_embed_class = nn.Parameter(
                    torch.zeros(1, 1, decoder_embed_dim)
                )
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, decoder_embed_dim),
            )

        self.decoder_blocks = nn.ModuleList(
            [
                video_vit.Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            self.t_pred_patch_size * patch_size**2 * in_chans,
            bias=True,
        )

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        print("model initialized")

    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.sep_pos_embed:
            torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

            torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)

            if self.cls_embed:
                torch.nn.init.trunc_normal_(self.pos_embed_class, std=0.02)
                torch.nn.init.trunc_normal_(self.decoder_pos_embed_class, std=0.02)
        else:
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        w = self.patch_embed.proj.weight.data
        if self.trunc_init:
            torch.nn.init.trunc_normal_(w)
            torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs, predict=True):
        """
        imgs: (N, C, T, H, W)
        x: (N, L, patch_size**2 *C)

        use the predictor t patch size when predict=True, and the patch embedding t
        patch size otherwise.
        """
        N, C, T, H, W = imgs.shape
        ph, pw = self.patch_embed.patch_size
        u = self.t_pred_patch_size if predict else self.patch_embed.t_patch_size
        assert H % ph == 0 and W % pw == 0 and T % u == 0
        h = H // ph
        w = W // pw
        t = T // u

        x = imgs.reshape(shape=(N, C, t, u, h, ph, w, pw))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * ph * pw * C))
        self.patch_info = (N, C, T, H, W, ph, pw, u, t, h, w)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *C)
        imgs: (N, C, T, H, W)
        """
        N, C, T, H, W, ph, pw, u, t, h, w = self.patch_info

        x = x.reshape(shape=(N, t, h, w, u, ph, pw, C))

        x = torch.einsum("nthwupqc->nctuhpwq", x)
        imgs = x.reshape(shape=(N, C, T, H, W))
        return imgs

    def random_masking(self, x, mask_ratio, visible_patch_mask):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        visible_patch_mask: [N, L] mask of visible patches, 1=visible, 0=not visible
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # shift missing patches to not be selected
        # if there are not enough visible patches, invisible patches will still
        # get selected. we have filled them with zeros. but should still be sure to
        # generate visible masks with enough patches to avoid this.
        if visible_patch_mask is not None:
            noise = noise + (1.0 - visible_patch_mask)

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x, mask_ratio, visible_mask):
        """
        x: [N, T, C, H, W]
        visible_mask: [*, H, W] mask of visible pixels, 1=visible, 0=not visible
        """
        if visible_mask is not None:
            # mask invisible part of x with zeros.
            x = visible_mask * x
            # [N, L] mask of patches containing some visible pixels
            visible_mask = visible_mask.expand_as(x)
            visible_patch_mask = self.patchify(visible_mask, predict=False)
            visible_patch_mask = visible_patch_mask.sum(dim=-1).clip(max=1)
        else:
            visible_patch_mask = None

        # embed patches
        x = self.patch_embed(x)
        N, T, L, C = x.shape

        x = x.reshape(N, T * L, C)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_keep = self.random_masking(
            x, mask_ratio, visible_patch_mask
        )
        x = x.view(N, -1, C)
        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # add pos embed w/o cls token
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            pos_embed = pos_embed.expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        else:
            if self.cls_embed:
                cls_ind = 1
            else:
                cls_ind = 0
            pos_embed = self.pos_embed[:, cls_ind:, :].expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed[:, :1, :].expand(x.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        x = x.view([N, -1, C]) + pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # note, in the case that the img_mask is small, we are doing a forward pass on
        # many extra patches where we won't evaluate the loss. this is a waste of
        # compute. a better strategy would be to evaluate the decoder on only a fixed
        # size subset of the valid image mask, as in cross MAE.
        N = x.shape[0]
        T = self.patch_embed.t_grid_size
        H, W = self.patch_embed.grid_size

        # embed tokens
        x = self.decoder_embed(x)
        C = x.shape[-1]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(N, T * H * W + 0 - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x_ = x_.view([N, T * H * W, C])
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        )  # unshuffle
        x = x_.view([N, T * H * W, C])
        # append cls token
        if self.cls_embed:
            decoder_cls_token = self.decoder_cls_token
            decoder_cls_tokens = decoder_cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((decoder_cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            decoder_pos_embed = self.decoder_pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.decoder_pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            if self.cls_embed:
                decoder_pos_embed = torch.cat(
                    [
                        self.decoder_pos_embed_class.expand(
                            decoder_pos_embed.shape[0], -1, -1
                        ),
                        decoder_pos_embed,
                    ],
                    1,
                )
        else:
            decoder_pos_embed = self.decoder_pos_embed[:, :, :]

        # add pos embed
        x = x + decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask, img_mask):
        """
        imgs: [N, C, T, H, W]
        pred: [N, t*h*w, u*p*p*C]
        mask: [N, t*h*w], 0 is keep, 1 is remove,
        img_mask: [N, C, T, H, W], 0 is invalid, 1 is valid
        """
        # nb, change here vs mae_st reference.
        # fixed the index selection to take the first frame from each temporal patch as
        # the target in the default case t_pred_patch_size = 1. also should now work for
        # any t_pred_patch_size that divides t_patch_size.
        N, C, T, H, W = imgs.shape
        t_step = self.patch_embed.t_patch_size // self.t_pred_patch_size
        t_indices = torch.arange(0, T, t_step, device=imgs.device)
        _imgs = torch.index_select(imgs, 2, t_indices)

        target = self.patchify(_imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2

        # exclude invalid pixels from loss
        if img_mask is not None:
            img_mask = img_mask.expand_as(imgs)
            img_mask = torch.index_select(img_mask, 2, t_indices)
            img_mask_patches = self.patchify(img_mask)
            # [N, L, D] mask of valid pixels to compute loss over
            mask = mask.unsqueeze(-1) * img_mask_patches
        else:
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75, img_mask=None, visible_mask=None):
        if visible_mask is None:
            visible_mask = img_mask
        elif img_mask is not None:
            visible_mask = img_mask * visible_mask
        latent, mask, ids_restore = self.forward_encoder(
            imgs, mask_ratio, visible_mask
        )
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*C]
        loss = self.forward_loss(imgs, pred, mask, img_mask)
        return loss, pred, mask

    @torch.no_grad()
    def forward_masked_recon(self, imgs, pred, mask, img_mask=None):
        # imgs: [N, C, T, H, W]
        # pred: [N, t*h*w, u*p*p*C]
        # mask: [N, t*h*w], 0 is keep, 1 is remove,
        N, C, T, H, W = imgs.shape
        t_step = self.patch_embed.t_patch_size // self.t_pred_patch_size
        t_indices = torch.arange(0, T, t_step, device=imgs.device)
        target = torch.index_select(imgs, 2, t_indices)

        # this caches patch info for unpatchify
        # necessary if batch size is different from training
        self.patchify(target)

        target = torch.einsum("ncthw->nthwc", target)

        pred = self.unpatchify(pred)
        pred = torch.einsum("ncthw->nthwc", pred)

        ph, pw = self.patch_embed.patch_size
        mask = mask.unsqueeze(-1).repeat(1, 1, ph * pw * C)  # (N, T*H*W, p*p*c)
        mask = self.unpatchify(mask)  # 1 is removing, 0 is keeping

        mask = torch.einsum("ncthw->nthwc", mask)

        # masked image
        im_masked = target * (1 - mask)

        # MAE reconstruction pasted with visible patches
        im_paste = target * (1 - mask) + pred * mask

        # process the img_mask to match the target.
        if img_mask is not None:
            img_mask = img_mask.expand_as(imgs)
            img_mask = torch.index_select(img_mask, 2, t_indices)
            img_mask = img_mask[:, 0]  # (N, T, H, W)

        return target, pred, mask, im_masked, im_paste, img_mask


def mae_vit_small_patch16(**kwargs):
    # nb, decoder is fixed across encoder sizes to embed dim 512.
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_base_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_large_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_huge_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
