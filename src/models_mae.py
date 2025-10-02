# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# MAE-st: https://github.com/facebookresearch/mae_st
# --------------------------------------------------------

import ast
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
        target_norm="none",
        num_frames=16,
        t_patch_size=4,
        t_pred_patch_size=None,
        patch_embed=video_vit.PatchEmbed,
        no_qkv_bias=False,
        sep_pos_embed=False,
        trunc_init=False,
        cls_embed=False,
        reg_tokens=0,
        no_cls_pos=False,
        init_decoder_scale=None,
        mask_patch_embed=False,
        t_embed_patch_indices=None,
        t_pred_patch_indices="0:16",
        **kwargs,
    ):
        super().__init__()
        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.cls_embed = cls_embed
        self.reg_tokens = reg_tokens
        self.no_cls_pos = no_cls_pos
        self.init_decoder_scale = init_decoder_scale
        self.mask_patch_embed = mask_patch_embed

        # we have the option to encode/decode arbitrary subset of frames in each patch
        # first get the within-patch indices for the encoder (embed) and decoder (pred)
        # the indices can be a list or slice expression like '0:8' or '0:None:2'
        t_embed_patch_indices = _parse_indices(t_embed_patch_indices, t_patch_size)
        # fall back to t_pred_patch_size for backwards compatibility.
        if t_pred_patch_indices is None and t_pred_patch_size:
            t_step = t_patch_size // t_pred_patch_size
            t_pred_patch_indices = list(range(0, t_patch_size, t_step))
        t_pred_patch_indices = _parse_indices(t_pred_patch_indices, t_patch_size)
        # encoder and decoder patch size
        self.t_embed_patch_size = len(t_embed_patch_indices)
        self.t_pred_patch_size = len(t_pred_patch_indices)
        # repeat with offsets to get selection indices for the full sequence
        t_num_patches = num_frames // t_patch_size
        t_patch_offsets = t_patch_size * torch.arange(t_num_patches)
        t_embed_indices = (t_patch_offsets[:, None] + t_embed_patch_indices).flatten()
        t_pred_indices = (t_patch_offsets[:, None] + t_pred_patch_indices).flatten()
        self.register_buffer("t_embed_indices", t_embed_indices)
        self.register_buffer("t_pred_indices", t_pred_indices)

        self.patch_embed = patch_embed(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            len(t_embed_indices),
            self.t_embed_patch_size,
        )
        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if self.reg_tokens:
            self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim))

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

        self.target_norm = target_norm

        self.initialize_weights()

        print("model initialized")

    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.reg_tokens:
            torch.nn.init.trunc_normal_(self.reg_token, std=0.02)
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

        # scale decoder head init to prevent hockey stick loss.
        # init head with zero works, cf timm init_weights_vit_jax.
        if self.init_decoder_scale is not None:
            self.decoder_pred.weight.data.mul_(self.init_decoder_scale)

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
        u = self.t_pred_patch_size if predict else self.t_embed_patch_size
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

    def random_masking(self, x, mask_ratio, visible_patch_mask, generator):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        visible_patch_mask: [N, L] mask of visible patches, 1=visible, 0=not visible
        generator: torch.Generator or None, for reproducibility
        """
        N, L, D = x.shape  # batch, length, dim

        # mask ratio is relative to the total number of patches, regardless of the
        # visible mask. this way, we get a consistent number of observed patches.
        # nb, old configs use a mask ratio that is relative to the number of visible
        # patches, so this would need to be adjusted.
        len_keep = int(L * (1 - mask_ratio))

        # adjust for number of visible patches
        if visible_patch_mask is not None:
            total_patches = int(visible_patch_mask.sum(dim=1).min().item())
            len_keep = min(len_keep, total_patches)

        noise = torch.rand(
            N, L, device=x.device, generator=generator
        )  # noise in [0, 1]

        # shift invisible patches to not be selected
        if visible_patch_mask is not None:
            noise = noise + (1.0 - visible_patch_mask)

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], dtype=x.dtype, device=x.device)
        mask = mask.scatter(
            dim=1,
            index=ids_keep,
            src=torch.zeros([N, len_keep], dtype=x.dtype, device=x.device),
        )
        return x_masked, mask, ids_keep

    def apply_pos_embed(self, x, ids_keep=None, decoder=False):
        """
        Apply position embedding and prepend extra tokens.

        x: [N, L, C]
        ids_keep: [N, L] position indices of x

        Reference: timm vit
        """
        N, L, C = x.shape

        if self.sep_pos_embed:
            if decoder:
                pos_embed_temporal = self.decoder_pos_embed_temporal
                pos_embed_spatial = self.decoder_pos_embed_spatial
            else:
                pos_embed_temporal = self.pos_embed_temporal
                pos_embed_spatial = self.pos_embed_spatial

            pos_embed = pos_embed_temporal[:, :, None] + pos_embed_spatial[:, None, :]
            pos_embed = pos_embed.flatten(1, 2)

            if decoder:
                cls_pos_embed = self.decoder_pos_embed_class
            else:
                cls_pos_embed = self.pos_embed_class
        else:
            pos_embed = self.decoder_pos_embed if decoder else self.pos_embed
            cls_pos_embed = pos_embed[:, :1, :]
            pos_embed = pos_embed[:, 1:, :]

        pos_embed = pos_embed.expand(N, -1, -1)

        if ids_keep is not None:
            pos_embed = torch.gather(
                pos_embed, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, C)
            )

        x = x + pos_embed

        to_cat = []
        if self.cls_embed:
            cls_token = self.decoder_cls_token if decoder else self.cls_token
            if not self.no_cls_pos:
                cls_token = cls_token + cls_pos_embed
            to_cat.append(cls_token.expand(N, -1, -1))

        if self.reg_tokens and not decoder:
            to_cat.append(self.reg_token.expand(N, -1, -1))

        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)

        return x

    def forward_encoder(self, x, mask_ratio, visible_mask, generator):
        """
        x: [N, C, T, H, W]
        visible_mask: [N, C, T, H, W] mask of visible pixels, 1=visible, 0=not visible
        """
        # select frames to encode
        x = torch.index_select(x, 2, self.t_embed_indices)

        if visible_mask is not None:
            visible_mask = torch.index_select(visible_mask, 2, self.t_embed_indices)
            # mask invisible part of x with zeros.
            x = visible_mask * x
            # [N, L] mask of patches containing some visible pixels
            visible_patch_mask = self.patchify(visible_mask, predict=False)
            visible_patch_mask = visible_patch_mask.sum(dim=-1).clip(max=1)
        else:
            visible_patch_mask = None

        # embed patches
        x = self.patch_embed(x, mask=visible_mask if self.mask_patch_embed else None)
        N, T, L, C = x.shape

        x = x.reshape(N, T * L, C)

        # masking: length -> length * mask_ratio
        x, mask, ids_keep = self.random_masking(
            x, mask_ratio, visible_patch_mask, generator
        )

        x = self.apply_pos_embed(x, ids_keep)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # remove prefix tokens
        num_prefix_tokens = int(self.cls_embed) + self.reg_tokens
        prefix = x[:, :num_prefix_tokens, :]
        x = x[:, num_prefix_tokens:, :]

        return prefix, x, mask, ids_keep

    def forward_decoder(
        self, x, mask, ids_keep, decoder_mask_ratio, img_mask, generator
    ):
        # embed tokens to match embed dims
        x = self.decoder_embed(x)

        N, L, C = x.shape
        T = self.patch_embed.t_grid_size
        H, W = self.patch_embed.grid_size

        mask_tokens = self.mask_token.expand(N, T * H * W, -1).to(x.dtype)

        if decoder_mask_ratio is None:
            # full decoding
            # scatter encoder embeddings into the sequence of mask tokens.
            x = mask_tokens.scatter(
                dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, C), src=x
            )

            decoder_ids_keep = None
            decoder_mask = mask
        else:
            # partial decoding
            # only decode unobserved patches (duh)
            decoder_patch_mask = mask

            # don't decode patches outside of img mask
            if img_mask is not None:
                img_mask = torch.index_select(img_mask, 2, self.t_embed_indices)
                img_patch_mask = self.patchify(img_mask, predict=False)
                img_patch_mask = img_patch_mask.sum(dim=-1).clip(max=1)
                decoder_patch_mask = img_patch_mask * decoder_patch_mask

            # select which tokens to decode
            mask_tokens, decoder_mask, decoder_ids_keep = self.random_masking(
                mask_tokens, decoder_mask_ratio, decoder_patch_mask, generator
            )

            # append selected mask tokens
            x = torch.cat([x, mask_tokens], dim=1)
            decoder_ids_keep = torch.cat([ids_keep, decoder_ids_keep], dim=1)

            # invert decoder mask, so that 1 = decoded unobserved patches
            decoder_mask = 1 - decoder_mask

        # pos embed, subset of ids
        x = self.apply_pos_embed(x, decoder_ids_keep, decoder=True)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]

        # reconstruct full prediction for consistency
        if decoder_mask_ratio is not None:
            zeros = torch.zeros(
                N, T * H * W, x.shape[-1], dtype=x.dtype, device=x.device
            )
            x = zeros.scatter(
                dim=1,
                index=decoder_ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1]),
                src=x,
            )

        return x, decoder_mask

    def forward_loss(self, imgs, pred, mask, img_mask):
        """
        imgs: [N, C, T, H, W]
        pred: [N, t*h*w, u*p*p*C]
        mask: [N, t*h*w], 0 is keep, 1 is remove,
        img_mask: [N, C, T, H, W], 0 is invalid, 1 is valid
        """
        N, C, T, H, W = imgs.shape
        target = torch.index_select(imgs, 2, self.t_pred_indices)
        target = self.patchify(target)

        if img_mask is not None:
            img_mask = torch.index_select(img_mask, 2, self.t_pred_indices)
            img_mask_patches = self.patchify(img_mask)
        else:
            img_mask_patches = None

        target = self.normalize(target, img_mask_patches)

        loss = (pred - target) ** 2

        # exclude invalid pixels from loss
        if img_mask is not None:
            # [N, L, D] mask of valid pixels to compute loss over
            mask = mask.unsqueeze(-1) * img_mask_patches
        else:
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def normalize(self, target, mask):
        # apply target normalization
        # target: [N, L, D]
        # mask: [N, L, D]
        if self.target_norm in {None, "none"}:
            self.norm_info = None
            return target

        N, L, D = target.shape
        T = self.patch_embed.t_grid_size
        H, W = self.patch_embed.grid_size

        target = target.reshape(N, T, H * W, D)
        mask = mask.reshape(N, T, H * W, D)

        # normalize over:
        #   - global: full spatiotemporal sequence
        #   - frame: each temporal "framelet" (number of true frames = t_pred_patch_size)
        #   - patch: each patch "tubelet"
        dim = {"global": (1, 2, 3), "frame": (2, 3), "patch": 3}[self.target_norm]

        mask_count = mask.sum(dim=dim, keepdim=True).clip(min=1)
        mean = (mask * target).sum(dim=dim, keepdim=True) / mask_count
        var = (mask * (target - mean) ** 2).sum(dim=dim, keepdim=True) / mask_count
        std = (var + 1.0e-6) ** 0.5

        target = mask * (target - mean) / std
        target = target.reshape(N, L, D)

        self.norm_info = (mask, mean, std)
        return target

    def unnormalize(self, pred):
        # pred: [N, L, D]
        if self.target_norm in {None, "none"}:
            return pred

        N, L, D = pred.shape
        T = self.patch_embed.t_grid_size
        H, W = self.patch_embed.grid_size

        mask, mean, std = self.norm_info

        pred = pred.reshape(N, T, H * W, D)
        pred = mask * (pred * std + mean)

        pred = pred.reshape(N, L, D)
        return pred

    def forward(
        self,
        imgs,
        mask_ratio=0.75,
        decoder_mask_ratio=None,
        img_mask=None,
        visible_mask=None,
        generator=None,
    ):
        if visible_mask is None:
            visible_mask = img_mask
        elif img_mask is not None:
            visible_mask = img_mask * visible_mask

        if img_mask is not None:
            img_mask = img_mask.expand_as(imgs)
        if visible_mask is not None:
            visible_mask = visible_mask.expand_as(imgs)

        prefix, latent, mask, ids_keep = self.forward_encoder(
            imgs, mask_ratio, visible_mask, generator
        )
        pred, decoder_mask = self.forward_decoder(
            latent, mask, ids_keep, decoder_mask_ratio, img_mask, generator
        )
        loss = self.forward_loss(imgs, pred, decoder_mask, img_mask)
        return loss, pred, mask, decoder_mask

    def forward_embedding(
        self,
        imgs,
        mask_ratio=0.0,
        img_mask=None,
        visible_mask=None,
        generator=None,
    ):
        if visible_mask is None:
            visible_mask = img_mask
        elif img_mask is not None:
            visible_mask = img_mask * visible_mask

        if img_mask is not None:
            img_mask = img_mask.expand_as(imgs)
        if visible_mask is not None:
            visible_mask = visible_mask.expand_as(imgs)

        prefix, latent, mask, ids_keep = self.forward_encoder(
            imgs, mask_ratio, visible_mask, generator
        )

        N, _, C = latent.shape
        T = self.patch_embed.t_grid_size
        H, W = self.patch_embed.grid_size

        zeros = torch.zeros(N, T * H * W, C, dtype=latent.dtype, device=latent.device)
        latent = zeros.scatter(
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, C),
            src=latent,
        )

        cls_offset = int(self.cls_embed)
        cls_token = prefix[:, :cls_offset]
        reg_tokens = prefix[:, cls_offset:]

        return cls_token, reg_tokens, latent

    @torch.no_grad()
    def forward_masked_recon(self, imgs, pred, mask, img_mask=None, normalize=True):
        # imgs: [N, C, T, H, W]
        # pred: [N, t*h*w, u*p*p*C]
        # mask: [N, t*h*w], 0 is keep, 1 is remove,
        N, C, T, H, W = imgs.shape
        input = torch.index_select(imgs, 2, self.t_embed_indices)
        target = torch.index_select(imgs, 2, self.t_pred_indices)

        # process the img_mask to match the target.
        if img_mask is not None:
            img_mask = img_mask.expand_as(imgs)
            img_mask = torch.index_select(img_mask, 2, self.t_pred_indices)

        # align input to target if the number of frames are not the same
        T_embed = input.shape[2]
        T_pred = target.shape[2]
        if T_embed != T_pred:
            t_sub_indices = (
                torch.linspace(0, T_embed - 1, T_pred).long().to(input.device)
            )
            input = torch.index_select(input, 2, t_sub_indices)

        # this caches patch and normalization info for unpatchify and unnormalize
        target_patches = self.patchify(target)
        if img_mask is not None:
            img_mask_patches = self.patchify(img_mask)
        else:
            img_mask_patches = None
        # apply target normalization
        target_patches = self.normalize(target_patches, img_mask_patches)

        if normalize:
            target = self.unpatchify(target_patches)

        input = torch.einsum("ncthw->nthwc", input)
        target = torch.einsum("ncthw->nthwc", target)

        if not normalize:
            pred = self.unnormalize(pred)
        pred = self.unpatchify(pred)
        pred = torch.einsum("ncthw->nthwc", pred)

        ph, pw = self.patch_embed.patch_size
        pt = self.t_pred_patch_size
        mask = mask.unsqueeze(-1).expand(
            -1, -1, pt * ph * pw * C
        )  # (N, T*H*W, u*p*p*c)
        mask = self.unpatchify(mask)  # 1 is removing, 0 is keeping

        mask = torch.einsum("ncthw->nthwc", mask)

        # masked image
        im_masked = input * (1 - mask)

        # MAE reconstruction pasted with visible patches
        im_paste = target * (1 - mask) + pred * mask

        # drop channels dim for mask
        if img_mask is not None:
            img_mask = img_mask[:, 0]  # (N, T, H, W)

        return target, pred, mask, im_masked, im_paste, img_mask

    @torch.no_grad()
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
        N, C, T, H, W = imgs.shape

        # get patch dimensions
        T_patches = self.patch_embed.t_grid_size
        H_patches = self.patch_embed.grid_size[0]
        W_patches = self.patch_embed.grid_size[1]
        # total_patches = T_patches * H_patches * W_patches - commented out because it's not used

        # we assume a shared img_mask for all samples for simplicity.
        if img_mask is not None:
            assert img_mask.shape == (H_patches, W_patches), (
                "invalid img_mask for denoising"
            )
        else:
            img_mask = torch.ones(H_patches, W_patches, device=imgs.device)

        # generate random patch permutation (using generator for reproducibility)
        valid_ids = img_mask.flatten().nonzero().flatten()
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
                img_mask,
                patch_permutation,
                T_patches,
                H_patches,
                W_patches,
            )

            # run mae. note: masking_ratio is 0.0 since we are already choosing which patches to mask
            prefix, latent, mask, ids_keep = self.forward_encoder(
                imgs, 0.0, run_visible_mask, generator
            )
            pred, decoder_mask = self.forward_decoder(
                latent,
                mask,
                ids_keep,
                decoder_mask_ratio=0.0,
                img_mask=None,  # bypass temporal masking for explicit denoising
                generator=generator,
            )

            # unnormalize prediction before storing
            pred_unnorm = self.unnormalize(pred)
            reconstructions.append(pred_unnorm)
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

        # avoid division by zero: if no reconstructions, keep original (shouldn't happen)
        final_patches = torch.where(
            weight_sum > 0,
            weighted_sum / weight_sum,
            stacked_reconstructions[0],  # fallback to first reconstruction
        )

        # convert back to image space using unpatchify
        # need to call patchify with target frames to set correct patch_info for unpatchify
        target_imgs = torch.index_select(imgs, 2, self.t_pred_indices)
        _ = self.patchify(target_imgs)  # sets self.patch_info for prediction dimensions
        denoised_imgs = self.unpatchify(final_patches)

        return denoised_imgs

    def _generate_visible_mask(
        self,
        imgs,
        run_idx,
        total_runs,
        img_mask,
        patch_permutation,
        T_patches,
        H_patches,
        W_patches,
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

        # Call patchify to set self.patch_info
        _ = self.patchify(imgs, predict=True)
        N, C, T, H, W, ph, pw, u, t, h, w = self.patch_info

        # Create patch_mask in patch space
        patch_mask = torch.zeros(
            (N, t * h * w), device=imgs.device, dtype=torch.float32
        )
        for idx in visible_patch_indices:
            patch_mask[:, idx] = 1.0

        # Expand to include voxel content: (N, t*h*w, u*ph*pw*C)
        patch_mask_expanded = patch_mask.unsqueeze(-1).expand(-1, -1, u * ph * pw * C)
        visible_mask = self.unpatchify(patch_mask_expanded)

        return visible_mask


def _parse_indices(indices: str | list[int] | None, size: int) -> torch.Tensor:
    if indices is None:
        indices = torch.arange(0, size)
    elif isinstance(indices, str):
        start, stop, step = _parse_slice(indices)
        indices = torch.arange(start, stop or size, step)
    elif isinstance(indices, list):
        indices = torch.as_tensor(indices)
    else:
        raise TypeError(f"Unsupported indices: {indices}")
    return indices


def _parse_slice(slc: str) -> tuple[int, int | None, int]:
    """Parse a slice expression like '0:8:2' to a tuple of (start, stop, step)."""
    values = [ast.literal_eval(val) for val in slc.strip().split(":")]
    if len(values) == 2:
        values.append(1)
    return tuple(values)


def mae_vit_nano_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=192,
        depth=3,
        num_heads=3,
        decoder_embed_dim=192,
        decoder_num_heads=3,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_micro_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=192,
        depth=6,
        num_heads=3,
        decoder_embed_dim=192,
        decoder_num_heads=3,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_tiny_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        decoder_embed_dim=192,
        decoder_num_heads=3,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_small_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        decoder_embed_dim=384,
        decoder_num_heads=6,
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
