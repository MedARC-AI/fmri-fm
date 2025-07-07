import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import to_2tuple
from util.logging import master_print as print

EPS = 1e-6


class MaskedLinear(nn.Linear):
    """
    Linear layer that scales output to account for size of observed mask.
    """
    def forward(
        self, input: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        input: [..., D]
        mask: [..., D], 1 = observed, 0 = unobserved
        """
        input = input * mask
        output = F.linear(input, self.weight, self.bias)
        if mask is not None:
            obs_rate = mask.mean(dim=-1, keepdim=True)
            output = output / (obs_rate + EPS)
        return output


class MaskedAutoencoderLinear(nn.Module):
    """
    Linear Masked Autoencoder.

    Copied with minor modifications from MaskedAutoencoderViT.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        num_frames=16,
        t_patch_size=4,
        t_pred_patch_size=1,
        use_masked_linear=True,
        framewise=False,
        **kwargs,
    ):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.num_frames = num_frames
        self.t_patch_size = t_patch_size
        self.t_pred_patch_size = t_pred_patch_size
        self.use_masked_linear = use_masked_linear
        self.framewise = framewise

        T = num_frames
        H, W = self.img_size
        ph, pw = self.patch_size
        u = t_patch_size
        self.num_patches = (T // u) * (H // ph) * (W // pw)

        assert t_patch_size % t_pred_patch_size == 0
        t_step = t_patch_size // t_pred_patch_size

        # (C, T, H, W)
        if framewise:
            in_features = math.prod((in_chans, *self.img_size))
            out_features = math.prod((in_chans, *self.img_size))
        else:
            in_features = math.prod((in_chans, num_frames, *self.img_size))
            out_features = math.prod((in_chans, num_frames // t_step, *self.img_size))

        encoder_linear_layer = MaskedLinear if use_masked_linear else nn.Linear
        self.linear_encoder = encoder_linear_layer(in_features, embed_dim)
        self.linear_decoder = nn.Linear(embed_dim, out_features)

        self.initialize_weights()

        print(
            f"img_size {self.img_size} patch_size {self.patch_size} "
            f"frames {num_frames} t_patch_size {t_patch_size}"
        )
        print("model initialized")

    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # nb, not using the vit init for linear weight, since the dimension can be
            # very large, using the default sqrt(1 / in_features) scaled init.
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def patchify(self, imgs, predict=True):
        """
        imgs: (N, C, T, H, W)
        x: (N, L, patch_size**2 *C)

        use the predictor t patch size when predict=True, and the patch embedding t
        patch size otherwise.
        """
        N, C, T, H, W = imgs.shape
        ph, pw = self.patch_size
        u = self.t_pred_patch_size if predict else self.t_patch_size
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

        # mask ratio is relative to the (minimum) size of the visible mask
        if visible_patch_mask is not None:
            total_patches = visible_patch_mask.sum(dim=1).min().item()
        else:
            total_patches = L
        len_keep = int(total_patches * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # shift invisible patches to not be selected
        if visible_patch_mask is not None:
            noise = noise + (1.0 - visible_patch_mask)

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], dtype=x.dtype, device=x.device)
        mask = mask.scatter(
            dim=1,
            index=ids_keep,
            src=torch.zeros([N, len_keep], dtype=x.dtype, device=x.device),
        )
        return mask

    def forward_encoder(self, x, mask_ratio, visible_mask):
        """
        x: [N, C, T, H, W]
        visible_mask: [N, C, T, H, W] mask of visible pixels, 1=visible, 0=not visible
        """
        if visible_mask is not None:
            # mask invisible part of x with zeros.
            x = visible_mask * x
            # [N, L] mask of patches containing some visible pixels
            visible_patch_mask = self.patchify(visible_mask, predict=False)
            visible_patch_mask = visible_patch_mask.sum(dim=-1).clip(max=1)
        else:
            visible_patch_mask = None

        # [N, L, D]
        x = self.patchify(x, predict=False)

        # [N, L], 0 is keep, 1 is remove
        mask = self.random_masking(x, mask_ratio, visible_patch_mask)

        observed_mask = 1 - mask
        observed_mask = observed_mask.unsqueeze(-1).expand_as(x)
        x = observed_mask * x

        if self.framewise:
            # [N, C, T, H, W]
            x = self.unpatchify(x)
            observed_mask = self.unpatchify(observed_mask)
            # [N, T, C * H * W]
            x = x.transpose(1, 2).flatten(2)
            observed_mask = observed_mask.transpose(1, 2).flatten(2)
        else:
            # [N, L * D]
            x = x.flatten(1)
            observed_mask = observed_mask.flatten(1)

        if self.use_masked_linear:
            x = self.linear_encoder(x, mask=observed_mask)
        else:
            x = self.linear_encoder(x)
        return x, mask

    def forward_decoder(self, x):
        x = self.linear_decoder(x)

        if self.framewise:
            # [N, T, C * H * W]
            N, T, D = x.shape
            C = self.in_chans
            H, W = self.img_size
            # [N, C, T, H, W]
            x = x.view(N, T, C, H, W).transpose(1, 2)

            t_step = self.t_patch_size // self.t_pred_patch_size
            t_indices = torch.arange(0, T, t_step, device=x.device)
            x = torch.index_select(x, 2, t_indices)

            x = self.patchify(x, predict=True)
        else:
            # [N, L * D]
            x = x.view(x.shape[0], self.num_patches, -1)
        return x

    def forward_loss(self, imgs, pred, mask, img_mask):
        """
        imgs: [N, C, T, H, W]
        pred: [N, t*h*w, u*p*p*C]
        mask: [N, t*h*w], 0 is keep, 1 is remove,
        img_mask: [N, C, T, H, W], 0 is invalid, 1 is valid
        """
        N, C, T, H, W = imgs.shape
        t_step = self.t_patch_size // self.t_pred_patch_size
        t_indices = torch.arange(0, T, t_step, device=imgs.device)
        _imgs = torch.index_select(imgs, 2, t_indices)

        target = self.patchify(_imgs)

        loss = (pred - target) ** 2

        # exclude invalid pixels from loss
        if img_mask is not None:
            img_mask = torch.index_select(img_mask, 2, t_indices)
            img_mask_patches = self.patchify(img_mask)
            # [N, L, D] mask of valid pixels to compute loss over
            mask = mask.unsqueeze(-1) * img_mask_patches
        else:
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(
        self,
        imgs,
        mask_ratio=0.75,
        decoder_mask_ratio=None,
        img_mask=None,
        visible_mask=None,
    ):
        assert decoder_mask_ratio is None, "decoder_mask_ratio not used"

        if visible_mask is None:
            visible_mask = img_mask
        elif img_mask is not None:
            visible_mask = img_mask * visible_mask

        if img_mask is not None:
            img_mask = img_mask.expand_as(imgs)
        if visible_mask is not None:
            visible_mask = visible_mask.expand_as(imgs)

        latent, mask = self.forward_encoder(
            imgs, mask_ratio, visible_mask
        )
        decoder_mask = mask  # for consistency with mae vit
        pred = self.forward_decoder(latent)
        loss = self.forward_loss(imgs, pred, decoder_mask, img_mask)
        return loss, pred, mask, decoder_mask

    @torch.no_grad()
    def forward_masked_recon(self, imgs, pred, mask, img_mask=None):
        # imgs: [N, C, T, H, W]
        # pred: [N, t*h*w, u*p*p*C]
        # mask: [N, t*h*w], 0 is keep, 1 is remove,
        N, C, T, H, W = imgs.shape
        t_step = self.t_patch_size // self.t_pred_patch_size
        t_indices = torch.arange(0, T, t_step, device=imgs.device)
        target = torch.index_select(imgs, 2, t_indices)

        # this caches patch info for unpatchify
        # necessary if batch size is different from training
        self.patchify(target)

        target = torch.einsum("ncthw->nthwc", target)

        pred = self.unpatchify(pred)
        pred = torch.einsum("ncthw->nthwc", pred)

        ph, pw = self.patch_size
        pt = self.t_pred_patch_size
        mask = mask.unsqueeze(-1).expand(-1, -1, pt * ph * pw * C)  # (N, T*H*W, u*p*p*c)
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


def mae_linear_patch16(**kwargs):
    model = MaskedAutoencoderLinear(patch_size=16, **kwargs)
    return model
