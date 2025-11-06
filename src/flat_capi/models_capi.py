from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn, norm_except_dim
from torch.nn import Module
from torch.nn.parameter import Parameter, UninitializedParameter


class Mlp(nn.Module):
    """Two-layer MLP with GELU used inside transformer blocks."""
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        mlp_ratio: int | float | None = 4,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_features is None:
            assert mlp_ratio is not None
            hidden_features = int(in_features * mlp_ratio)
        else:
            assert mlp_ratio is None
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Rope(nn.Module):
    """2D rotary position embedding applied per-head to Q/K projections."""
    def __init__(self, dim: int, max_freq: float | int = 7, min_freq: float | int = 7e-4) -> None:
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq
        self.min_freq = min_freq
        self.freqs = nn.Parameter(torch.empty(2, self.dim))

    def _device_weight_init(self):
        freqs_1d = self.max_freq * (self.max_freq / self.min_freq) ** torch.linspace(0, -1, self.dim // 4)
        freqs_1d = torch.cat([freqs_1d, freqs_1d])
        freqs_2d = torch.zeros(2, self.dim)
        freqs_2d[0, : self.dim // 2] = freqs_1d
        freqs_2d[1, -self.dim // 2 :] = freqs_1d
        self.freqs.data.copy_(freqs_2d * 2 * torch.pi)

    def forward(self, x: Tensor, coords: Tensor) -> Tensor:
        angle = coords @ self.freqs
        return x * angle.cos() + rotate_half(x) * angle.sin()


class Attention(nn.Module):
    """Multi-head attention with optional cross-attention and RoPE."""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        context_dim: int | None = None,
        rope_kwargs: Mapping = {},
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        context_dim = context_dim or dim
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.rope = Rope(dim=self.head_dim, **rope_kwargs)

    def forward(
        self,
        x: Tensor,
        coords: Tensor,
        context: Tensor | None = None,
        context_coords: Tensor | None = None,
    ) -> Tensor:
        if context is None or context_coords is None:
            context = x
            context_coords = coords
        b, n_q, d = x.shape
        h = self.num_heads
        q = self.q_proj(x).reshape(b, n_q, h, d // h).transpose(1, 2)
        k = self.k_proj(context).reshape(b, context.shape[1], h, d // h).transpose(1, 2)
        v = self.v_proj(context).reshape(b, context.shape[1], h, d // h).transpose(1, 2)
        q = self.rope(q, coords[:, None, :, :])
        k = self.rope(k, context_coords[:, None, :, :])
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape([b, n_q, d])
        x = self.proj(x)
        return x


class NaiveResidual(nn.Module):
    """Pre-norm residual wrapper with optional stochastic depth."""
    def __init__(self, drop_prob: float | int, norm: nn.Module, fn: nn.Module):
        super().__init__()
        self.norm = norm
        self.fn = fn
        self.keep_prob = 1 - drop_prob

    def forward(self, x: Tensor, **kwargs: Tensor | None) -> Tensor:
        fn_out = self.fn(self.norm(x), **kwargs)
        if self.keep_prob == 1.0 or not self.training:
            return x + fn_out
        mask = fn_out.new_empty(x.shape[0]).bernoulli_(self.keep_prob)[:, None, None]
        return x + fn_out * mask / self.keep_prob


class EfficientResidual(NaiveResidual):
    """Residual wrapper that sparsifies forward passes during training for efficiency."""
    def forward(self, x: Tensor, **kwargs: Tensor | None) -> Tensor:
        if self.keep_prob == 1.0 or not self.training:
            return x + self.fn(self.norm(x), **kwargs)
        b, _, _ = x.shape
        n_keep = max(int(b * self.keep_prob), 1)
        indices = torch.randperm(b, device=x.device)[:n_keep]
        for k, v in kwargs.items():
            if v is not None:
                kwargs[k] = v[indices]
        src = self.fn(self.norm(x[indices]), **kwargs)
        if src.dtype != x.dtype:
            src = src.to(dtype=x.dtype)
        return torch.index_add(x, dim=0, source=src, index=indices, alpha=b / n_keep)


class Block(nn.Module):
    """Transformer block with attention (with RoPE) and MLP."""
    def __init__(
        self,
        dim: int,
        drop_path: float | int,
        norm_layer: nn.Module,
        context_dim: int | None,
        drop_path_type: str = "efficient",
        attn_kwargs: Mapping = {},
    ) -> None:
        super().__init__()
        residual_module = {"naive": NaiveResidual, "efficient": EfficientResidual}[drop_path_type]
        self.residual1 = residual_module(drop_path, norm_layer(dim), Attention(dim, context_dim=context_dim, **attn_kwargs))
        self.residual2 = residual_module(drop_path, norm_layer(dim), Mlp(in_features=dim))

    def forward(
        self,
        x: Tensor,
        context: Tensor | None = None,
        coords: Tensor | None = None,
        context_coords: Tensor | None = None,
    ) -> Tensor:
        x = self.residual1(x, context=context, coords=coords, context_coords=context_coords)
        x = self.residual2(x)
        return x


class Transformer(nn.Module):
    """Stack of transformer blocks, returning selected intermediate layers."""
    def __init__(
        self,
        embed_dim: int,
        norm_layer: nn.Module,
        depth: int,
        drop_path_rate: float | int,
        context_dim: int | None = None,
        block_kwargs: Mapping[str, Any] = {},
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_blocks = depth
        self.blocks = nn.ModuleList(
            [
                Block(dim=embed_dim, drop_path=drop_path_rate, norm_layer=norm_layer, context_dim=context_dim, **block_kwargs)
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: Tensor,
        return_layers: set[int],
        contexts: list[Tensor] | None = None,
        coords: Tensor | None = None,
        context_coords: Tensor | None = None,
    ) -> dict[int, Tensor]:
        outputs: dict[int, Tensor] = {}
        if 0 in return_layers:
            outputs[0] = x
        for blk_idx, blk in enumerate(self.blocks):
            context = contexts[blk_idx] if contexts is not None else None
            x = blk(x, context=context, coords=coords, context_coords=context_coords)
            if blk_idx + 1 in return_layers:
                outputs[blk_idx + 1] = x
        return outputs


class EncoderDecoder(nn.Module):
    """CAPI encoder-decoder backbone producing registers and a feature map."""
    def __init__(
        self,
        patch_size: int = 14,
        in_chans: int = 3,
        norm_layer_type: str = "RMSNorm",
        n_registers: int = 16,
        register_coordinates: str = "edge",
        scale_reg_to_visible: bool = True,
        transformers_kwargs: Mapping[str, Any] = {},
        encoder_kwargs: Mapping[str, Any] = {},
        decoder_kwargs: Mapping[str, Any] = {},
        norm_layer_kwargs: Mapping[str, Any] = {"eps": 1e-5},
        final_norm_kwargs: Mapping[str, Any] = {"elementwise_affine": False},
        out_layer: int = -1,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_prefix_tokens = self.n_registers = n_registers
        self.register_coordinates = register_coordinates
        self.scale_reg_to_visible = scale_reg_to_visible

        norm_layer = partial(getattr(torch.nn, norm_layer_type), **norm_layer_kwargs)
        self.encoder = Transformer(**transformers_kwargs, **encoder_kwargs, norm_layer=norm_layer)
        self.decoder = Transformer(**transformers_kwargs, **decoder_kwargs, context_dim=self.encoder.embed_dim, norm_layer=norm_layer)
        self.embed_dim = self.encoder.embed_dim
        self.pred_dim = self.decoder.embed_dim
        self.n_blocks = len(self.encoder.blocks)
        self.out_layer = out_layer % (len(self.encoder.blocks) + 1)

        self.mask_token = nn.Parameter(torch.empty(1, self.decoder.embed_dim))
        self.registers = nn.Parameter(torch.empty(1, n_registers, self.encoder.embed_dim))
        self.patch_embed = nn.Conv2d(in_chans, self.encoder.embed_dim, kernel_size=patch_size, stride=patch_size)
        self.enc_norm = norm_layer(self.embed_dim, **final_norm_kwargs)
        self.dec_norm = norm_layer(self.decoder.embed_dim, **final_norm_kwargs)

    def init_weights(self) -> "EncoderDecoder":
        self.patch_embed.reset_parameters()
        w = self.patch_embed.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.normal_(self.registers, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(_init_weights)
        return self

    def prepare_tokens_and_drop(self, x: Tensor, visible_indices: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        b, _, h, w = x.shape
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        coord_x = torch.linspace(0, 1, h // self.patch_size, device=x.device, dtype=x.dtype)
        coord_y = torch.linspace(0, 1, w // self.patch_size, device=x.device, dtype=x.dtype)
        coords_all = torch.cartesian_prod(coord_x, coord_y)
        coords_all = coords_all[None].expand(b, -1, -1)
        if visible_indices is not None:
            coords = coords_all.flatten(0, 1)[visible_indices].reshape(b, -1, 2)
            x = x.flatten(0, 1)[visible_indices].reshape(b, -1, self.embed_dim)
        else:
            coords = coords_all
        if self.register_coordinates == "zeros":
            reg_coords = torch.zeros(b, self.n_registers, 2, device=x.device, dtype=x.dtype)
        elif self.register_coordinates == "edge":
            reg_coords = self.get_edge_coordinates(self.n_registers, x.dtype, x.device).expand(b, -1, -1)
        else:
            raise ValueError(self.register_coordinates)
        if self.scale_reg_to_visible:
            mi, ma = (coords.min(dim=1, keepdim=True).values, coords.max(dim=1, keepdim=True).values)
            reg_coords = mi + reg_coords * (ma - mi)
        x = torch.cat([self.registers.expand(b, -1, -1), x], dim=1)
        coords = torch.cat([reg_coords, coords], dim=1)
        return x, coords, coords_all

    def get_edge_coordinates(self, n: int, dtype: torch.dtype, device: torch.device) -> Tensor:
        side = n // 4
        assert n == 4 * side
        reg_coords = torch.zeros(1, n, 2, dtype=dtype, device=device)
        c = torch.arange(side, dtype=dtype, device=device) / side
        reg_coords[:, 0 * side : 1 * side, 0] = c
        reg_coords[:, 0 * side : 1 * side, 1] = 0
        reg_coords[:, 1 * side : 2 * side, 0] = 1
        reg_coords[:, 1 * side : 2 * side, 1] = c
        reg_coords[:, 2 * side : 3 * side, 0] = 1 - c
        reg_coords[:, 2 * side : 3 * side, 1] = 1
        reg_coords[:, 3 * side : 4 * side, 0] = 0
        reg_coords[:, 3 * side : 4 * side, 1] = 1 - c
        return reg_coords

    def forward_features(
        self,
        x: Tensor,
        visible_indices: Tensor | None,
        predict_indices: Tensor | None,
        enc_layer: int,
        dec_layer: int | None,
    ) -> tuple[Tensor, Tensor | None]:
        b, _, _, _ = x.shape
        x, coords_enc, coords_all = self.prepare_tokens_and_drop(x, visible_indices)
        enc_layers = {enc_layer}
        if dec_layer is not None:
            enc_layers.add(len(self.encoder.blocks))
        encoder_outputs = self.encoder(x, coords=coords_enc, return_layers=enc_layers)
        encoder_outputs = {k: self.enc_norm(v) for k, v in encoder_outputs.items()}
        if dec_layer is not None:
            coords_dec = coords_all.flatten(0, 1)[predict_indices].reshape(b, -1, 2)
            decoder_outputs = self.decoder(
                self.mask_token[None].expand(*coords_dec.shape[:2], -1),
                contexts=[encoder_outputs[len(self.encoder.blocks)]] * self.decoder.n_blocks,
                coords=coords_dec,
                context_coords=coords_enc,
                return_layers={dec_layer},
            )
            dec_out = self.dec_norm(decoder_outputs[dec_layer])
        else:
            dec_out = None
        enc_out = encoder_outputs[enc_layer]
        return (enc_out, dec_out)

    def forward_pretrain(
        self,
        x: Tensor,
        visible_indices: Tensor | None = None,
        predict_indices: Tensor | None = None,
        do_prediction: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        encoder_output, decoder_output = self.forward_features(
            x, visible_indices, predict_indices, enc_layer=self.out_layer, dec_layer=len(self.decoder.blocks) if do_prediction else None
        )
        if decoder_output is not None:
            decoder_output = decoder_output.flatten(0, 1)
        return encoder_output[:, self.num_prefix_tokens :], decoder_output

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        b, _, h, w = x.shape
        enc_out, dec_out = self.forward_features(x, None, None, enc_layer=self.out_layer, dec_layer=1)
        global_repr = dec_out.mean(dim=1)  # type: ignore[arg-type]
        registers = enc_out[:, : self.num_prefix_tokens]
        feature_map = enc_out[:, self.num_prefix_tokens :].reshape(b, h // self.patch_size, w // self.patch_size, self.embed_dim)
        return (global_repr, registers, feature_map)


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm | nn.RMSNorm) and m.elementwise_affine:
        nn.init.constant_(m.weight, 1.0)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if hasattr(m, "_device_weight_init"):
        m._device_weight_init()


class WeightNorm:
    """Lightweight functional weight normalization utility."""
    name: str
    dim: int

    def __init__(self, name: str, dim: int) -> None:
        if dim is None:
            dim = -1
        self.name = name
        self.dim = dim

    def compute_weight(self, module: Module) -> Any:
        g = getattr(module, self.name + "_g")
        v = getattr(module, self.name + "_v")
        return v * (g / (norm_except_dim(v, dim=self.dim) + 1e-8))

    @staticmethod
    def apply(module: Module, name: str = "weight", dim: int = 0) -> "WeightNorm":
        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, WeightNorm) and hook.name == name:
                raise RuntimeError(f"Double weight_norm hook on the same parameter {name}")

        if dim is None:
            dim = -1

        fn = WeightNorm(name, dim)

        weight = getattr(module, name)
        assert not isinstance(weight, UninitializedParameter)
        del module._parameters[name]

        module.register_parameter(name + "_g", Parameter(norm_except_dim(weight, 2, dim).data))
        module.register_parameter(name + "_v", Parameter(weight.data))
        setattr(module, name, fn.compute_weight(module))

        module.register_forward_pre_hook(fn)
        return fn

    def remove(self, module: Module) -> None:
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + "_g"]
        del module._parameters[self.name + "_v"]
        setattr(module, self.name, Parameter(weight.data))

    def __call__(self, module: Module, inputs: Any) -> None:
        setattr(module, self.name, self.compute_weight(module))


def weight_norm(module: Module, name: str = "weight", dim: int = 0) -> Module:
    """Apply weight normalization to a module parameter in-place."""
    WeightNorm.apply(module, name, dim)
    return module


class L2NormLinear(nn.Module):
    """Linear layer on L2-normalized inputs; optional weight normalization."""
    def __init__(self, in_dim: int, out_dim: int, *, do_weight_norm: bool = True) -> None:
        super().__init__()
        self.last_layer = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.trunc_normal_(self.last_layer.weight, std=0.02)
        if do_weight_norm:
            self.last_layer = weight_norm(self.last_layer)
            if hasattr(self.last_layer, "weight_g"):
                self.last_layer.weight_g.data.fill_(1)

    def forward(self, x: Tensor) -> Tensor:
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, eps=eps)
        return self.last_layer(x)


exp_max_values = {
    torch.float16: 0,
    torch.float32: 50,
    torch.float64: 50,
    torch.bfloat16: 50,
}


def stable_exp(M: Tensor) -> Tensor:
    """Exponentiation stabilized by subtracting the global maximum across ranks."""
    shift = M.max(dim=-2, keepdim=True).values
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(shift, torch.distributed.ReduceOp.MAX)
    M = M + (exp_max_values[M.dtype] - shift)
    return M.exp()


def reduced_sum(*args, **kwargs):
    """All-reduced sum when distributed, otherwise local sum."""
    summed = torch.sum(*args, **kwargs)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(summed)
    return summed


@torch.no_grad()
def sinkhorn_knopp(M: Tensor, n_iterations: int, eps: float | int = 1e-8) -> Tensor:
    """Stabilized Sinkhorn-Knopp normalization over the last two dims."""
    M = stable_exp(M)
    for _ in range(n_iterations):
        M = M / (reduced_sum(M, dim=-2, keepdim=True) + eps)
        M = M / (torch.sum(M, dim=-1, keepdim=True) + eps)
    return M


class OnlineClustering(nn.Module):
    """Online clustering head producing assignments and a contrastive-like loss."""
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        bias: bool,
        n_sk_iter: int,
        target_temp: float | int,
        pred_temp: float | int,
        positionwise_sk: bool = True,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.n_sk_iter = n_sk_iter
        self.target_temp = target_temp
        self.pred_temp = pred_temp
        self.positionwise_sk = positionwise_sk
        self.layer = nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.normal_(self.layer.weight, std=1)
        if bias:
            torch.nn.init.zeros_(self.layer.bias)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x_n = nn.functional.normalize(x, dim=-1, p=2, eps=1e-7)
        logits = self.layer(x_n)
        if not self.positionwise_sk:
            logits = logits.flatten(0, -2)
        assignments = sinkhorn_knopp(logits.detach() / self.target_temp, n_iterations=self.n_sk_iter)
        tgt = assignments.flatten(0, -2).float()
        pred = logits.flatten(0, -2).float()
        loss = -torch.sum(tgt * F.log_softmax(pred / self.pred_temp, dim=-1), dim=-1).mean()
        return assignments.detach(), loss


def vit_l14_capi(**kwargs) -> EncoderDecoder:
    transformers_kwargs = kwargs.get("transformers_kwargs", {})
    transformers_kwargs.setdefault("embed_dim", 1024)
    transformers_kwargs.setdefault("drop_path_rate", 0.2)
    transformers_kwargs.setdefault("block_kwargs", {"attn_kwargs": {"num_heads": 16}})
    encoder_kwargs = kwargs.get("encoder_kwargs", {"depth": 24})
    decoder_kwargs = kwargs.get("decoder_kwargs", {"depth": 12})
    model = EncoderDecoder(
        patch_size=kwargs.get("patch_size", 14),
        in_chans=kwargs.get("in_chans", 3),
        transformers_kwargs=transformers_kwargs,
        encoder_kwargs=encoder_kwargs,
        decoder_kwargs=decoder_kwargs,
        out_layer=kwargs.get("out_layer", -1),
    )
    return model


def vit_b16_capi(**kwargs) -> EncoderDecoder:
    transformers_kwargs = kwargs.get("transformers_kwargs", {})
    transformers_kwargs.setdefault("embed_dim", 768)
    transformers_kwargs.setdefault("drop_path_rate", 0.2)
    transformers_kwargs.setdefault("block_kwargs", {"attn_kwargs": {"num_heads": 12}})
    encoder_kwargs = kwargs.get("encoder_kwargs", {"depth": 12})
    decoder_kwargs = kwargs.get("decoder_kwargs", {"depth": 6})
    model = EncoderDecoder(
        patch_size=kwargs.get("patch_size", 16),
        in_chans=kwargs.get("in_chans", 3),
        transformers_kwargs=transformers_kwargs,
        encoder_kwargs=encoder_kwargs,
        decoder_kwargs=decoder_kwargs,
        out_layer=kwargs.get("out_layer", -1),
    )
    return model


def vit_l16_capi(**kwargs) -> EncoderDecoder:
    """Alias for ViT-L with patch size 16 by default.
    If patch_size is provided in kwargs, it is respected.
    """
    if "patch_size" not in kwargs:
        kwargs["patch_size"] = 16
    return vit_l14_capi(**kwargs)

