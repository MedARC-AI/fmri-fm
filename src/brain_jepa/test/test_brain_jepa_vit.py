import torch

from brain_jepa.helper import init_model
from brain_jepa.gradient import load_gradient


def test_brain_jepa_vit():
    gradient = load_gradient()
    device = torch.device("cpu")

    encoder, predictor = init_model(
        device=device,
        patch_size=16,
        model_name="vit_small",
        crop_size=[450, 160],
        gradient_pos_embed=gradient,
        add_w="mapping",
    )

    # [N, C, H, W] = [N, 1, R, T]
    img = torch.randn(2, 1, 450, 160)

    out = encoder.forward(img)

    # nb, sequence length is R * T / patch_size, bc 2D patch size is (1, 16)
    assert out.shape == (2, 450 * 10, 384)
    assert not torch.any(torch.isnan(out))
