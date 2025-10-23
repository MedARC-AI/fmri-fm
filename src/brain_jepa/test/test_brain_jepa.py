# Load a brain jepa model from a checkpoint
# Load data
# Do forward
# Verify that it looks correct

import torch
from brain_jepa.brain_jepa import VisionTransformer


# Something like this.


def test():
    # We use a bunch of default args
    model = VisionTransformer(
        model_name="vit_base",
        num_classes=2,
        global_pool=True,
        device="cuda",
        add_w="mapping",
    )

    checkpoint = torch.load("./jepa-ep300.pth.tar", map_location="cpu")
    checkpoint_model = checkpoint["target_encoder"]
    state_dict = model.state_dict()

    new_checkpoint_model = {}
    for key in checkpoint_model.keys():
        new_key = key.replace("module.", "encoder.")  # Remove 'module.' from each key
        new_checkpoint_model[new_key] = checkpoint_model[key]

    for k in ["head.weight", "head.bias"]:
        if (
            k in new_checkpoint_model
            and new_checkpoint_model[k].shape != state_dict[k].shape
        ):
            print(f"Removing key {k} from pretrained checkpoint")
            del new_checkpoint_model[k]

        # load pre-trained model
    msg = model.load_state_dict(new_checkpoint_model, strict=False)
    print(msg)

    try:
        assert set(msg.missing_keys) == {
            "head.weight",
            "head.bias",
            "fc_norm.weight",
            "fc_norm.bias",
        }
    except Exception:
        assert set(msg.missing_keys) == {"head.weight", "head.bias"}
        # manually initialize fc layer: following MoCo v3
        # trunc_normal_(model.head.weight, std=0.01)

    # 450, 160?
    # data = torch.randn(1, 1, 450, 160, dtype=torch.float32).to("cuda")
    data = torch.load("data.pt")
    expected_res = torch.load("res.pt")
    with torch.no_grad():
        model = model.to("cuda")
        res = model.testing_forward(data)
        print(res.shape)
        print(res)
    print("mean difference between real and expected:", torch.mean(res - expected_res))
