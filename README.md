# MedARC - fMRI Foundation Model

*In-progress -- this repo is under active development by [MedARC](https://www.medarc.ai). [Join our Discord](https://discord.com/invite/CqsMthnauZ) to get involved!*

<p align="center">
  <img src=".github/fMRI MAE ViT.svg" width="600">
</p>

## Installation

Clone the repo, install [uv](https://docs.astral.sh/uv/getting-started/installation/), and run

```bash
uv sync
```

This will create a new virtual environment for the project with all the required dependencies. Activate the environment with

```bash
source .venv/bin/activate
```

or use `uv run`. See the [uv docs](https://docs.astral.sh/uv/getting-started/) for more details.

### Pre-commit hooks

If you are planning to contribute, you should also install our [pre-commit](https://pre-commit.com/) hooks.

```bash
pre-commit install
```

## Codebase structure

The codebase follows a [flat organization](https://www.evandemond.com/programming/wide-and-flat) for easy forking and hacking. It was originally forked from [MAE-st](https://github.com/facebookresearch/mae_st).

**Training scripts**

- [`main_pretrain.py`](src/main_pretrain.py): Main foundation model pretraining script.
- [`main_classification.py`](src/main_classification.py): Script for downstream classification evaluations.
- [`main_regression.py`](src/main_regression.py): Script for downstream regression evaluations.

**Models**

- [`models_mae.py`](src/models_mae.py): Spatiotemporal masked autoencoder (MAE-st) for fMRI data.
- [`models_vit.py`](src/models_vit.py): Vanilla vision transformer (ViT) for fMRI data.
- [`models_mae_linear.py`](src/models_mae_linear.py): Linear masked autoencoder baseline.

**Datasets**

- [`flat_data.py`](src/flat_data.py): Dataset code for fMRI cortical flat map datasets.

## Contributing

This is a community-driven open science project. We welcome all contributions. To get started contributing, see our [contributing guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md). Then take a look at our [open issues](https://github.com/SophontAI/fmri-fm/issues/).

## License

The code is released under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
