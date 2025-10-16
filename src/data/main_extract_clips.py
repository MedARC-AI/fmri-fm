# Copyright (c) Sophont, Inc
#
# This source code is licensed under the CC-BY-NC license
# found in the LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from timm.utils import random_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.flat_data import make_flat_wds_dataset

DEFAULT_CONFIG = Path(__file__).parents[1] / "config/default_extract_clips.yaml"


def main(args: DictConfig):
    output_dir = Path(args.output_dir)

    include_datasets: list[str] = args.include_datasets or list(args.datasets)

    for dataset_name in include_datasets:
        dataset_config = args.datasets[dataset_name]
        print(f"dataset: {dataset_name}\n\n{OmegaConf.to_yaml(dataset_config)}")

        dataset_dir = output_dir / dataset_name
        if dataset_dir.exists():
            print(f"dataset output dir {dataset_dir} exists; skipping.")
            continue

        random_seed(args.seed)
        dataset = make_flat_wds_dataset(**dataset_config, shuffle=False)

        loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=cfg.num_workers)
        dataset_dir.mkdir(exist_ok=True, parents=True)
        OmegaConf.save(args, dataset_dir / "config.yaml")

        for ii, sample in enumerate(tqdm(loader)):
            save_sample(ii, sample, dataset_dir)


def save_sample(sample_id: int, sample: dict[str, Any], dataset_dir: Path):
    sample = {k: cast_value(v) for k, v in sample.items()}
    path = dataset_dir / f"{sample_id:07d}.pt"
    torch.save(sample, path)


def cast_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif isinstance(value, np.generic):
        value = value.item()
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default=None)
    parser.add_argument("--overrides", type=str, default=None, nargs="+")
    args = parser.parse_args()
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    if args.cfg_path:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(args.cfg_path))
    if args.overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(args.overrides))
    main(cfg)
