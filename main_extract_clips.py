import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf
from timm.utils import random_seed
from tqdm import tqdm

from flat_data import make_flat_wds_dataset

DEFAULT_CONFIG = Path(__file__).parent / "config/default_extract_clips.yaml"


def main(args: DictConfig):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True)
    OmegaConf.save(args, output_dir / "config.yaml")

    for dataset_name, dataset_config in args.datasets.items():
        print(f"dataset: {dataset_name}\n\n{OmegaConf.to_yaml(dataset_config)}")

        random_seed(args.seed)
        dataset = make_flat_wds_dataset(**dataset_config, shuffle=False)

        dataset_dir = Path(output_dir / dataset_name)
        dataset_dir.mkdir()

        for ii, sample in enumerate(tqdm(dataset)):
            save_sample(ii, sample, dataset_dir)


def save_sample(sample_id: int, sample: dict[str, Any], dataset_dir: Path):
    key = sample["__key__"]
    start = sample["start"]
    image = sample["image"]
    path = dataset_dir / f"{sample_id:07d}_{key}_start-{start:05d}.npy"
    np.save(path, image)


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
