import argparse
import glob
import tarfile
from io import BytesIO
from typing import Tuple

import numpy as np
import pandas as pd


def load_npy_from_tar(tf: tarfile.TarFile, member: tarfile.TarInfo) -> np.ndarray:
    """Load a .npy array from a tar member robustly."""
    f = tf.extractfile(member)
    assert f is not None, f"Failed to extract {member.name}"
    try:
        return np.load(f, allow_pickle=False)
    except Exception:
        # Fallback: read bytes then wrap in BytesIO
        try:
            f.seek(0)
        except Exception:
            pass
        data = f.read()
        return np.load(BytesIO(data), allow_pickle=False)
    finally:
        f.close()


def determine_orientation(arr: np.ndarray, roi_count: int) -> Tuple[str, int]:
    """Return (orientation, W_full). Orientation is 'roi_first', 'time_first', or 'invalid'."""
    if arr.ndim != 2:
        return ("invalid", -1)
    h, w = arr.shape
    if h == roi_count:
        return ("roi_first", w)
    if w == roi_count:
        return ("time_first", h)
    return ("invalid", -1)


def inspect_shards(tar_glob: str, max_shards: int, per_shard: int, roi_count: int) -> int:
    """Inspect .bold.npy entries across shards. Returns number of invalid files found."""
    shard_paths = sorted(glob.glob(tar_glob))[:max_shards]
    invalid = 0
    total = 0
    print(f"Inspecting up to {per_shard} .bold.npy files from each of {len(shard_paths)} shard(s)...\n")
    for shard in shard_paths:
        try:
            with tarfile.open(shard) as tf:
                printed = 0
                for member in tf.getmembers():
                    if not member.name.endswith(".bold.npy"):
                        continue
                    arr = load_npy_from_tar(tf, member)
                    orientation, w_full = determine_orientation(arr, roi_count)
                    total += 1
                    if orientation == "invalid":
                        invalid += 1
                    # Print a few samples per shard
                    if printed < per_shard:
                        print(
                            f"{shard} :: {member.name} :: shape={arr.shape} dtype={arr.dtype} "
                            f"-> orientation={orientation} W_full={w_full}"
                        )
                        printed += 1
        except Exception as e:
            print(f"ERROR opening shard {shard}: {e}")
    print(f"\nSummary: checked={total}, invalid_dim={invalid}, expected_roi_count={roi_count}")
    return invalid


def write_gradient_400(src_path: str, dst_path: str, roi_count: int) -> None:
    df = pd.read_csv(src_path, header=None)
    if len(df) < roi_count:
        raise ValueError(
            f"Gradient source rows={len(df)} < roi_count={roi_count}. Cannot slice."
        )
    df.iloc[:roi_count].to_csv(dst_path, index=False, header=False)
    print(
        f"Wrote {dst_path} from {src_path} :: src_shape={tuple(df.shape)} dst_shape={(roi_count, df.shape[1])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect HCP shards and prepare gradient CSV")
    parser.add_argument(
        "--tar_glob",
        type=str,
        default="/teamspace/filestore_folders/shared/fmri-fm/datasets/hcp-parc/hcp-parc_*.tar",
        help="Glob for HCP tar shards",
    )
    parser.add_argument("--max_shards", type=int, default=3, help="Max shards to inspect")
    parser.add_argument("--per_shard", type=int, default=3, help="Files to print per shard")
    parser.add_argument("--roi_count", type=int, default=400, help="Expected number of ROIs")
    parser.add_argument(
        "--write_gradient",
        action="store_true",
        help="If set, slice 450-row gradient to 400 and write to dst",
    )
    parser.add_argument(
        "--gradient_src",
        type=str,
        default="src/brain_jepa/gradient_mapping_450.csv",
        help="Path to 450-row gradient CSV",
    )
    parser.add_argument(
        "--gradient_dst",
        type=str,
        default="src/brain_jepa/gradient_mapping_400.csv",
        help="Output path for 400-row gradient CSV",
    )
    args = parser.parse_args()

    invalid = inspect_shards(
        tar_glob=args.tar_glob,
        max_shards=args.max_shards,
        per_shard=args.per_shard,
        roi_count=args.roi_count,
    )

    if args.write_gradient:
        write_gradient_400(args.gradient_src, args.gradient_dst, args.roi_count)

    # Exit non-zero if invalid files encountered
    if invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()



