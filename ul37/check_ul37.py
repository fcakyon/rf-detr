import argparse
import importlib.metadata
import json
import re
import sys
from pathlib import Path

import rfdetr
import run_ul37 as runner
from rfdetr.config import TrainConfig
from rfdetr.datasets.yolo import is_valid_yolo_dataset


def canonical(name: str) -> str:
    """Normalize a distribution name like pip does."""
    return re.sub(r"[-_.]+", "-", name).lower()


parser = argparse.ArgumentParser()
parser.add_argument("--benchmark", type=Path, required=True, help="UL37 benchmark checkout with val_ultrabench.py")
parser.add_argument("--datasets", action="store_true", help="Also check every UL37 dataset with RF-DETR's loader")
args = parser.parse_args()

lock = Path(__file__).with_name("requirements-ul37-lock.txt").read_text().split()
locked = {canonical(name): version for name, version in (line.split("==") for line in lock)}
installed = {canonical(dist.metadata["Name"]): dist.version for dist in importlib.metadata.distributions()}
installed.pop("rfdetr")
assert installed == locked, {
    name: (installed.get(name), locked.get(name))
    for name in installed.keys() | locked.keys()
    if installed.get(name) != locked.get(name)
}
assert sys.version_info[:2] == (3, 13)
for name, resolution in runner.VARIANTS.values():
    config = getattr(rfdetr, name)().model_config
    assert (config.resolution, config.num_queries, config.num_select) == (resolution, 300, 300), (name, config)
TrainConfig(dataset_dir="/tmp", epochs=100, checkpoint_interval=101, **runner.TRAIN)

fraction, roster, data_yamls, ultralytics_sha = runner.load_roster(args.benchmark)
print(json.dumps({"fraction": fraction, "datasets": len(roster), "ultralytics_sha": ultralytics_sha}))

if args.datasets:
    counts = {}
    for name, data_yaml in data_yamls.items():
        assert is_valid_yolo_dataset(str(data_yaml.parent)), data_yaml
        counts[name] = {split: len(runner.split_images(data_yaml.parent, split)) for split in ("train", "val")}
    print(json.dumps(counts, sort_keys=True))
