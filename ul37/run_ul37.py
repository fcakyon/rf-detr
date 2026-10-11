#!/usr/bin/env python3
"""Train RF-DETR on Platform-backed UL37 datasets and save canonical prediction artifacts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ul37_common import (
    PROTOCOL,
    load_roster,
    lock_dataset,
    pending_datasets,
    run_benchmark,
    sha256,
    write_args,
)

RUNNER_SHA256 = sha256(Path(__file__))  # at import: the code this process runs, even if redeployed under it
VARIANTS = {  # RF-DETR class and native resolution, word variants slug to their first letter (nano -> n)
    "nano": ("RFDETRNano", 384),
    "small": ("RFDETRSmall", 512),
    "medium": ("RFDETRMedium", 576),
    "large": ("RFDETRLarge", 704),
    "x": ("RFDETRXLarge", 700),
    "xx": ("RFDETR2XLarge", 880),
}
TRAIN = {
    "batch_size": 16,
    "grad_accum_steps": 1,
    "seed": 0,
    "eval_interval": 10,
    "multi_scale": False,
    "expanded_scales": False,
    "early_stopping": False,
    "num_workers": 4,
}


def split_images(root: Path, split: str) -> list[Path]:
    """List one split's images exactly as RF-DETR's YOLO loader resolves them."""
    from rfdetr.datasets.yolo import _list_yolo_image_paths, _resolve_yolo_split_dirs, find_yolo_data_file

    images_dir, _ = _resolve_yolo_split_dirs(root, find_yolo_data_file(root), split)
    return [Path(path) for path in _list_yolo_image_paths(str(images_dir))]


def save_predictions(model, images: list[Path], output: Path, batch_size: int) -> None:
    """Save RF-DETR detections of dataset classes in the canonical Ultra Benchmark JSON shape."""
    num_classes = model.model_config.num_classes
    records = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        detections = model.predict([str(path) for path in batch], threshold=0.0, include_source_image=False)
        for path, result in zip(batch, detections):
            for (x1, y1, x2, y2), score, class_id in zip(result.xyxy.tolist(), result.confidence, result.class_id):
                if class_id >= num_classes:  # extra no-object head slot, never a dataset class
                    continue
                records.append(
                    {
                        "file_name": path.name,
                        "category_id": int(class_id) + 1,
                        "bbox": [round(x1, 3), round(y1, 3), round(x2 - x1, 3), round(y2 - y1, 3)],
                        "score": round(float(score), 5),
                    }
                )
    output.write_text(json.dumps(records))


def train_run(args: argparse.Namespace, run_dir: Path) -> None:
    """Train every selected dataset without args.yaml, then predict from its best checkpoint."""
    fraction, datasets, data_yamls, ultralytics_sha = load_roster(args.benchmark)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device.removeprefix("cuda:")
    import rfdetr
    import torch
    from rfdetr.config import TrainConfig
    from rfdetr.datasets.yolo import _extract_yolo_class_names

    class_name, resolution = VARIANTS[args.variant]
    model_class = getattr(rfdetr, class_name)
    clone = Path(rfdetr.__file__).parent
    common = {
        "variant": args.variant,
        "rfdetr_sha": subprocess.check_output(["git", "-C", str(clone), "rev-parse", "HEAD"], text=True).strip(),
        "rfdetr_patch_sha256": hashlib.sha256(
            subprocess.check_output(["git", "-C", str(clone), "diff", "HEAD"])
        ).hexdigest(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "ultralytics_sha": ultralytics_sha,
        "roster_sha256": sha256(args.benchmark / "ul37_platform_uris.txt"),
        "fraction": fraction,
        "imgsz": resolution,
        "epochs": args.epochs,
        **TRAIN,
    }
    pending = pending_datasets(run_dir, datasets, args.subset, common)

    for name, expected in pending.items():
        output_dir, scratch = run_dir / name, Path.home() / "rfdetr-scratch" / run_dir.parent.name / name
        lock = lock_dataset(run_dir, name)  # noqa: F841, held until the next dataset
        shutil.rmtree(output_dir, ignore_errors=True)  # no args.yaml, so anything here is from an unfinished attempt
        output_dir.mkdir(parents=True)
        model = model_class(num_classes=len(_extract_yolo_class_names(data_yamls[name])))  # XLarge pins 90 otherwise
        source, last = Path(model.model_config.pretrain_weights), scratch / "last.ckpt"
        print(f"{name}: {data_yamls[name].parent} resolution={model.model_config.resolution} (RF-DETR defaults)")
        model.train(
            dataset_dir=str(data_yamls[name].parent),
            dataset_file="yolo",
            output_dir=str(scratch),
            resume=str(last) if last.is_file() else None,
            epochs=args.epochs,
            checkpoint_interval=args.epochs + 1,  # only last.ckpt for resume, no periodic archive
            device="cuda",
            run_test=False,
            notes=expected,
            **TRAIN,
        )
        config = model.model_config
        best = scratch / "checkpoint_best_total.pth"
        model = model_class.from_checkpoint(best, trust_checkpoint=True)
        images = split_images(data_yamls[name].parent, "val")
        save_predictions(model, images, output_dir / "predictions.json", args.prediction_batch)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        for path in scratch.iterdir():  # keep the best checkpoint and logs, move into the setgid run dir
            if path == best or path.suffix not in {".ckpt", ".pth"}:
                shutil.move(path, output_dir / path.name)
        shutil.rmtree(scratch)
        (output_dir / best.name).chmod(0o664)
        provenance = {
            **expected,
            "runner_sha256": RUNNER_SHA256,
            "data": str(data_yamls[name]),
            "num_classes": config.num_classes,
            "max_det": config.num_select,
            "num_queries": config.num_queries,
            "eval_max_dets": TrainConfig.model_fields["eval_max_dets"].default,
            "source_model": str(source),
            "source_sha256": sha256(source),
            "val_images": len(images),
            "prediction_checkpoint": str(output_dir / best.name),
        }
        write_args(output_dir, provenance)


def main() -> None:
    """Train or score one RF-DETR UL37 run."""
    os.umask(0o002)
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=tuple(VARIANTS))
    parser.add_argument("--device", default="0")
    parser.add_argument("--subset", help="Dataset slice like 0:5 or comma-separated dataset names")
    parser.add_argument("--name")
    parser.add_argument("--project", type=Path, required=True, help="Output root, runs go to PROJECT/NAME/multitrain")
    parser.add_argument("--benchmark", type=Path, required=True, help="UL37 benchmark checkout with val_ultrabench.py")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--prediction-batch", type=int, default=1)
    parser.add_argument("--val-only", action="store_true")
    parser.add_argument("--native-size-bins", action="store_true")
    args = parser.parse_args()

    run_dir = (
        args.project
        / (args.name or f"ultrabench-{PROTOCOL}-rfdetr-{args.variant if len(args.variant) <= 2 else args.variant[0]}")
        / "multitrain"
    ).resolve()
    if not args.val_only:
        train_run(args, run_dir)
    if args.val_only or not args.subset:
        bins = ["--native-size-bins"] if args.native_size_bins else []
        run_benchmark(args.benchmark, "val_ultrabench.py", str(run_dir), *bins)


if __name__ == "__main__":
    main()
