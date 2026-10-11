#!/usr/bin/env python3
r"""Evaluate a released RF-DETR COCO detector on COCO val2017 and score its detections.

Predict resizes each image to the variant's native square resolution with an antialiased bilinear resize, the
preprocessing of Roboflow's own benchmark (roboflow/single_artifact_benchmarking) behind the published COCO numbers,
because `RFDETR.predict` resizes without antialiasing (roboflow/rf-detr#1516) and scores RF-DETR-N 48.0 instead of
48.4. It then runs `RFDETR.predict` in FP32 with threshold 0 and writes a COCO results json: the postprocessor's
top-300 detections per image, xywh boxes in original image pixels and the pretrained head's class ids, which are
already 91-space COCO category ids. Score evaluates that json with faster-coco-eval (bbox, default maxDets) against
every given annotation file, which must share the image ids and categories of the first one, and prints one SCORE line
with AP, AP50, APs, APm and APl as percentages.

Usage:
    CUDA_VISIBLE_DEVICES=0 python ul37/coco_eval.py nano rf-detr-nano.pth --images coco/val2017 --name rfdetr-n \
        --output out --ann coco=coco/annotations/instances_val2017.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

VARIANTS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
    "xlarge": "RFDETRXLarge",
    "2xlarge": "RFDETR2XLarge",
}


def predict(variant: str, weights: Path, images: Path, ann_file: Path, output: Path, batch: int) -> None:
    """Write the COCO results json of an RF-DETR checkpoint on every image of one annotation file.

    Args:
        variant (str): Key of VARIANTS.
        weights (Path): Released COCO checkpoint of that variant.
        images (Path): Directory holding the annotated images.
        ann_file (Path): COCO annotation json whose images are evaluated.
        output (Path): Results json to write.
        batch (int): Images per predict call.
    """
    import rfdetr
    import torchvision.transforms.functional as F
    from PIL import Image

    model = getattr(rfdetr, VARIANTS[variant])(pretrain_weights=str(weights))
    assert model.model_config.num_select == 300, model.model_config.num_select
    size = model.model.resolution
    files = [(image["id"], images / image["file_name"]) for image in json.loads(ann_file.read_text())["images"]]
    records = []
    for start in range(0, len(files), batch):
        chunk = files[start : start + batch]
        tensors = [F.to_tensor(Image.open(path).convert("RGB")) for _, path in chunk]
        resized = [F.resize(tensor, [size, size], antialias=True).clamp(0, 1) for tensor in tensors]
        detections = model.predict(resized, threshold=0.0, include_source_image=False)
        for (image_id, _), tensor, result in zip(chunk, tensors, detections):
            sx, sy = tensor.shape[2] / size, tensor.shape[1] / size
            for (x1, y1, x2, y2), score, class_id in zip(result.xyxy.tolist(), result.confidence, result.class_id):
                records.append(
                    {
                        "image_id": image_id,
                        "category_id": int(class_id),
                        "bbox": [round(x1 * sx, 3), round(y1 * sy, 3), round((x2 - x1) * sx, 3), round((y2 - y1) * sy, 3)],
                        "score": round(float(score), 5),
                    }
                )
    output.write_text(json.dumps(records))


def score(results: Path, anns: dict[str, Path]) -> dict[str, list[float]]:
    """Return AP, AP50, APs, APm and APl in percent of a COCO results json against each annotation file.

    Args:
        results (Path): COCO results json.
        anns (dict[str, Path]): Annotation json by name.

    Returns:
        (dict[str, list[float]]): The five metrics by annotation name.
    """
    from faster_coco_eval import COCO, COCOeval_faster

    detections, scores, reference = json.loads(results.read_text()), {}, None
    for name, path in anns.items():
        gt = COCO(str(path))
        ids = (sorted(gt.getImgIds()), gt.getCatIds())
        assert reference in (None, ids), f"{path} differs from the first annotation file in image ids or categories"
        reference = ids
        evaluator = COCOeval_faster(gt, gt.loadRes(detections), iouType="bbox", print_function=lambda *a: None)
        evaluator.params.imgIds = ids[0]
        evaluator.evaluate(), evaluator.accumulate(), evaluator.summarize()
        stats = evaluator.stats_as_dict
        scores[name] = [round(stats[k] * 100, 2) for k in ("AP_all", "AP_50", "AP_small", "AP_medium", "AP_large")]
    return scores


def main() -> None:
    """Predict COCO val detections with one RF-DETR checkpoint and score them."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("variant", choices=tuple(VARIANTS))
    parser.add_argument("weights", type=Path, help="released COCO checkpoint of the variant")
    parser.add_argument("--images", type=Path, required=True, help="COCO val2017 image directory")
    parser.add_argument("--ann", nargs="+", required=True, help="NAME=PATH annotation jsons, the first drives predict")
    parser.add_argument("--output", type=Path, required=True, help="directory for the results json")
    parser.add_argument("--name", required=True, help="model name for the results file and SCORE line")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    anns = {name: Path(path) for name, path in (item.split("=", 1) for item in args.ann)}
    args.output.mkdir(parents=True, exist_ok=True)
    results = args.output / f"{args.name}.json"
    predict(args.variant, args.weights, args.images, next(iter(anns.values())), results, args.batch)
    sha = hashlib.sha256(args.weights.read_bytes()).hexdigest()[:8]
    print("SCORE", json.dumps({"model": args.name, "sha256": sha, **score(results, anns)}), flush=True)


if __name__ == "__main__":
    main()
