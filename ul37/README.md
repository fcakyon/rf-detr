# UL37 benchmark runner

`run_ul37.py` fine-tunes RF-DETR on each of the 37 UL37 detection datasets and saves the predictions the benchmark
scores. Each variant keeps RF-DETR's native resolution, queries, `num_select` and evaluator, reads the benchmark's
pinned YOLO dataset directly and trains 100 epochs at batch 16 (seed 0, no multi-scale, no early stopping) from the
released COCO checkpoint, which RF-DETR downloads into `~/.roboflow/models`. It then writes `predictions.json` (best
total checkpoint, threshold 0), `checkpoint_best_total.pth` and `args.yaml`. A dataset with `args.yaml` is skipped
only when its saved provenance, this repo's commit included, matches the current run.

## Setup

```bash
git clone --single-branch -b ul37 git@github.com:fcakyon/rf-detr.git && cd rf-detr && git checkout SHA
uv venv -p 3.13 .venv && uv pip install --python .venv/bin/python -r ul37/requirements-ul37-lock.txt -e '.[train]'
.venv/bin/python ul37/check_ul37.py --benchmark BENCH [--datasets]
```

`check_ul37.py` compares every installed package with the lock, checks each variant's resolution and 300 queries, and
with `--datasets` checks every UL37 dataset with RF-DETR's YOLO loader.

## Run

`BENCH` is a UL37 benchmark checkout: runs read its `ul37_platform_uris.txt`, `ul37_datasets.txt` and `.venv-ul37`
runtime with `.config-ul37-081026`, and scoring runs its `val_ultrabench.py`. `VARIANT` is `nano`,
`small`, `medium`, `large`, `x` or `xx`.

```bash
.venv/bin/python ul37/run_ul37.py VARIANT --project RUNS --benchmark BENCH --device 0
```

Results go to `RUNS/ultrabench-081026-rfdetr-<n|s|m|l|x|xx>/multitrain`. An unsharded run scores itself into
`val_ultrabench.json`. For shards, give each process `--subset START:STOP` (or comma-separated dataset names) and the
same `--name`, then score once with `--val-only` after all shards finish.

## Latency export

`export_rfdetr.py` exports each variant to FP32 ONNX with RF-DETR's own exporter. Its docstring lists the ONNX
packages.

## COCO-SAMA labels

COCO-SAMA re-annotates COCO val2017 images. Build its `instances_val2017.json` from Sama's release:

```bash
curl -LO https://sama-documentation-assets.s3.amazonaws.com/sama-coco/sama-coco-val.zip
echo "c55454f6657ce63891783136cc5dfe5f1e9c5e1748c0d8ca08b456a804eab15f  sama-coco-val.zip" | sha256sum -c
unzip -j sama-coco-val.zip '*sama_coco_coco_format_val_*.json'
python3 -c "import json; s = [json.load(open(f'sama_coco_coco_format_val_{i}.json')) for i in range(5)]; open('instances_val2017.json', 'w').write(json.dumps({k: s[0][k] for k in ('info', 'licenses', 'categories')} | {'images': sorted((x for d in s for x in d['images']), key=lambda x: x['id']), 'annotations': [x for d in s for x in d['annotations']]}))"
sha256sum instances_val2017.json # c8575a4b8eb39a8f7316c3e6f51349834ca0dc3b11e35f615497064c5d0b79b0
```
