r"""Export RF-DETR detection sizes to static batch-1 ONNX at native resolution with RF-DETR's own exporter.

Usage: python export_rfdetr.py OUT_DIR [nano small medium large xlarge xxlarge]
Each size writes the FP32 graph rf-detr-SIZE.onnx and rf-detr-SIZE.fp16.onnx, the FP16 graph RF-DETR's TensorRT 11
export builds its engine from (`_cast_onnx_to_fp16`: a full onnxconverter-common cast with FP32 IO), and is merged into
OUT_DIR/export_info.json.
Run on CPU from the rf-detr checkout with its .venv plus the ONNX packages it lacks, in a scratch dir D:
    uv pip install --python .venv/bin/python --target D --no-deps ml-dtypes==0.6.0 onnx==1.23.2 \
        onnxconverter-common==1.16.0 protobuf==6.33.6
    CUDA_VISIBLE_DEVICES= PYTHONPATH=D .venv/bin/python ul37/export_rfdetr.py OUT_DIR SIZE
"""

import json
import sys
from pathlib import Path

import rfdetr
import torch
from rfdetr.export._tensorrt.exporter import _cast_onnx_to_fp16

SIZES = {
    "nano": ("RFDETRNano", 384),
    "small": ("RFDETRSmall", 512),
    "medium": ("RFDETRMedium", 576),
    "large": ("RFDETRLarge", 704),
    "xlarge": ("RFDETRXLarge", 700),
    "xxlarge": ("RFDETR2XLarge", 880),
}

out = Path(sys.argv[1])
out_json = out / "export_info.json"
info = json.loads(out_json.read_text()) if out_json.exists() else {}
for size in sys.argv[2:] or SIZES:
    cls, res = SIZES[size]
    model = getattr(rfdetr, cls)(device="cpu")
    assert model.model.resolution == res, (size, model.model.resolution)
    params = sum(p.numel() for p in model.model.model.parameters())
    path = model.export(
        output_dir=str(out), output_name=f"rf-detr-{size}", shape=(res, res), batch_size=1, verbose=False
    )
    fp16 = Path(_cast_onnx_to_fp16(str(path), dynamic_batch=False)).rename(out / f"rf-detr-{size}.fp16.onnx")
    info[size] = {
        "class": cls,
        "imgsz": res,
        "params": params,
        "onnx": str(path),
        "onnx_fp16": str(fp16),
        "weights": str(model.model_config.pretrain_weights),
    }
    print("EXPORTED", json.dumps(info[size]), flush=True)
    del model
out_json.write_text(json.dumps(info, indent=2))
print("rfdetr", rfdetr.__file__, "torch", torch.__version__)
