r"""Export RF-DETR detection sizes to static batch-1 FP32 ONNX at native resolution with RF-DETR's own exporter.

Usage: python export_rfdetr.py OUT_DIR [nano small medium large xlarge xxlarge]
Each size is merged into OUT_DIR/export_info.json.
Run on CPU from the rf-detr checkout with its .venv plus the ONNX packages it lacks, in a scratch dir D:
    uv pip install --python .venv/bin/python --target D --no-deps flatbuffers==25.12.19 ml-dtypes==0.6.0 \
        onnx==1.23.2 onnx-graphsurgeon==0.6.1 onnxruntime==1.30.0 onnxsim==0.7.3
    CUDA_VISIBLE_DEVICES= PYTHONPATH=D .venv/bin/python ul37/export_rfdetr.py OUT_DIR SIZE
"""

import json
import sys
from pathlib import Path

import rfdetr
import torch

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
    info[size] = {
        "class": cls,
        "imgsz": res,
        "params": params,
        "onnx": str(path),
        "weights": str(model.model_config.pretrain_weights),
    }
    print("EXPORTED", json.dumps(info[size]), flush=True)
    del model
out_json.write_text(json.dumps(info, indent=2))
print("rfdetr", rfdetr.__file__, "torch", torch.__version__)
