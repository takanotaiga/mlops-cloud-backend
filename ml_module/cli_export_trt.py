from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict


def export_engine(weights_path: str, *, data_yaml: str | None = None, int8: bool = True) -> Dict[str, Any]:
    from ultralytics import RTDETR
    m = RTDETR(weights_path)
    # The export will place artifacts next to weights under weights/..../weights
    m.export(format="engine", int8=bool(int8), data=data_yaml)
    # Try to discover produced files
    wdir = Path(weights_path).parent
    candidates = list(wdir.rglob("best.engine")) + list(wdir.rglob("*.engine"))
    onnx_candidates = list(wdir.rglob("best.onnx")) + list(wdir.rglob("*.onnx"))
    return {
        "engine": str(candidates[0]) if candidates else None,
        "onnx": str(onnx_candidates[0]) if onnx_candidates else None,
        "pt": weights_path if weights_path.endswith(".pt") else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Export RT-DETR weights to TensorRT engine")
    ap.add_argument("--weights", required=True, help="Path to .pt or .onnx weights")
    ap.add_argument("--data-yaml", default=None, help="Optional data.yaml for class names")
    ap.add_argument("--int8", action="store_true", help="Export INT8 engine when possible")
    ap.add_argument("--result", required=True, help="Path to write JSON result")
    args = ap.parse_args()

    Path(args.result).parent.mkdir(parents=True, exist_ok=True)

    res = export_engine(args.weights, data_yaml=args.data_yaml, int8=args.int8)
    with open(args.result, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

