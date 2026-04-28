from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def export_engine(
    weights_path: str,
    *,
    precision: str = "fp16",
    data: str | None = None,
    batch: int = 1,
    fraction: float = 1.0,
    dynamic: bool = False,
) -> Dict[str, Any]:
    from ultralytics import RTDETR

    precision = precision.lower()
    if precision not in {"fp32", "fp16", "int8"}:
        raise ValueError(f"Unsupported TensorRT precision: {precision}")
    if precision == "int8" and not data:
        raise ValueError("INT8 TensorRT export requires --data for calibration")

    m = RTDETR(weights_path)
    kwargs: Dict[str, Any] = {
        "format": "engine",
        "half": precision == "fp16",
        "int8": precision == "int8",
        "batch": batch,
        "dynamic": dynamic,
    }
    if data:
        kwargs["data"] = data
    if precision == "int8":
        kwargs["fraction"] = fraction
    m.export(**kwargs)
    wdir = Path(weights_path).parent
    candidates = list(wdir.rglob("best.engine")) + list(wdir.rglob("*.engine"))
    onnx_candidates = list(wdir.rglob("best.onnx")) + list(wdir.rglob("*.onnx"))
    return {
        "engine": str(candidates[0]) if candidates else None,
        "onnx": str(onnx_candidates[0]) if onnx_candidates else None,
        "pt": weights_path if weights_path.endswith(".pt") else None,
        "precision": precision,
        "data": data,
        "batch": batch,
        "fraction": fraction if precision == "int8" else None,
        "dynamic": dynamic,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Export RT-DETR weights to TensorRT engine")
    ap.add_argument("--weights", required=True, help="Path to .pt or .onnx weights")
    ap.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    ap.add_argument("--data", default=None, help="data.yaml path used for INT8 calibration")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--dynamic", action="store_true")
    ap.add_argument("--result", required=True, help="Path to write JSON result")
    args = ap.parse_args()

    Path(args.result).parent.mkdir(parents=True, exist_ok=True)

    res = export_engine(
        args.weights,
        precision=args.precision,
        data=args.data,
        batch=args.batch,
        fraction=args.fraction,
        dynamic=args.dynamic,
    )
    with open(args.result, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
