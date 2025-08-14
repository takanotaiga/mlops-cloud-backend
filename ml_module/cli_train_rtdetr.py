from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from ml_module.rtdetr_trainer import train_rtdetr


def main() -> int:
    ap = argparse.ArgumentParser(description="Train RT-DETR and optionally export TensorRT")
    ap.add_argument("--dataset", required=True, help="Path to YOLO-format dataset root (contains data.yaml)")
    ap.add_argument("--out-dir", required=True, help="Output directory for training artifacts")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--base-model", default="rtdetr-l.pt")
    ap.add_argument("--export-engine", action="store_true")
    ap.add_argument("--no-export-engine", dest="export_engine", action="store_false")
    ap.add_argument("--export-int8", action="store_true")
    ap.add_argument("--no-export-int8", dest="export_int8", action="store_false")
    ap.add_argument("--result", required=True, help="Path to write JSON result")
    ap.set_defaults(export_engine=True, export_int8=True)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result).parent.mkdir(parents=True, exist_ok=True)

    res: Dict[str, Any] = train_rtdetr(
        dataset_dir=args.dataset,
        out_dir=args.out_dir,
        epochs=args.epochs,
        imgsz=args.imgsz,
        base_model=args.base_model,
        export_engine=args.export_engine,
        export_int8=args.export_int8,
    )

    with open(args.result, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

