from __future__ import annotations

import argparse
import json
import os
import glob
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ultralytics import RTDETR

PathLike = Union[str, Path]


def train_rtdetr(
    dataset_dir: PathLike,
    out_dir: PathLike,
    *,
    epochs: int = 1,
    base_model: str = "rtdetr-l.pt",
) -> Dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"'data.yaml' is not found: {data_yaml}")

    model_ref = base_model[:-3] + ".yaml" if base_model.endswith(".pt") else base_model
    model = RTDETR(str(model_ref))

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        name="train_result",
        project=str(out_dir),
        exist_ok=True,
        pretrained=False,
        batch=-1,  # auto-batch
    )

    candidates = glob.glob(str(out_dir / "train_result*" / "weights" / "best.pt"))
    best_pt: Optional[str] = None
    if candidates:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        best_pt = candidates[0]

    return {"pt": best_pt}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train RT-DETR (Ultralytics)")
    ap.add_argument("--dataset", required=True, help="YOLO形式データセットのルート（data.yaml を含む）")
    ap.add_argument("--out-dir", required=True, help="学習成果物の出力ディレクトリ")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--base-model", default="rtdetr-l.pt")
    ap.add_argument("--result", required=True, help="結果JSONの出力パス")
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    res: Dict[str, Any] = train_rtdetr(
        dataset_dir=args.dataset,
        out_dir=args.out_dir,
        epochs=args.epochs,
        base_model=args.base_model,
    )

    result_path = Path(args.result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
