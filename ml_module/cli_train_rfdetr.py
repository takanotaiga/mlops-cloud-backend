from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def _model_class(variant: str):
    variant = variant.lower().replace("-", "_")
    if variant in {"nano", "n"}:
        from rfdetr import RFDETRNano

        return RFDETRNano
    if variant in {"small", "s"}:
        from rfdetr import RFDETRSmall

        return RFDETRSmall
    if variant in {"medium", "m"}:
        from rfdetr import RFDETRMedium

        return RFDETRMedium
    if variant in {"large", "l"}:
        from rfdetr import RFDETRLarge

        return RFDETRLarge
    raise ValueError(f"Unsupported RF-DETR variant: {variant}")


def train_rfdetr(
    dataset_dir: str | Path,
    out_dir: str | Path,
    *,
    epochs: int,
    variant: str = "medium",
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    lr: float = 1e-4,
) -> Dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"'data.yaml' is not found: {data_yaml}")

    model_cls = _model_class(variant)
    model = model_cls()
    model.train(
        dataset_dir=str(dataset_dir),
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=lr,
        output_dir=str(out_dir),
    )

    patterns = [
        "**/checkpoint_best_total.pth",
        "**/checkpoint_best_ema.pth",
        "**/checkpoint_best_regular.pth",
        "**/checkpoint.pth",
    ]
    checkpoint: Optional[str] = None
    for pattern in patterns:
        candidates = glob.glob(str(out_dir / pattern), recursive=True)
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            checkpoint = candidates[0]
            break

    return {
        "checkpoint": checkpoint,
        "variant": variant,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "lr": lr,
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train RF-DETR on a YOLO/COCO dataset")
    ap.add_argument("--dataset", required=True, help="Dataset root containing data.yaml")
    ap.add_argument("--out-dir", required=True, help="Training output directory")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--variant", default="medium", choices=["nano", "small", "medium", "large"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum-steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--result", required=True, help="Path to write JSON result")
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    result = train_rfdetr(
        dataset_dir=args.dataset,
        out_dir=args.out_dir,
        epochs=args.epochs,
        variant=args.variant,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
    )
    result_path = Path(args.result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
