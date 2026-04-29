from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import pandas as pd


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


def _draw_detection(frame, x1: int, y1: int, x2: int, y2: int, label: str, conf: Optional[float]) -> None:
    hv = abs(hash(label)) % 255
    color = (int((50 + 2 * hv) % 255), int((120 + hv) % 255), int((200 + 3 * hv) % 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.2f}" if conf is not None else label
    cv2.putText(frame, text, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, lineType=cv2.LINE_AA)


def run_inference(
    model_path: str,
    video_path: str,
    *,
    out_parquet: str,
    out_video: Optional[str],
    variant: str = "medium",
    conf: float = 0.25,
) -> Dict[str, Any]:
    model_cls = _model_class(variant)
    model = model_cls(pretrain_weights=model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if out_video:
        Path(out_video).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video, fourcc, fps or 30.0, (width, height))

    rows: List[Dict[str, Any]] = []
    frame_idx = -1
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_idx += 1
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            detections = model.predict(frame_rgb, threshold=conf)

            xyxy = getattr(detections, "xyxy", [])
            class_ids = getattr(detections, "class_id", [])
            confidences = getattr(detections, "confidence", [])
            for i, box in enumerate(xyxy):
                x1, y1, x2, y2 = [int(v) for v in box]
                class_id = int(class_ids[i]) if i < len(class_ids) and class_ids[i] is not None else -1
                score = float(confidences[i]) if i < len(confidences) and confidences[i] is not None else None
                label = str(class_id if class_id >= 0 else "object")
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                if w <= 0 or h <= 0:
                    continue
                rows.append(
                    {
                        "frame_index": frame_idx,
                        "label": label,
                        "x": x1,
                        "y": y1,
                        "w": w,
                        "h": h,
                        "conf": score,
                        "class_id": class_id,
                    }
                )
                if writer is not None:
                    _draw_detection(frame_bgr, x1, y1, x2, y2, label, score)

            if writer is not None:
                writer.write(frame_bgr)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    df = pd.DataFrame(rows, columns=["frame_index", "label", "x", "y", "w", "h", "conf", "class_id"])
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)

    return {
        "parquet": out_parquet,
        "video": out_video,
        "variant": variant,
        "fps": fps,
        "width": width,
        "height": height,
        "frames": frame_idx + 1,
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run RF-DETR inference on a video")
    ap.add_argument("--model", required=True, help="Path to RF-DETR checkpoint .pth")
    ap.add_argument("--video", required=True, help="Input video path")
    ap.add_argument("--out-parquet", required=True, help="Output parquet path")
    ap.add_argument("--out-video", default=None, help="Optional output overlay mp4")
    ap.add_argument("--variant", default="medium", choices=["nano", "small", "medium", "large"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--half", action="store_true", help="Accepted for compatibility; RF-DETR chooses precision internally")
    ap.add_argument("--result", required=True, help="Path to write JSON result")
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    result = run_inference(
        args.model,
        args.video,
        out_parquet=args.out_parquet,
        out_video=args.out_video,
        variant=args.variant,
        conf=args.conf,
    )
    result_path = Path(args.result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
