from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import pandas as pd


def run_inference(model_path: str, video_path: str, *, out_parquet: str, out_video: str | None = None,
                  conf: float = 0.25, imgsz: int = 640) -> Dict[str, Any]:
    from ultralytics import RTDETR

    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = None
    if out_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video, fourcc, fps or 30.0, (width, height))

    model = RTDETR(model_path)

    rows: List[Dict[str, Any]] = []
    frame_idx = -1
    for results in model.predict(video_path, stream=True, verbose=False, conf=conf, imgsz=imgsz):
        frame_idx += 1
        names = results.names if hasattr(results, "names") else {}
        # boxes.xywh, boxes.cls
        try:
            boxes = results.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    b = boxes.xywh[i]
                    cls = boxes.cls[i]
                    conf_i = boxes.conf[i] if hasattr(boxes, "conf") else None
                    x, y, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                    cls_id = int(cls.item()) if hasattr(cls, "item") else int(cls)
                    label = str(names.get(cls_id, cls_id))
                    rows.append({
                        "frame_index": int(frame_idx),
                        "label": label,
                        "x": int(x - w / 2.0),
                        "y": int(y - h / 2.0),
                        "w": int(w),
                        "h": int(h),
                        "conf": float(conf_i.item()) if conf_i is not None else None,
                    })
        except Exception:
            pass

        if writer is not None:
            try:
                img = results.plot()
                writer.write(img)
            except Exception:
                pass

    if writer is not None:
        writer.release()

    # Save parquet
    df = pd.DataFrame(rows, columns=["frame_index", "label", "x", "y", "w", "h", "conf"])
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)

    return {
        "parquet": str(out_parquet),
        "video": str(out_video) if out_video else None,
        "fps": fps,
        "width": width,
        "height": height,
        "frames": (frame_idx + 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run RT-DETR inference on a video and write parquet and optional overlay video")
    ap.add_argument("--model", required=True, help="Path to .engine or .pt weights")
    ap.add_argument("--video", required=True, help="Input video path")
    ap.add_argument("--out-parquet", required=True, help="Output parquet path")
    ap.add_argument("--out-video", default=None, help="Optional output overlay mp4 path")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--result", required=True, help="Path to write JSON result")
    args = ap.parse_args()

    res = run_inference(args.model, args.video, out_parquet=args.out_parquet, out_video=args.out_video,
                        conf=args.conf, imgsz=args.imgsz)
    Path(args.result).parent.mkdir(parents=True, exist_ok=True)
    with open(args.result, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

