# track_sam2_parquet.py
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from contextlib import nullcontext


def _natural_key(s: str):
    # 数字を数値扱いにして自然順ソート
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _list_image_files(d: str) -> List[Path]:
    p = Path(d)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts]
    files.sort(key=lambda x: _natural_key(x.name))
    if not files:
        raise FileNotFoundError(f"No images found under: {d}")
    return files


def _build_sam2_predictor():
    # sam2 の import 経路
    sam2_repo = os.environ.get("SAM2_REPO_DIR", "/workspace/samurai/sam2")
    if sam2_repo and sam2_repo not in sys.path:
        sys.path.append(sam2_repo)

    from sam2.build_sam import build_sam2_video_predictor  # type: ignore

    ckpt = os.environ.get(
        "SAM2_CHECKPOINT",
        "/workspace/samurai/sam2/checkpoints/sam2.1_hiera_base_plus.pt",
    )
    cfg = os.environ.get(
        "SAM2_CONFIG",
        "configs/samurai/sam2.1_hiera_b+.yaml",
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and torch.cuda.get_device_properties(0).major >= 8:
      torch.backends.cuda.matmul.allow_tf32 = True
      torch.backends.cudnn.allow_tf32 = True
      try:
          torch.set_float32_matmul_precision("high")
      except Exception:
          pass
    predictor = build_sam2_video_predictor(cfg, ckpt, device=device)
    return predictor, device


def sam2_track_to_parquet(
    images_dir: str,
    x0, x1, y0, y1,
    parquet_path: Path,
) -> Dict[str, object]:
    """
    1オブジェクト（key）の追跡結果をParquetに保存。
    戻り値はサマリ情報（JSON用）。
    """
    files = _list_image_files(images_dir)
    files = files[:1000]
    file_names = [f.name for f in files]
    frame_count = len(files)

    predictor, device = _build_sam2_predictor()

    bbox_prompt = (x0, y0, x1, y1)
    if device.startswith("cuda"):
        try:
            amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
        except Exception:
            amp_ctx = torch.autocast("cuda", dtype=torch.float16)
    else:
        amp_ctx = nullcontext()

    records: List[Dict[str, object]] = []

    with torch.inference_mode(), amp_ctx:
        state = predictor.init_state(
            images_dir,
            offload_video_to_cpu=False,
            offload_state_to_cpu=True,
            async_loading_frames=True,
        )
        predictor.add_new_points_or_box(state, box=bbox_prompt, frame_idx=0, obj_id=0)

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            bbox_out = (-1, -1, -1, -1)
            for obj_id, mask in zip(object_ids, masks):
                if obj_id != 0:
                    continue
                m = mask[0].detach().cpu().numpy()
                m = m > 0.0
                idx = np.argwhere(m)
                if idx.size > 0:
                    y_min, x_min = idx.min(axis=0).tolist()
                    y_max, x_max = idx.max(axis=0).tolist()
                    bbox_out = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
                break

            records.append(
                {
                    "frame_idx": int(frame_idx),
                    "x": int(bbox_out[0]),
                    "y": int(bbox_out[1]),
                    "w": int(bbox_out[2]),
                    "h": int(bbox_out[3]),
                    "image_file": file_names[frame_idx] if frame_idx < len(file_names) else None,
                }
            )

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_records(records).sort_values("frame_idx")
    df.to_parquet(parquet_path, index=False)

    return {
        "parquet": str(parquet_path),
        "frames": frame_count,
        "images_dir": str(Path(images_dir).resolve()),
        "columns": list(df.columns),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="SAM2 single-object tracking over image sequence -> Parquet")
    ap.add_argument("--images", required=True, help="連番画像ディレクトリ（jpg/png等）")
    ap.add_argument("--result", required=True, help="出力JSONのパス（Parquetの場所などのサマリを書き出す）")
    ap.add_argument("--x0", required=True, type=int, default=0, help="key bbox x0")
    ap.add_argument("--x1", required=True, type=int, default=0, help="key bbox x1")
    ap.add_argument("--y0", required=True, type=int, default=0, help="key bbox y0")
    ap.add_argument("--y1", required=True, type=int, default=0, help="key bbox y1")

    args = ap.parse_args()

    result_json = Path(args.result)
    result_json.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = result_json.with_suffix(".parquet")

    summary = sam2_track_to_parquet(args.images, args.x0,  args.x1,  args.y0,  args.y1, parquet_path)

    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
