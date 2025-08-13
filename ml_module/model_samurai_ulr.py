import os
import os.path as osp
import json
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Any

import cv2
import numpy as np
import torch
import pandas as pd
import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore

# SAM2 predictor
import sys
sys.path.append("/workspace/samurai/sam2")
from sam2.build_sam import build_sam2_video_predictor  # type: ignore


MODEL_PATH = "/workspace/samurai/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
MODEL_CONFIG = "configs/samurai/sam2.1_hiera_b+.yaml"


@dataclass
class SeedBox:
    label: str
    x1: float  # normalized [0,1]
    y1: float  # normalized [0,1]
    x2: float  # normalized [0,1]
    y2: float  # normalized [0,1]


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _get_total_frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        # Try property first; may be 0 for some codecs
        cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return max(0, cnt)
    finally:
        cap.release()


def _extract_frames(video_path: str, out_dir: str) -> Tuple[int, int, int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            # Hard safety: abort if frame count threshold exceeded during extraction
            if count >= 27000:
                raise ValueError("Video has 27000 or more frames; aborting per policy.")
            cv2.imwrite(osp.join(out_dir, f"{count:08d}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    finally:
        cap.release()
    return count, w, h, fps


def _overlay_masks_and_boxes(frame: np.ndarray, masks: Dict[int, np.ndarray], boxes: Dict[int, Tuple[int, int, int, int]], labels: Dict[int, str]) -> np.ndarray:
    # deterministic colors by obj id
    def _color(idx: int) -> Tuple[int, int, int]:
        rng = (37 * (idx + 1)) % 255
        return (int(50 + rng) % 255, int(120 + 2*rng) % 255, int(200 + 3*rng) % 255)

    img = frame
    for obj_id, mask in masks.items():
        if mask is None:
            continue
        mask_img = np.zeros_like(img, dtype=np.uint8)
        mask_img[mask.astype(bool)] = _color(obj_id)
        img = cv2.addWeighted(img, 1.0, mask_img, 0.35, 0)

    for obj_id, box in boxes.items():
        x, y, w, h = box
        c = _color(obj_id)
        cv2.rectangle(img, (x, y), (x + w, y + h), c, 2)
        text = labels.get(obj_id) or f"id:{obj_id}"
        cv2.putText(img, text, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, lineType=cv2.LINE_AA)
    return img


def _to_int(x) -> int:
    try:
        return int(x.item())  # torch scalar
    except Exception:
        try:
            return int(x)
        except Exception:
            # last resort
            return int(str(x))


def _to_2d_bool_mask(mask) -> Optional[np.ndarray]:
    """Convert various mask shapes to a 2D boolean array (H, W).

    Accepts torch.Tensor or np.ndarray with shapes like:
    - (1, H, W) -> (H, W)
    - (H, W) -> (H, W)
    - (H, W, 1) -> (H, W)
    - (N, H, W) or (H, W, N) -> take the first channel/slice.
    Returns None if unable to convert to 2D.
    """
    try:
        if isinstance(mask, torch.Tensor):
            arr = mask.detach().cpu().numpy()
        else:
            arr = np.asarray(mask)
        # Squeeze size-1 dims
        arr = np.squeeze(arr)
        # If still >2D, reduce by taking first along leading/trailing axes until 2D
        while arr.ndim > 2:
            # Prefer to drop channels if present
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[-1] == 1:
                arr = arr[..., 0]
            else:
                # Take the first slice along axis 0
                arr = arr[0]
        if arr.ndim != 2:
            return None
        return (arr > 0.0)
    except Exception:
        return None


def _normalize_propagate_item(item):
    """Normalize predictor.propagate_in_video yielded item to (frame_idx, object_ids, masks).

    Supports variants:
      - (frame_idx, masks)
      - (frame_idx, object_ids, masks)
      - (frame_idx, object_ids, masks, ...extras)
    """
    try:
        if isinstance(item, (tuple, list)):
            n = len(item)
            if n == 2:
                frame_idx, masks = item
                object_ids = None
                return frame_idx, object_ids, masks
            elif n >= 3:
                frame_idx = item[0]
                object_ids = item[1]
                masks = item[2]
                return frame_idx, object_ids, masks
    except Exception:
        pass
    # Fallback: try to treat as (frame_idx, masks)
    try:
        frame_idx, masks = item  # type: ignore
        return frame_idx, None, masks
    except Exception:
        return item, None, None


def track_video(
    video_path: str,
    seeds: List[SeedBox],
    *,
    out_path: Optional[str] = None,
    work_dir: Optional[str] = None,
    device: str = "cuda:0",
) -> Dict[str, object]:
    """
    Track multiple objects in a video using SAM2, given normalized seed boxes.

    - Reads video frames to a temporary directory for SAM2's predictor.
    - Adds each seed as a separate obj_id.
    - Propagates masks across frames and renders overlays with labels.
    - Writes an MP4 result video and returns metadata.
    """

    tmp_root = work_dir or tempfile.mkdtemp(prefix="sam2_track_")
    frames_dir = osp.join(tmp_root, "frames")
    _ensure_dir(frames_dir)

    frame_count, width, height, fps = _extract_frames(video_path, frames_dir)

    # Prepare output
    if out_path is None:
        out_path = osp.join(tmp_root, "result.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps or 30.0, (width, height))

    # Build predictor
    predictor = build_sam2_video_predictor(MODEL_CONFIG, MODEL_PATH, device=device)

    # Initialize state and add all seed boxes at frame 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(frames_dir, offload_video_to_cpu=True, offload_state_to_cpu=True, async_loading_frames=True)

        labels_by_id: Dict[int, str] = {}
        for i, seed in enumerate(seeds):
            x1 = max(0, min(width - 1, int(seed.x1 * width)))
            y1 = max(0, min(height - 1, int(seed.y1 * height)))
            x2 = max(0, min(width - 1, int(seed.x2 * width)))
            y2 = max(0, min(height - 1, int(seed.y2 * height)))
            box = (x1, y1, x2, y2)
            labels_by_id[i] = seed.label
            predictor.add_new_points_or_box(state, box=box, frame_idx=0, obj_id=i)

        # Propagate and render frame-by-frame to avoid high memory usage
        for _it in predictor.propagate_in_video(state):
            frame_idx, object_ids, masks = _normalize_propagate_item(_it)
            fi = _to_int(frame_idx)
            frame_path = osp.join(frames_dir, f"{fi + 1:08d}.jpg")
            frame = cv2.imread(frame_path)
            # prepare overlays
            mask_map: Dict[int, np.ndarray] = {}
            bbox_map: Dict[int, Tuple[int, int, int, int]] = {}
            obj_ids_py = [ _to_int(oid) for oid in (object_ids or []) ] if object_ids is not None else list(range(len(masks or [])))
            for obj_id, mask in zip(obj_ids_py, masks or []):
                mask_np = _to_2d_bool_mask(mask)
                if mask_np is None:
                    continue
                mask_map[obj_id] = mask_np
                nz = np.argwhere(mask_np)
                if nz.size == 0 or nz.shape[1] < 2:
                    bbox_map[obj_id] = (0, 0, 0, 0)
                else:
                    y_min, x_min = nz.min(axis=0).tolist()
                    y_max, x_max = nz.max(axis=0).tolist()
                    bbox_map[obj_id] = (x_min, y_min, max(0, x_max - x_min), max(0, y_max - y_min))

            composed = _overlay_masks_and_boxes(frame, mask_map, bbox_map, labels_by_id)
            writer.write(composed)

    writer.release()

    # Optional: cleanup frames to free space
    try:
        for f in os.listdir(frames_dir):
            if f.endswith('.jpg'):
                try:
                    os.remove(osp.join(frames_dir, f))
                except Exception:
                    pass
    except Exception:
        pass

    return {
        "output_path": out_path,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
        "labels": [s.label for s in seeds],
    }


# New helpers to support per-key prediction and final combined overlay
class _ResultsSink:
    """Append-only sink that writes results to Parquet and DuckDB as we go."""

    def __init__(self, parquet_path: str):
        self.parquet_path = parquet_path
        self._rows: List[Dict[str, Any]] = []
        self._batch_size = 500

        self._arrow_schema = pa.schema([
            (pa.field("file_id", pa.string())),
            (pa.field("video_name", pa.string())),
            (pa.field("seed_index", pa.int32())),
            (pa.field("label", pa.string())),
            (pa.field("frame_index", pa.int32())),
            (pa.field("x", pa.int32())),
            (pa.field("y", pa.int32())),
            (pa.field("w", pa.int32())),
            (pa.field("h", pa.int32())),
            (pa.field("area", pa.int64())),
        ])
        self._pq_writer: Optional[pq.ParquetWriter] = None

    def append(self, row: Dict[str, Any]):
        self._rows.append(row)
        if len(self._rows) >= self._batch_size:
            self._flush()

    def _flush(self):
        if not self._rows:
            return
        df = pd.DataFrame(self._rows, columns=[
            "file_id", "video_name", "seed_index", "label", "frame_index", "x", "y", "w", "h", "area"
        ])
        # Append to Parquet
        table = pa.Table.from_pandas(df, schema=self._arrow_schema, preserve_index=False)
        if self._pq_writer is None:
            self._pq_writer = pq.ParquetWriter(self.parquet_path, self._arrow_schema)
        self._pq_writer.write_table(table)
        self._rows.clear()

    def close(self):
        self._flush()
        if self._pq_writer is not None:
            self._pq_writer.close()
            self._pq_writer = None


def _results_schema_json() -> dict:
    return {
        "name": "samurai_ulr_inference",
        "version": 1,
        "description": "Per-frame object tracking results (bounding boxes)",
        "fields": [
            {"name": "file_id", "type": "string", "description": "SurrealDB file record id"},
            {"name": "video_name", "type": "string", "description": "Original video logical name"},
            {"name": "seed_index", "type": "int", "description": "Index of seed box for the object"},
            {"name": "label", "type": "string", "description": "Label of the seed box"},
            {"name": "frame_index", "type": "int", "description": "Zero-based frame index"},
            {"name": "x", "type": "int", "description": "Top-left x"},
            {"name": "y", "type": "int", "description": "Top-left y"},
            {"name": "w", "type": "int", "description": "Width of bbox"},
            {"name": "h", "type": "int", "description": "Height of bbox"},
            {"name": "area", "type": "int", "description": "Mask area in pixels"}
        ],
    }


def _run_single_seed_collect(frames_dir: str, width: int, height: int, seed: SeedBox, seed_index: int,
                             file_id: str, video_name: str, sink: _ResultsSink, *, device: str = "cuda:0") -> None:
    predictor = build_sam2_video_predictor(MODEL_CONFIG, MODEL_PATH, device=device)
    # Use autocast only on CUDA device
    autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"
    with torch.inference_mode(), torch.autocast(autocast_device, dtype=torch.float16 if autocast_device == "cuda" else torch.bfloat16):
        state = predictor.init_state(frames_dir, offload_video_to_cpu=True, offload_state_to_cpu=True, async_loading_frames=True)

        # normalize and add box as obj_id 0 for this single-seed run
        x1 = max(0, min(width - 1, int(seed.x1 * width)))
        y1 = max(0, min(height - 1, int(seed.y1 * height)))
        x2 = max(0, min(width - 1, int(seed.x2 * width)))
        y2 = max(0, min(height - 1, int(seed.y2 * height)))
        predictor.add_new_points_or_box(state, box=(x1, y1, x2, y2), frame_idx=0, obj_id=0)

        for _it in predictor.propagate_in_video(state):
            frame_idx, _object_ids, masks = _normalize_propagate_item(_it)
            fi = _to_int(frame_idx)
            if masks is None:
                continue
            # Normalize as list
            if isinstance(masks, torch.Tensor):
                masks_seq = [masks]
            else:
                try:
                    masks_seq = list(masks)
                except TypeError:
                    masks_seq = [masks]
            if len(masks_seq) == 0:
                continue
            mask_np = _to_2d_bool_mask(masks_seq[0])
            if mask_np is None:
                continue
            nz = np.argwhere(mask_np)
            if nz.size == 0 or nz.shape[1] < 2:
                bx = by = bw = bh = 0
                area = 0
            else:
                y_min, x_min = nz.min(axis=0).tolist()
                y_max, x_max = nz.max(axis=0).tolist()
                bx, by, bw, bh = x_min, y_min, max(0, x_max - x_min), max(0, y_max - y_min)
                area = int(mask_np.sum())

            sink.append({
                "file_id": str(file_id),
                "video_name": str(video_name),
                "seed_index": int(seed_index),
                "label": str(seed.label),
                "frame_index": int(fi),
                "x": int(bx),
                "y": int(by),
                "w": int(bw),
                "h": int(bh),
                "area": int(area),
            })


def _plot_from_parquet(frames_dir: str, results_parquet: str, *,
                       frame_count: int, width: int, height: int, fps: float, out_path: str) -> str:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps or 30.0, (width, height))

    try:
        # Load recorded results
        df = pd.read_parquet(results_parquet)
        # Group rows by frame_index for quick lookup
        groups = df.groupby("frame_index")
        for fi in range(frame_count):
            frame_path = osp.join(frames_dir, f"{fi + 1:08d}.jpg")
            frame = cv2.imread(frame_path)
            if frame is None:
                # If frame missing, skip gracefully
                continue
            # Prepare overlay from saved bboxes
            bbox_map: Dict[int, Tuple[int, int, int, int]] = {}
            labels_by_id: Dict[int, str] = {}
            try:
                g = groups.get_group(fi)
            except Exception:
                g = None
            if g is not None:
                for _, row in g.iterrows():
                    obj_id = int(row["seed_index"])  # one obj per seed
                    x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
                    bbox_map[obj_id] = (x, y, w, h)
                    labels_by_id[obj_id] = str(row["label"]) if not pd.isna(row["label"]) else f"id:{obj_id}"

            # Draw simple rectangles and labels
            img = frame
            for obj_id, (x, y, w, h) in bbox_map.items():
                rng = (37 * (obj_id + 1)) % 255
                c = (int(50 + rng) % 255, int(120 + 2*rng) % 255, int(200 + 3*rng) % 255)
                cv2.rectangle(img, (x, y), (x + w, y + h), c, 2)
                text = labels_by_id.get(obj_id, f"id:{obj_id}")
                cv2.putText(img, text, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, lineType=cv2.LINE_AA)
            writer.write(img)
    finally:
        writer.release()

    return out_path


# ---------------- Adapter for inference pipeline ----------------

from pathlib import Path
from query.annotation_query import get_key_bboxes_for_file
from backend_module.encoder import concat_videos, timelapse_merge
from ml_module.rtdetr_trainer import train_rtdetr
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil


class SamuraiULRModel:
    """Adapter that prepares inputs and runs SAM2 tracking per group of files.

    file_group: list of dicts, each like {
        'file_id': <rid or leaf>,
        'dataset': <str>,
        'name': <str>,
        'segments': [<local segment paths in order>]
    }
    """

    def process_group(self, db_manager, file_group: List[Dict[str, Any]], work_dir: str) -> Dict[str, Any]:
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)

        per_file_outputs: List[str] = []
        per_file_labels: List[str] = []
        results_artifacts: List[Dict[str, str]] = []
        group_parquet_paths: List[str] = []
        schema_json_path = str(work / "samurai_ulr_schema.json")
        # Write schema JSON once per group
        try:
            with open(schema_json_path, "w", encoding="utf-8") as f:
                json.dump(_results_schema_json(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        for item in file_group:
            fid = item.get("file_id")
            segs = item.get("segments") or []
            if not segs:
                continue
            # 1) Timelapse each segment in parallel and merge to keep <= 27000 frames
            merged = str(work / f"{Path(str(fid)).name}_merged_timelapse.mp4")
            try:
                _merged_path, _step = timelapse_merge(segs, merged, max_frames=27000, max_workers=4)
            except Exception as e:
                # If timelapse fails, fall back to simple concat (may still fail due to frame cap)
                merged = str(work / f"{Path(str(fid)).name}_merged.mp4")
                concat_videos(segs, merged)

            # fetch seed boxes from annotations for this file
            anns = get_key_bboxes_for_file(db_manager, fid)
            seeds: List[SeedBox] = []
            for a in anns:
                try:
                    seeds.append(SeedBox(
                        label=str(a.get("label") or ""),
                        x1=float(a.get("x1")), y1=float(a.get("y1")),
                        x2=float(a.get("x2")), y2=float(a.get("y2")),
                    ))
                except Exception:
                    continue
            if not seeds:
                # if no seeds, skip this file
                continue

            # Extract frames once for this file (from timelapse-merged video)
            frames_dir = str(work / f"{Path(str(fid)).name}_frames")
            _ensure_dir(frames_dir)
            # Pre-check total frames; reject >= 27000
            total_fc = _get_total_frame_count(merged)
            if total_fc >= 27000:
                raise ValueError(f"Video {merged} has {total_fc} frames (>=27000); rejecting before inference.")
            frame_count, width, height, fps = _extract_frames(merged, frames_dir)

            # Prepare results sink (Parquet) for this file
            parquet_path = str(work / f"{Path(str(fid)).name}_results.parquet")
            sink = _ResultsSink(parquet_path)

            # Run predict strictly one key per call and record results
            for si, seed in enumerate(seeds):
                _run_single_seed_collect(
                    frames_dir, width, height, seed, si, str(fid), str(item.get("name") or ""), sink
                )
            sink.close()

            # Record artifact paths for upload by the caller
            results_artifacts.append({
                "file_id": str(fid),
                "name": str(item.get("name") or ""),
                "parquet": parquet_path,
            })

            # Plotting: read saved results and overlay per-frame bboxes
            out_path = str(work / f"{Path(str(fid)).name}_tracked.mp4")
            _plot_from_parquet(frames_dir, parquet_path,
                               frame_count=frame_count, width=width, height=height, fps=fps,
                               out_path=out_path)

            per_file_outputs.append(out_path)
            per_file_labels.extend([s.label for s in seeds])

            # 2) YOLO dataset export from SAM2 results (timelapse frames + parquet)
            try:
                dataset_root = Path(work) / f"yolo_dataset_{Path(str(fid)).name}"
                images_train = dataset_root / "images" / "train"
                images_val = dataset_root / "images" / "val"
                images_test = dataset_root / "images" / "test"
                labels_train = dataset_root / "labels" / "train"
                labels_val = dataset_root / "labels" / "val"
                labels_test = dataset_root / "labels" / "test"
                for p in (images_train, images_val, images_test, labels_train, labels_val, labels_test):
                    p.mkdir(parents=True, exist_ok=True)

                df = pd.read_parquet(parquet_path)
                # Map labels to class ids
                label_names = sorted(df["label"].dropna().unique().tolist()) or sorted(set([s.label for s in seeds]))
                cls_map = {name: i for i, name in enumerate(label_names)}
                # Per-frame grouping to write one label file per image
                g = df.groupby("frame_index")
                import random
                # Deterministic random split based on file id
                seed_val = abs(hash(str(fid))) % (2**32)
                rnd = random.Random(seed_val)
                assigned_train = 0
                assigned_val = 0
                assigned_test = 0
                frames_train: list[int] = []
                frames_val: list[int] = []
                frames_test: list[int] = []
                frame_indices_present: list[int] = []
                for fi in range(frame_count):
                    frame_file = os.path.join(frames_dir, f"{fi + 1:08d}.jpg")
                    if not os.path.exists(frame_file):
                        continue
                    frame_indices_present.append(fi)
                    # Split 80/10/10 train:val:test
                    r = rnd.random()
                    if r < 0.8:
                        split = "train"
                        img_dir, lbl_dir = images_train, labels_train
                    elif r < 0.9:
                        split = "val"
                        img_dir, lbl_dir = images_val, labels_val
                    else:
                        split = "test"
                        img_dir, lbl_dir = images_test, labels_test
                    # Copy image into dataset images
                    out_img = img_dir / f"{fi + 1:08d}.jpg"
                    shutil.copy2(frame_file, out_img)
                    # Write YOLO label .txt
                    out_lbl = lbl_dir / f"{fi + 1:08d}.txt"
                    rows = []
                    try:
                        gg = g.get_group(fi)
                        for _, r in gg.iterrows():
                            if int(r.get("w", 0)) <= 0 or int(r.get("h", 0)) <= 0:
                                continue
                            xc = (int(r["x"]) + int(r["w"]) / 2.0) / float(width)
                            yc = (int(r["y"]) + int(r["h"]) / 2.0) / float(height)
                            nw = int(r["w"]) / float(width)
                            nh = int(r["h"]) / float(height)
                            cname = str(r["label"]) if not pd.isna(r["label"]) else seeds[0].label
                            cid = cls_map.get(cname, 0)
                            rows.append(f"{cid} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
                    except Exception:
                        pass
                    with open(out_lbl, "w", encoding="utf-8") as f:
                        if rows:
                            f.write("\n".join(rows))
                        else:
                            f.write("")  # empty file to indicate no objects
                    if split == "train":
                        assigned_train += 1
                        frames_train.append(fi)
                    elif split == "val":
                        assigned_val += 1
                        frames_val.append(fi)
                    else:
                        assigned_test += 1
                        frames_test.append(fi)

                # Ensure both splits have at least one sample if possible
                try:
                    def _move_one(src_split: str, dst_split: str, idx: int):
                        if src_split == "train":
                            s_img, s_lbl = images_train, labels_train
                        elif src_split == "val":
                            s_img, s_lbl = images_val, labels_val
                        else:
                            s_img, s_lbl = images_test, labels_test
                        if dst_split == "train":
                            d_img, d_lbl = images_train, labels_train
                        elif dst_split == "val":
                            d_img, d_lbl = images_val, labels_val
                        else:
                            d_img, d_lbl = images_test, labels_test
                        img = s_img / f"{idx + 1:08d}.jpg"
                        lbl = s_lbl / f"{idx + 1:08d}.txt"
                        if img.exists():
                            shutil.move(str(img), str(d_img / img.name))
                        if lbl.exists():
                            shutil.move(str(lbl), str(d_lbl / lbl.name))

                    # Ensure each split has at least one sample if possible
                    if assigned_train == 0:
                        src = frames_val if len(frames_val) > 1 else (frames_test if len(frames_test) > 1 else [])
                        if src:
                            idx = src[-1]
                            _move_one("val" if src is frames_val else "test", "train", idx)
                            assigned_train += 1
                            if src is frames_val:
                                assigned_val -= 1
                            else:
                                assigned_test -= 1
                    if assigned_val == 0:
                        src = frames_train if len(frames_train) > 1 else (frames_test if len(frames_test) > 1 else [])
                        if src:
                            idx = src[-1]
                            _move_one("train" if src is frames_train else "test", "val", idx)
                            assigned_val += 1
                            if src is frames_train:
                                assigned_train -= 1
                            else:
                                assigned_test -= 1
                    if assigned_test == 0:
                        src = frames_train if len(frames_train) > 1 else (frames_val if len(frames_val) > 1 else [])
                        if src:
                            idx = src[-1]
                            _move_one("train" if src is frames_train else "val", "test", idx)
                            assigned_test += 1
                            if src is frames_train:
                                assigned_train -= 1
                            else:
                                assigned_val -= 1
                except Exception:
                    pass

                # Minimal data.yaml
                data_yaml = dataset_root / "data.yaml"
                with open(data_yaml, "w", encoding="utf-8") as f:
                    f.write(
                        "\n".join([
                            f"path: {dataset_root}",
                            "train: images/train",
                            "val: images/val",
                            "test: images/test",
                            f"names: {label_names}",
                        ])
                    )

                # 3) Train RT-DETR and export TensorRT
                train_out = Path(work) / f"train_result_{Path(str(fid)).name}"
                train_res = train_rtdetr(str(dataset_root), str(train_out), epochs=2, imgsz=640,
                                         base_model="rtdetr-l.pt", export_engine=True, export_int8=True)
                # Pick an inference model path preference: engine > pt
                model_path = train_res.get("engine") or train_res.get("pt")
            except Exception as _e:
                # If training fails, skip downstream inference
                model_path = None

            # 4) Inference on original segments in parallel -> per-segment parquet, then merge
            def _infer_one(seg_path: str, seg_index: int) -> Optional[str]:
                try:
                    from ultralytics import RTDETR
                    if not model_path:
                        return None
                    model = RTDETR(model_path)
                    # Determine local frame count for global indexing
                    cap = cv2.VideoCapture(seg_path)
                    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    cap.release()
                    rows = []
                    frame_counter = -1
                    for results in model.predict(seg_path, stream=True, verbose=False):
                        frame_counter += 1
                        # results.boxes: per-frame detections
                        for box_data in results.boxes:
                            box = box_data.xywh[0]
                            x = int(max(0, min(float(box[0]), 1e9)))
                            y = int(max(0, min(float(box[1]), 1e9)))
                            w_ = int(max(0, min(float(box[2]), 1e9)))
                            h_ = int(max(0, min(float(box[3]), 1e9)))
                            conf = float(box_data.conf)
                            cls_id = int(box_data.cls)
                            rows.append({
                                "segment_index": seg_index,
                                "frame_index_local": frame_counter,
                                "x": x, "y": y, "w": w_, "h": h_,
                                "conf": conf, "class_id": cls_id,
                            })
                    # write parquet for this segment
                    pq_path = str(work / f"{Path(str(fid)).name}_seg{seg_index:03d}_infer.parquet")
                    if rows:
                        df_inf = pd.DataFrame(rows)
                        df_inf.to_parquet(pq_path, index=False)
                        return pq_path
                    else:
                        # write empty parquet with schema
                        df_inf = pd.DataFrame(rows, columns=["segment_index", "frame_index_local", "x", "y", "w", "h", "conf", "class_id"])  # type: ignore
                        df_inf.to_parquet(pq_path, index=False)
                        return pq_path
                except Exception:
                    return None

            seg_paths = list(segs)
            seg_pq_paths: List[str] = []
            if model_path and seg_paths:
                with ThreadPoolExecutor(max_workers=2) as ex:
                    futs = [ex.submit(_infer_one, sp, si) for si, sp in enumerate(seg_paths)]
                    for fu in as_completed(futs):
                        p = fu.result()
                        if p:
                            seg_pq_paths.append(p)

            # Merge segment parquets into group-level parquet for this file
            if seg_pq_paths:
                try:
                    dfs = [pd.read_parquet(p) for p in seg_pq_paths]
                    df_all = pd.concat(dfs, ignore_index=True)
                    group_pq = str(work / f"{Path(str(fid)).name}_infer_merged.parquet")
                    df_all.to_parquet(group_pq, index=False)
                    group_parquet_paths.append(group_pq)
                except Exception:
                    pass

            # Cleanup masks and frames for this file (best-effort)
            # try:
            #     # cleanup frames
            #     for f in os.listdir(frames_dir):
            #         if f.endswith('.jpg'):
            #             try:
            #                 os.remove(os.path.join(frames_dir, f))
            #             except Exception:
            #                 pass
            #     try:
            #         # os.rmdir(frames_dir)
            #     except Exception:
            #         pass
            # except Exception:
            #     pass

        if not per_file_outputs:
            return {"output_path": None, "labels": [], "schema_json_path": schema_json_path, "results_artifacts": results_artifacts}

        # concat per-file outputs if more than one
        if len(per_file_outputs) > 1:
            final_path = str(work / "group_tracked.mp4")
            concat_videos(per_file_outputs, final_path)
        else:
            final_path = per_file_outputs[0]

        # If any group parquet exists, select the first for upload at group level
        group_parquet = group_parquet_paths[0] if group_parquet_paths else None

        return {
            "output_path": final_path,
            "labels": sorted(set(per_file_labels)),
            "schema_json_path": schema_json_path,
            "results_artifacts": results_artifacts,
            "group_parquet": group_parquet,
        }
