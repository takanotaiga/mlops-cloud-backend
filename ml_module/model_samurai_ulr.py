import os
import os.path as osp
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Any

import cv2
import numpy as np
import torch

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
def _run_single_seed_collect_masks(frames_dir: str, width: int, height: int, seed: SeedBox, out_dir: str, device: str = "cuda:0") -> None:
    os.makedirs(out_dir, exist_ok=True)

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
            # Normalize masks to a list to avoid boolean evaluation on tensors
            if masks is None:
                continue
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
            np.save(osp.join(out_dir, f"{fi:08d}.npy"), mask_np, allow_pickle=False)


def _compose_masks_to_video(frames_dir: str, seed_labels: Dict[int, str], mask_dirs: Dict[int, str], *,
                            frame_count: int, width: int, height: int, fps: float, out_path: str) -> str:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps or 30.0, (width, height))

    try:
        for fi in range(frame_count):
            frame_path = osp.join(frames_dir, f"{fi + 1:08d}.jpg")
            frame = cv2.imread(frame_path)
            if frame is None:
                # If frame missing, skip gracefully
                continue

            mask_map: Dict[int, np.ndarray] = {}
            bbox_map: Dict[int, Tuple[int, int, int, int]] = {}

            for obj_id, mdir in mask_dirs.items():
                mpath = osp.join(mdir, f"{fi:08d}.npy")
                if not osp.exists(mpath):
                    continue
                try:
                    mask_np = np.load(mpath, allow_pickle=False)
                except Exception:
                    continue
                # Ensure mask is 2D bool in case file contains an array of different shape
                mask_bool = _to_2d_bool_mask(mask_np)
                if mask_bool is None:
                    continue
                mask_map[obj_id] = mask_bool
                nz = np.argwhere(mask_bool)
                if nz.size == 0 or nz.shape[1] < 2:
                    bbox_map[obj_id] = (0, 0, 0, 0)
                else:
                    y_min, x_min = nz.min(axis=0).tolist()
                    y_max, x_max = nz.max(axis=0).tolist()
                    bbox_map[obj_id] = (x_min, y_min, max(0, x_max - x_min), max(0, y_max - y_min))

            composed = _overlay_masks_and_boxes(frame, mask_map, bbox_map, seed_labels)
            writer.write(composed)
    finally:
        writer.release()

    return out_path


# ---------------- Adapter for inference pipeline ----------------

from pathlib import Path
from query.annotation_query import get_key_bboxes_for_file
from backend_module.encoder import concat_videos


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

        for item in file_group:
            fid = item.get("file_id")
            segs = item.get("segments") or []
            if not segs:
                continue
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

            # Extract frames once for this file
            frames_dir = str(work / f"{Path(str(fid)).name}_frames")
            _ensure_dir(frames_dir)
            frame_count, width, height, fps = _extract_frames(merged, frames_dir)

            # Run predict strictly one key per call and collect masks
            mask_dirs: Dict[int, str] = {}
            labels_by_id: Dict[int, str] = {}
            for si, seed in enumerate(seeds):
                mask_dir_i = str(work / f"{Path(str(fid)).name}_masks_{si:03d}")
                _run_single_seed_collect_masks(frames_dir, width, height, seed, mask_dir_i)
                mask_dirs[si] = mask_dir_i
                labels_by_id[si] = seed.label

            # Compose all masks simultaneously into one overlaid video
            out_path = str(work / f"{Path(str(fid)).name}_tracked.mp4")
            _compose_masks_to_video(frames_dir, labels_by_id, mask_dirs,
                                    frame_count=frame_count, width=width, height=height, fps=fps,
                                    out_path=out_path)

            per_file_outputs.append(out_path)
            per_file_labels.extend([s.label for s in seeds])

            # Cleanup masks and frames for this file (best-effort)
            try:
                for _k, mdir in mask_dirs.items():
                    try:
                        import shutil
                        shutil.rmtree(mdir, ignore_errors=True)
                    except Exception:
                        pass
                # cleanup frames
                for f in os.listdir(frames_dir):
                    if f.endswith('.jpg'):
                        try:
                            os.remove(os.path.join(frames_dir, f))
                        except Exception:
                            pass
                try:
                    os.rmdir(frames_dir)
                except Exception:
                    pass
            except Exception:
                pass

        if not per_file_outputs:
            return {"output_path": None, "labels": []}

        # concat per-file outputs if more than one
        if len(per_file_outputs) > 1:
            final_path = str(work / "group_tracked.mp4")
            concat_videos(per_file_outputs, final_path)
        else:
            final_path = per_file_outputs[0]

        return {"output_path": final_path, "labels": sorted(set(per_file_labels))}
