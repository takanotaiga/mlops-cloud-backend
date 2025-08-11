"""
SAM2-based video object tracking integrated for this project.

This module provides a minimal, dependency-light wrapper around
Meta's SAM2 video predictor to track multiple objects from initial
bounding boxes and render a result video. It purposely avoids the
previous YOLO/horus/gradio utilities.
"""

import os
import os.path as osp
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

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
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            frame_path = osp.join(frames_dir, f"{frame_idx + 1:08d}.jpg")
            frame = cv2.imread(frame_path)
            # prepare overlays
            mask_map: Dict[int, np.ndarray] = {}
            bbox_map: Dict[int, Tuple[int, int, int, int]] = {}
            for obj_id, mask in zip(object_ids, masks):
                mask_np = mask[0].detach().cpu().numpy() > 0.0
                mask_map[obj_id] = mask_np
                nz = np.argwhere(mask_np)
                if nz.size == 0:
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

