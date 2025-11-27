import os
import os.path as osp
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Any

import cv2
import pandas as pd
from threading import Lock


@dataclass
class SeedBox:
    label: str
    x1: float  # normalized [0,1]
    y1: float  # normalized [0,1]
    x2: float  # normalized [0,1]
    y2: float  # normalized [0,1]


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# Cache for TensorRT exports to avoid repeated conversions per weights path
# key: path to .pt weights, value: path to exported engine (or None if failed)
_TRT_EXPORT_CACHE: dict[str, Optional[str]] = {}
_TRT_EXPORT_LOCK = Lock()


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
            cv2.imwrite(osp.join(out_dir, f"{count:08d}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    finally:
        cap.release()
    return count, w, h, fps


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
                if isinstance(groups, dict):
                    g = None
                else:
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

            # Draw frame index at top-left
            try:
                idx_text = f"frame: {fi}"
                (tw, th), _ = cv2.getTextSize(idx_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                x0, y0 = 8, 10 + th
                cv2.rectangle(img, (x0 - 6, y0 - th - 6), (x0 + tw + 6, y0 + 6), (0, 0, 0), -1)
                cv2.putText(img, idx_text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            except Exception:
                pass
            writer.write(img)
    finally:
        writer.release()

    return out_path


def _plot_on_video_from_parquet(video_path: str, results_parquet: str, out_path: str) -> str:
    """
    Overlay RT-DETR bbox results (parquet) onto a single concatenated video.

    Assumes parquet has columns: frame_index (0-based), x, y, w, h, label.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps or 30.0, (width, height))

    try:
        df = pd.read_parquet(results_parquet)
        if "frame_index" in df.columns:
            groups = df.groupby("frame_index")
        else:
            groups = {}

        fi = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                g = groups.get_group(fi)
            except Exception:
                g = None

            if g is not None:
                for _, row in g.iterrows():
                    try:
                        x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
                        if w <= 0 or h <= 0:
                            continue
                        # color by label hash for stability across frames
                        lbl = str(row["label"]) if not pd.isna(row.get("label")) else "obj"
                        hv = abs(hash(lbl)) % 255
                        color = (int((50 + 2 * hv) % 255), int((120 + hv) % 255), int((200 + 3 * hv) % 255))
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, lbl, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, lineType=cv2.LINE_AA)
                    except Exception:
                        continue

            # Draw frame index at top-left of the original video
            try:
                idx_text = f"frame: {fi}"
                (tw, th), _ = cv2.getTextSize(idx_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                x0, y0 = 8, 10 + th
                cv2.rectangle(frame, (x0 - 6, y0 - th - 6), (x0 + tw + 6, y0 + 6), (0, 0, 0), -1)
                cv2.putText(frame, idx_text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            except Exception:
                pass

            writer.write(frame)
            fi += 1
    finally:
        writer.release()
        cap.release()

    return out_path


# ---------------- Adapter for inference pipeline ----------------

from pathlib import Path
from query.annotation_query import get_key_bboxes_for_file
from backend_module.encoder import (
    concat_videos_safe,
    timelapse_merge_to_duration,
    timelapse_single,
)
from backend_module.command_executer import cmd_exec
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil


from backend_module.progress_tracker import InferenceJobProgressTracker, StepState


class SamuraiULRModel:
    """Adapter that prepares inputs and runs SAM2 tracking per group of files.

    file_group: list of dicts, each like {
        'file_id': <rid or leaf>,
        'dataset': <str>,
        'name': <str>,
        'segments': [<local segment paths in order>]
    }
    """

    def process_group(self, db_manager, job_id: str, file_group: List[Dict[str, Any]], work_dir: str) -> Dict[str, Any]:
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)

        per_file_outputs: List[str] = []
        per_file_labels: List[str] = []
        results_artifacts: List[Dict[str, str]] = []
        group_parquet_paths: List[str] = []
        temp_datasets: List[str] = []
        # Schema JSON generation/upload is not required.

        # Progress tracker (do not re-initialize steps here; manager already did)
        tracker = None
        try:
            tracker = InferenceJobProgressTracker(db_manager, str(job_id))
        except Exception:
            tracker = None

        failed_steps: set[str] = set()

        def start_step(step_key: str) -> None:
            try:
                tracker.start(step_key) if tracker else None
            except Exception:
                pass

        def complete_step(step_key: str) -> None:
            if step_key in failed_steps:
                return
            try:
                tracker.complete(step_key) if tracker else None
            except Exception:
                pass

        def fail_step(step_key: str) -> None:
            failed_steps.add(step_key)
            try:
                tracker.fail(step_key) if tracker else None
            except Exception:
                pass

        # Preprocess step begins at the first file
        preprocess_started = False

        train_started = False
        infer_started = False
        dataset_started = False

        # Group-level controls
        per_file_infer_enabled = False  # Disable per-file inference; run once over merged group instead
        global_model_path: Optional[str] = None  # Model chosen from first successful train/export
        # Flatten all original segments across the group in member order
        group_seg_paths: List[str] = []
        for it in file_group:
            group_seg_paths.extend(list(it.get("segments") or []))

        # Build a single fixed-duration timelapse (15 minutes) once per group
        merged_ready = False
        merged = str(work / "group_merged_timelapse.mp4")
        for item in file_group:
            fid = item.get("file_id")
            segs = item.get("segments") or []
            if not segs:
                continue
            # 1) Timelapse each segment in parallel to achieve 15-minute total duration
            if not preprocess_started:
                start_step("preprocess")
            # Build a single timelapse video for the whole merge group (only once)
            if not merged_ready and not os.path.exists(merged):
                try:
                    print(f"[samurai/preprocess] timelapse_merge_to_duration start -> {merged}")
                    merged, speed = timelapse_merge_to_duration(
                        group_seg_paths,
                        merged,
                        duration_sec=15 * 60,
                        max_workers=8,
                        cpu_workers=6,
                        gpu_workers=2,
                        backend="auto",
                        target_fps=30,
                    )
                    print(f"[samurai/preprocess] timelapse_merge done  -> {merged}")
                    merged_ready = True
                except Exception:
                    # Robust fallback: safe concat then single-pass speed adjust to 15 minutes
                    raw_merged = str(work / "group_merged_raw.mp4")
                    try:
                        print(f"[samurai/preprocess] concat(safe) fallback start -> {raw_merged}")
                        concat_videos_safe(group_seg_paths, raw_merged)
                        print(f"[samurai/preprocess] concat fallback done  -> {raw_merged}")
                    except Exception:
                        # As a final fallback, pick the first segment to proceed (best-effort)
                        raw_merged = group_seg_paths[0]
                    # Speed adjust to 15 minutes
                    merged = str(work / "group_merged_timelapse.mp4")
                    try:
                        from backend_module.encoder import probe_video  # type: ignore
                        meta = probe_video(raw_merged)
                        dur = float(meta.get("durationSec") or 0.0)
                    except Exception:
                        dur = 0.0
                    speed = (dur / float(15 * 60)) if dur > 0 else 1.0
                    print(f"[samurai/preprocess] single speedup fallback start S={speed:.4f} -> {merged}")
                    # reuse _make_speedup_single via public helper timelapse_single with step=1 then setpts adjust
                    # Here we directly call timelapse_single with step=1 for CFR, then a final correction
                    tmp_tl = str(work / "group_merged_tmp.mp4")
                    timelapse_single(raw_merged, tmp_tl, 1)
                    # final adjust
                    from backend_module.encoder import _make_speedup_single as _speed  # type: ignore
                    _speed(tmp_tl, merged, max(0.0001, speed))
                    print(f"[samurai/preprocess] single speedup fallback done  -> {merged}")
                    merged_ready = True

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
            # Proceed with SAM2/dataset/train only if we have seeds and no model selected yet
            if not seeds:
                # No seeds on this file; move on to next file (training remains pending)
                continue

            # Extract frames once for this file (from timelapse-merged video)
            # Extract frames from the group-level merged timelapse
            frames_dir = str(work / "group_frames")
            _ensure_dir(frames_dir)
            # Frame count is no longer used for gating; proceed to extraction directly
            frame_count, width, height, fps = _extract_frames(merged, frames_dir)
            if not preprocess_started:
                complete_step("preprocess")
                preprocess_started = True

            # Start SAM2 immediately after preprocess for clarity in UI
            start_step("sam2")

            # Run predict strictly one key per call using CLI and collect results
            # Then merge per-seed parquet files into a single parquet for this file
            import pandas as pd  # local import to avoid global dependency timing
            per_seed_parquets: List[str] = []
            sam2_failed = False
            for si, seed in enumerate(seeds):
                # Convert normalized seeds to absolute pixel coords
                x0 = max(0, min(width - 1, int(min(seed.x1, seed.x2) * width)))
                y0 = max(0, min(height - 1, int(min(seed.y1, seed.y2) * height)))
                x1 = max(0, min(width - 1, int(max(seed.x1, seed.x2) * width)))
                y1 = max(0, min(height - 1, int(max(seed.y1, seed.y2) * height)))

                out_json = work / f"group_seed{si:03d}_sam2.json"
                rc = cmd_exec([
                    "python3", "-m", "ml_module.cli_infer_samrai",
                    "--images", str(frames_dir),
                    "--x0", str(x0), "--x1", str(x1),
                    "--y0", str(y0), "--y1", str(y1),
                    "--result", str(out_json),
                ])
                if rc != 0:
                    sam2_failed = True
                    fail_step("sam2")
                    break
                try:
                    import json
                    with open(out_json, "r", encoding="utf-8") as f:
                        j = json.load(f)
                    pq_path = j.get("parquet")
                    if pq_path:
                        per_seed_parquets.append(str(pq_path))
                except Exception:
                    continue

            if sam2_failed:
                # Skip downstream steps for this file when SAM2 CLI fails
                continue

            # Merge per-seed parquet files and add metadata columns for the whole group
            parquet_path = str(work / "group_results.parquet")
            dfs = []
            for si, pqp in enumerate(per_seed_parquets):
                try:
                    df = pd.read_parquet(pqp)
                except Exception:
                    continue
                # normalize/augment columns to match schema
                if "frame_idx" in df.columns and "frame_index" not in df.columns:
                    df = df.rename(columns={"frame_idx": "frame_index"})
                if "area" not in df.columns:
                    try:
                        df["area"] = (df["w"].astype(int) * df["h"].astype(int)).astype(int)
                    except Exception:
                        df["area"] = 0
                # Attribute to the first seeded file for metadata purposes
                df["file_id"] = str(fid)
                df["video_name"] = str(item.get("name") or "")
                df["seed_index"] = int(si)
                df["label"] = str(seeds[si].label if si < len(seeds) else "")
                dfs.append(df)

            try:
                if dfs:
                    df_all = pd.concat(dfs, ignore_index=True)
                else:
                    df_all = pd.DataFrame([], columns=[
                        "file_id", "video_name", "seed_index", "label", "frame_index", "x", "y", "w", "h", "area"
                    ])
                df_all.to_parquet(parquet_path, index=False)
            except Exception:
                # If merge fails, fallback to empty
                try:
                    df_empty = pd.DataFrame([], columns=[
                        "file_id", "video_name", "seed_index", "label", "frame_index", "x", "y", "w", "h", "area"
                    ])
                    df_empty.to_parquet(parquet_path, index=False)
                except Exception:
                    pass

            # Complete SAM2 right after result parquet is ready
            complete_step("sam2")

            # Plotting for timelapse (kept for reference; not used as final output)
            tl_out_path = str(work / "group_tracked_timelapse.mp4")
            _plot_from_parquet(
                frames_dir,
                parquet_path,
                frame_count=frame_count,
                width=width,
                height=height,
                fps=fps,
                out_path=tl_out_path,
            )
            # Keep per-file timelapse plot as a fallback visual output
            try:
                per_file_outputs.append(tl_out_path)
            except Exception:
                pass

            # Record artifact paths for upload by the caller
            results_artifacts.append({
                "file_id": str(fid),
                "name": str(item.get("name") or ""),
                "parquet": parquet_path,
                # Description to be propagated to DB metadata
                "description": "タイムラプス動画のSAM2推論結果（グループ全体）",
                # Also upload the timelapse plot video for the group
                "timelapse_plot": tl_out_path,
                "timelapse_description": "グループ連結タイムラプスにSAM2検出を重畳した動画",
            })

            # Keep labels for group summary
            per_file_labels.extend([s.label for s in seeds])

            # 2) YOLO dataset export from SAM2 results (timelapse frames + parquet)
            try:
                if not dataset_started:
                    start_step("dataset_export")
                # Place dataset under /workspace/src/datasets per RT-DETR spec
                datasets_base = Path("/workspace/src/datasets")
                dataset_name = f"yolo_dataset_{Path(str(fid)).name}"
                dataset_root = datasets_base / dataset_name
                dataset_root.mkdir(parents=True, exist_ok=True)
                try:
                    temp_datasets.append(str(dataset_root))
                except Exception:
                    pass
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
                if not dataset_started:
                    complete_step("dataset_export")
                    dataset_started = True
            except Exception:
                # Dataset export is best-effort; continue even if it fails
                pass

            # 3) Train RT-DETR via CLI and optionally export TensorRT
            if not train_started:
                # Transition SAM2 -> RT-DETR train on first encounter
                complete_step("sam2")
                start_step("rtdetr_train")
                train_started = True
            train_out = Path(work) / f"train_result_{Path(str(fid)).name}"
            train_json = Path(work) / f"train_result_{Path(str(fid)).name}.json"
            rc = cmd_exec([
                "python3", "-m", "ml_module.cli_train_rtdetr",
                # Pass absolute path under /workspace/src/datasets
                "--dataset", str(dataset_root),
                "--out-dir", str(train_out),
                "--epochs", str(8),
                "--base-model", "rtdetr-l.pt",
                "--result", str(train_json),
            ])
            if rc != 0:
                fail_step("rtdetr_train")
            if rc == 0 and train_json.exists():
                import json
                with open(train_json, "r", encoding="utf-8") as f:
                    train_res = json.load(f)
                model_pt = train_res.get("pt")
            else:
                model_pt = None

            try:
                model_engine = None
                if model_pt:
                    # Check global cache to ensure we export TensorRT at most once per weights
                    with _TRT_EXPORT_LOCK:
                        cached = _TRT_EXPORT_CACHE.get(str(model_pt))
                    if cached is not None:
                        model_engine = cached
                        # Even if cached, reflect that export is effectively complete
                        start_step("trt_export")
                        complete_step("trt_export")
                    else:
                        export_json = Path(work) / f"export_engine_{Path(str(fid)).name}.json"
                        # If an export JSON already exists (e.g., rerun), try to reuse it
                        if export_json.exists():
                            try:
                                with open(export_json, "r", encoding="utf-8") as f:
                                    exp_res = json.load(f)
                                maybe_engine = exp_res.get("engine")
                                if maybe_engine and os.path.exists(maybe_engine):
                                    model_engine = maybe_engine
                                    start_step("trt_export")
                                    complete_step("trt_export")
                            except Exception:
                                pass
                        # Perform export only if not already resolved
                        if not model_engine:
                            start_step("trt_export")
                            rc2 = cmd_exec([
                                "python3", "-m", "ml_module.cli_export_trt",
                                "--weights", str(model_pt),
                                "--result", str(export_json),
                            ])
                            if rc2 == 0 and export_json.exists():
                                try:
                                    with open(export_json, "r", encoding="utf-8") as f:
                                        exp_res = json.load(f)
                                    model_engine = exp_res.get("engine")
                                except Exception:
                                    model_engine = None
                                complete_step("trt_export")
                            else:
                                fail_step("trt_export")
                        # Update cache with result (including None to avoid repeated attempts)
                        with _TRT_EXPORT_LOCK:
                            _TRT_EXPORT_CACHE[str(model_pt)] = model_engine

                # Pick an inference model path preference: engine > pt
                model_path = model_engine or model_pt
            except Exception as _e:
                # If training/export fails, skip downstream inference
                model_path = None

            # Capture model path for later group-level inference
            if model_path and (global_model_path is None):
                global_model_path = model_path

        # After preparing model (from first seeded file), run inference once over the entire group
        final_path: Optional[str] = None
        group_parquet: Optional[str] = None
        if global_model_path and group_seg_paths:
            if not infer_started:
                complete_step("rtdetr_train") if train_started else None
                start_step("rtdetr_infer")
                infer_started = True

            infer_failed = False

            def _infer_one_group(seg_path: str, seg_index: int) -> Optional[Tuple[int, str]]:
                nonlocal infer_failed
                try:
                    out_parquet = str(work / f"group_seg{seg_index:03d}_infer.parquet")
                    out_video = str(work / f"group_seg{seg_index:03d}_infer.mp4")
                    out_json = str(work / f"group_seg{seg_index:03d}_infer.json")
                    rc = cmd_exec([
                        "python3", "-m", "ml_module.cli_infer_rtdetr",
                        "--model", str(global_model_path),
                        "--video", str(seg_path),
                        "--out-parquet", out_parquet,
                        "--out-video", out_video,
                        "--result", out_json,
                    ])
                    if rc != 0:
                        infer_failed = True
                        return None
                    return (seg_index, out_parquet) if os.path.exists(out_parquet) else None
                except Exception:
                    infer_failed = True
                    return None

            seg_results_all: Dict[int, str] = {}
            with ThreadPoolExecutor(max_workers=10) as ex:
                futs = [ex.submit(_infer_one_group, sp, si) for si, sp in enumerate(group_seg_paths)]
                for fu in as_completed(futs):
                    r = fu.result()
                    if r and isinstance(r, tuple) and len(r) == 2:
                        si, p = r
                        if isinstance(si, int) and isinstance(p, str):
                            seg_results_all[si] = p
            if infer_failed:
                fail_step("rtdetr_infer")

            # Aggregate across all segments in the group
            aggregate_started = False
            try:
                if not aggregate_started:
                    start_step("aggregate")
                seg_frame_counts: List[int] = []
                for sp in group_seg_paths:
                    try:
                        seg_frame_counts.append(_get_total_frame_count(sp))
                    except Exception:
                        seg_frame_counts.append(0)
                offsets: List[int] = []
                total = 0
                for cnt in seg_frame_counts:
                    offsets.append(total)
                    total += int(cnt or 0)

                dfs_all: List[pd.DataFrame] = []
                for si in range(len(group_seg_paths)):
                    p = seg_results_all.get(si)
                    if not p:
                        continue
                    try:
                        df = pd.read_parquet(p)
                    except Exception:
                        continue
                    if "frame_index" in df.columns:
                        try:
                            df["frame_index"] = df["frame_index"].astype(int) + int(offsets[si])
                        except Exception:
                            pass
                    dfs_all.append(df)

                df_all = pd.concat(dfs_all, ignore_index=True) if dfs_all else pd.DataFrame()
                group_parquet = str(work / "group_infer_merged.parquet")
                df_all.to_parquet(group_parquet, index=False)
                group_parquet_paths.append(group_parquet)

                # Overlay on fully concatenated original video
                concat_path = str(work / "group_concat.mp4")
                try:
                    concat_videos(group_seg_paths, concat_path)
                except Exception:
                    if group_seg_paths:
                        concat_path = group_seg_paths[0]
                try:
                    final_path = str(work / "group_rtdetr_tracked.mp4")
                    _plot_on_video_from_parquet(concat_path, group_parquet, final_path)
                except Exception:
                    final_path = concat_path
            except Exception:
                pass
            finally:
                if not aggregate_started:
                    complete_step("aggregate")
                    aggregate_started = True

        # If we could not produce a final group-level video, fallback to previous per-file outputs
        if not final_path:
            if not per_file_outputs:
                return {"output_path": None, "labels": [], "results_artifacts": results_artifacts}
            if len(per_file_outputs) > 1:
                final_path = str(work / "group_tracked.mp4")
                concat_videos(per_file_outputs, final_path)
            else:
                final_path = per_file_outputs[0]

        # If any group parquet exists, select the first for upload at group level
        group_parquet = group_parquet_paths[0] if group_parquet_paths else None

        # Complete inference step for the group
        complete_step("rtdetr_infer") if infer_started else None
        # If training never started but SAM2 was started, close SAM2
        if not train_started:
            complete_step("sam2")

        return {
            "output_path": final_path,
            "labels": sorted(set(per_file_labels)),
            "results_artifacts": results_artifacts,
            "group_parquet": group_parquet,
            "temp_datasets": sorted(set(temp_datasets)),
            # Human-readable descriptions propagated to uploader
            "video_description": "連結動画にRT-DETRの推論結果をプロットした動画",
            "group_parquet_description": "全ての動画に対する最終推論結果",
        }
