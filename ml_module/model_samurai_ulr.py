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
            # Hard safety: abort if frame count threshold exceeded during extraction
            if count >= 16384:
                raise ValueError("Video has 16384 or more frames; aborting per policy.")
            cv2.imwrite(osp.join(out_dir, f"{count:08d}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    finally:
        cap.release()
    return count, w, h, fps


    


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
        schema_json_path = str(work / "samurai_ulr_schema.json")
        # Write schema JSON once per group
        try:
            with open(schema_json_path, "w", encoding="utf-8") as f:
                json.dump(_results_schema_json(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # Progress tracker (do not re-initialize steps here; manager already did)
        tracker = None
        try:
            tracker = InferenceJobProgressTracker(db_manager, str(job_id))
        except Exception:
            tracker = None

        # Preprocess step begins at the first file
        preprocess_started = False

        train_started = False
        infer_started = False
        dataset_started = False

        for item in file_group:
            fid = item.get("file_id")
            segs = item.get("segments") or []
            if not segs:
                continue
            # 1) Timelapse each segment in parallel and merge to keep <= 16384 frames
            if not preprocess_started:
                try:
                    tracker.start("preprocess") if tracker else None
                except Exception:
                    pass
            merged = str(work / f"{Path(str(fid)).name}_merged_timelapse.mp4")
            try:
                _merged_path, _step = timelapse_merge(segs, merged, max_frames=16384, max_workers=4)
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
            # Pre-check total frames; reject >= 16384
            total_fc = _get_total_frame_count(merged)
            if total_fc >= 16384:
                raise ValueError(f"Video {merged} has {total_fc} frames (>=16384); rejecting before inference.")
            frame_count, width, height, fps = _extract_frames(merged, frames_dir)
            if not preprocess_started:
                try:
                    tracker.complete("preprocess") if tracker else None
                except Exception:
                    pass
                preprocess_started = True

            # Start SAM2 immediately after preprocess for clarity in UI
            try:
                tracker.start("sam2") if tracker else None
            except Exception:
                pass

            # Run predict strictly one key per call using CLI and collect results
            # Then merge per-seed parquet files into a single parquet for this file
            import pandas as pd  # local import to avoid global dependency timing
            per_seed_parquets: List[str] = []
            for si, seed in enumerate(seeds):
                # Convert normalized seeds to absolute pixel coords
                x0 = max(0, min(width - 1, int(min(seed.x1, seed.x2) * width)))
                y0 = max(0, min(height - 1, int(min(seed.y1, seed.y2) * height)))
                x1 = max(0, min(width - 1, int(max(seed.x1, seed.x2) * width)))
                y1 = max(0, min(height - 1, int(max(seed.y1, seed.y2) * height)))

                out_json = work / f"{Path(str(fid)).name}_seed{si:03d}_sam2.json"
                cmd = [
                    "python3", "-m", "ml_module.cli_infer_samrai",
                    "--images", str(frames_dir),
                    "--x0", str(x0), "--x1", str(x1),
                    "--y0", str(y0), "--y1", str(y1),
                    "--result", str(out_json),
                ]
                rc = cmd_exec(cmd)
                if rc != 0:
                    continue
                try:
                    import json
                    with open(out_json, "r", encoding="utf-8") as f:
                        j = json.load(f)
                    pq_path = j.get("parquet")
                    if pq_path:
                        per_seed_parquets.append(str(pq_path))
                except Exception:
                    continue

            # Merge per-seed parquet files and add metadata columns
            parquet_path = str(work / f"{Path(str(fid)).name}_results.parquet")
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
            try:
                tracker.complete("sam2") if tracker else None
            except Exception:
                pass

            # Record artifact paths for upload by the caller
            results_artifacts.append({
                "file_id": str(fid),
                "name": str(item.get("name") or ""),
                "parquet": parquet_path,
                # Description to be propagated to DB metadata
                "description": "各ファイルのSAM2推論結果 (Parquet)",
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
                if not dataset_started:
                    try:
                        tracker.start("dataset_export") if tracker else None
                    except Exception:
                        pass
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
                    try:
                        tracker.complete("dataset_export") if tracker else None
                    except Exception:
                        pass
                    dataset_started = True

                # 3) Train RT-DETR via CLI and optionally export TensorRT
                if not train_started:
                    try:
                        # Transition SAM2 -> RT-DETR train on first encounter
                        tracker.complete("sam2") if tracker else None
                        tracker.start("rtdetr_train") if tracker else None
                    except Exception:
                        pass
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
                if rc == 0 and train_json.exists():
                    import json
                    with open(train_json, "r", encoding="utf-8") as f:
                        train_res = json.load(f)
                    model_pt = train_res.get("pt")
                else:
                    model_pt = None

                model_engine = None
                print("DEBUG: [model_pt]")
                if model_pt:
                    # Check global cache to ensure we export TensorRT at most once per weights
                    with _TRT_EXPORT_LOCK:
                        cached = _TRT_EXPORT_CACHE.get(str(model_pt))
                    if cached is not None:
                        model_engine = cached
                        try:
                            # Even if cached, reflect that export is effectively complete
                            tracker.start("trt_export") if tracker else None
                            tracker.complete("trt_export") if tracker else None
                        except Exception:
                            pass
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
                                    try:
                                        tracker.start("trt_export") if tracker else None
                                        tracker.complete("trt_export") if tracker else None
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                        # Perform export only if not already resolved
                        if not model_engine:
                            try:
                                tracker.start("trt_export") if tracker else None
                            except Exception:
                                pass
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
                                try:
                                    tracker.complete("trt_export") if tracker else None
                                except Exception:
                                    pass
                            else:
                                try:
                                    tracker.fail("trt_export") if tracker else None
                                except Exception:
                                    pass
                        # Update cache with result (including None to avoid repeated attempts)
                        with _TRT_EXPORT_LOCK:
                            _TRT_EXPORT_CACHE[str(model_pt)] = model_engine

                # Pick an inference model path preference: engine > pt
                model_path = model_engine or model_pt
            except Exception as _e:
                # If training/export fails, skip downstream inference
                model_path = None

            # 4) Inference on original segments in parallel -> per-segment parquet, then merge
            if not infer_started:
                try:
                    # Transition RT-DETR train -> RT-DETR inference
                    tracker.complete("rtdetr_train") if (tracker and train_started) else None
                    tracker.start("rtdetr_infer") if tracker else None
                except Exception:
                    pass
                infer_started = True
            def _infer_one(seg_path: str, seg_index: int) -> Optional[str]:
                try:
                    if not model_path:
                        return None
                    out_parquet = str(work / f"{Path(str(fid)).name}_seg{seg_index:03d}_infer.parquet")
                    out_video = str(work / f"{Path(str(fid)).name}_seg{seg_index:03d}_infer.mp4")
                    out_json = str(work / f"{Path(str(fid)).name}_seg{seg_index:03d}_infer.json")
                    rc = cmd_exec([
                        "python3", "-m", "ml_module.cli_infer_rtdetr",
                        "--model", str(model_path),
                        "--video", str(seg_path),
                        "--out-parquet", out_parquet,
                        "--out-video", out_video,
                        "--result", out_json,
                    ])
                    if rc != 0:
                        return None
                    return out_parquet if os.path.exists(out_parquet) else None
                except Exception:
                    return None

            seg_paths = list(segs)
            seg_pq_paths: List[str] = []
            if model_path and seg_paths:
                with ThreadPoolExecutor(max_workers=10) as ex:
                    futs = [ex.submit(_infer_one, sp, si) for si, sp in enumerate(seg_paths)]
                    for fu in as_completed(futs):
                        p = fu.result()
                        if p:
                            seg_pq_paths.append(p)

            # Merge segment parquets into group-level parquet for this file
            aggregate_started = False
            if seg_pq_paths:
                try:
                    if not aggregate_started:
                        try:
                            tracker.start("aggregate") if tracker else None
                        except Exception:
                            pass
                    dfs = [pd.read_parquet(p) for p in seg_pq_paths]
                    df_all = pd.concat(dfs, ignore_index=True)
                    group_pq = str(work / f"{Path(str(fid)).name}_infer_merged.parquet")
                    df_all.to_parquet(group_pq, index=False)
                    group_parquet_paths.append(group_pq)
                except Exception:
                    pass
                finally:
                    if not aggregate_started:
                        try:
                            tracker.complete("aggregate") if tracker else None
                        except Exception:
                            pass
                        aggregate_started = True

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

        # Complete inference step for the group
        try:
            tracker.complete("rtdetr_infer") if (tracker and infer_started) else None
        except Exception:
            pass

        # Complete inference step for the group
        try:
            tracker.complete("rtdetr_infer") if (tracker and infer_started) else None
            # If training never started but SAM2 was started, close SAM2
            if tracker and not train_started:
                tracker.complete("sam2")
        except Exception:
            pass

        return {
            "output_path": final_path,
            "labels": sorted(set(per_file_labels)),
            "schema_json_path": schema_json_path,
            "results_artifacts": results_artifacts,
            "group_parquet": group_parquet,
            "temp_datasets": sorted(set(temp_datasets)),
            # Human-readable descriptions propagated to uploader
            "video_description": "タイムラプス動画にSAM2の推論結果をプロットした動画です。",
            "schema_description": "SAM2推論結果のスキーマ定義 (JSON)。",
            "group_parquet_description": "グループ全体のSAM2推論結果 (Parquet)。",
        }
