from __future__ import annotations

import json
import os
import os.path as osp
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from backend_module.command_executer import cmd_exec
from backend_module.encoder import probe_video, timelapse_single
from backend_module.progress_tracker import InferenceJobProgressTracker
from query.annotation_query import get_key_bboxes_for_file

from .model_samurai_ulr import (
    INFERENCE_BACKEND_PYTORCH_FP16,
    INFERENCE_BACKEND_TENSORRT_FP16,
    SeedBox,
    _ensure_dir,
    _extract_frames,
    _get_inference_backend,
    _get_rtdetr_epochs,
    _plot_from_parquet,
)


def _bounded_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _make_yolo_label_row(row: pd.Series, *, width: int, height: int, cls_map: Dict[str, int], fallback_label: str) -> Optional[str]:
    try:
        w = int(row.get("w", 0))
        h = int(row.get("h", 0))
        if w <= 0 or h <= 0:
            return None
        x = int(row.get("x", 0))
        y = int(row.get("y", 0))
        xc = _bounded_unit((x + w / 2.0) / float(width))
        yc = _bounded_unit((y + h / 2.0) / float(height))
        nw = _bounded_unit(w / float(width))
        nh = _bounded_unit(h / float(height))
        label = str(row.get("label") or fallback_label)
        cid = cls_map.get(label, 0)
        return f"{cid} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"
    except Exception:
        return None


class T260ULRModel:
    """T260-ULR: SAM2.1 pseudo-label generation followed by RF-DETR fine-tuning."""

    def process_group(self, db_manager, job_id: str, file_group: List[Dict[str, Any]], work_dir: str) -> Dict[str, Any]:
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)

        inference_backend = _get_inference_backend(db_manager, str(job_id))
        if inference_backend == INFERENCE_BACKEND_TENSORRT_FP16:
            print("[t260] TensorRT backend is not enabled for RF-DETR yet; falling back to PyTorch FP16")
            inference_backend = INFERENCE_BACKEND_PYTORCH_FP16
        print(f"[t260] job={job_id} inference_backend={inference_backend}")

        tracker = None
        try:
            tracker = InferenceJobProgressTracker(db_manager, str(job_id))
        except Exception:
            tracker = None

        failed_steps: set[str] = set()
        started_steps: set[str] = set()

        def start_step(step_key: str) -> None:
            if step_key in started_steps:
                return
            try:
                tracker.start(step_key) if tracker else None
            except Exception:
                pass
            started_steps.add(step_key)

        def complete_step(step_key: str) -> None:
            if step_key in failed_steps:
                return
            try:
                tracker.complete(step_key) if tracker else None
            except Exception:
                pass
            started_steps.add(step_key)

        def fail_step(step_key: str) -> None:
            failed_steps.add(step_key)
            try:
                tracker.fail(step_key) if tracker else None
            except Exception:
                pass
            started_steps.add(step_key)

        if not file_group:
            raise ValueError("file_group is empty; one video is required")
        if len(file_group) > 1:
            raise ValueError("Only one video input is supported per inference job")

        file_item = file_group[0]
        fid = file_item.get("file_id")
        file_name = str(file_item.get("name") or "")
        group_seg_paths: List[str] = list(file_item.get("segments") or [])
        if not fid or not group_seg_paths:
            raise ValueError("Missing video input for inference")

        results_artifacts: List[Dict[str, str]] = []
        temp_datasets: List[str] = []

        target_frames = int(os.getenv("T260_SAM2_TARGET_FRAMES", str(15 * 60 * 15)))
        timelapse_path = str(work / "timelapse.mp4")
        source_video = group_seg_paths[0]

        start_step("preprocess")
        print(f"[t260] job={job_id} preprocess start segments={len(group_seg_paths)}")
        try:
            step = 1
            timelapse_ready = False
            try:
                meta = probe_video(source_video)
                dur = float(meta.get("durationSec") or 0.0)
                afr = meta.get("avg_frame_rate")
                fps_meta = None
                if isinstance(afr, str) and "/" in afr:
                    num, den = afr.split("/", 1)
                    fps_meta = float(num) / float(den)
                elif isinstance(afr, (int, float)):
                    fps_meta = float(afr)
                if dur > 0 and fps_meta:
                    est_frames = dur * fps_meta
                    if est_frames <= target_frames:
                        timelapse_path = source_video
                        timelapse_ready = True
                    else:
                        step = max(1, int(est_frames / float(target_frames)))
            except Exception:
                pass
            if not timelapse_ready:
                print(f"[t260/preprocess] timelapse_single start step={step} -> {timelapse_path}")
                timelapse_single(source_video, timelapse_path, max(1, step))
            complete_step("preprocess")
        except Exception:
            fail_step("preprocess")
            raise RuntimeError("Preprocess failed for T260 timelapse.")

        anns = get_key_bboxes_for_file(db_manager, fid)
        seeds: List[SeedBox] = []
        for ann in anns:
            try:
                seeds.append(
                    SeedBox(
                        label=str(ann.get("label") or "object"),
                        x1=float(ann.get("x1")),
                        y1=float(ann.get("y1")),
                        x2=float(ann.get("x2")),
                        y2=float(ann.get("y2")),
                    )
                )
            except Exception:
                continue

        if not seeds:
            start_step("sam2")
            fail_step("sam2")
            raise RuntimeError("No seed boxes found in group; aborting.")

        frames_dir = str(work / "group_frames")
        _ensure_dir(frames_dir)
        frame_count, width, height, fps = _extract_frames(timelapse_path, frames_dir)

        start_step("sam2")
        per_seed_parquets: List[str] = []
        for seed_index, seed in enumerate(seeds):
            x0 = max(0, min(width - 1, int(min(seed.x1, seed.x2) * width)))
            y0 = max(0, min(height - 1, int(min(seed.y1, seed.y2) * height)))
            x1 = max(0, min(width - 1, int(max(seed.x1, seed.x2) * width)))
            y1 = max(0, min(height - 1, int(max(seed.y1, seed.y2) * height)))

            out_json = work / f"group_seed{seed_index:03d}_sam2.json"
            rc = cmd_exec(
                [
                    "uv",
                    "run",
                    "-m",
                    "ml_module.cli_infer_sam2_1",
                    "--images",
                    str(frames_dir),
                    "--x0",
                    str(x0),
                    "--x1",
                    str(x1),
                    "--y0",
                    str(y0),
                    "--y1",
                    str(y1),
                    "--result",
                    str(out_json),
                ]
            )
            if rc != 0:
                fail_step("sam2")
                raise RuntimeError("SAM2.1 inference failed; see logs for details.")
            try:
                payload = json.loads(out_json.read_text(encoding="utf-8"))
                parquet = payload.get("parquet")
                if parquet:
                    per_seed_parquets.append(str(parquet))
            except Exception:
                pass

        parquet_path = str(work / "group_results.parquet")
        dfs = []
        for seed_index, seed_parquet in enumerate(per_seed_parquets):
            try:
                df = pd.read_parquet(seed_parquet)
            except Exception:
                continue
            if "frame_idx" in df.columns and "frame_index" not in df.columns:
                df = df.rename(columns={"frame_idx": "frame_index"})
            if "area" not in df.columns:
                try:
                    df["area"] = (df["w"].astype(int) * df["h"].astype(int)).astype(int)
                except Exception:
                    df["area"] = 0
            df["file_id"] = str(fid)
            df["video_name"] = file_name
            df["seed_index"] = int(seed_index)
            df["label"] = str(seeds[seed_index].label if seed_index < len(seeds) else "object")
            dfs.append(df)

        if dfs:
            df_all = pd.concat(dfs, ignore_index=True)
        else:
            df_all = pd.DataFrame([], columns=["file_id", "video_name", "seed_index", "label", "frame_index", "x", "y", "w", "h", "area"])
        df_all.to_parquet(parquet_path, index=False)
        complete_step("sam2")

        timelapse_plot = str(work / "group_tracked_timelapse.mp4")
        _plot_from_parquet(
            frames_dir,
            parquet_path,
            frame_count=frame_count,
            width=width,
            height=height,
            fps=fps,
            out_path=timelapse_plot,
        )
        results_artifacts.append(
            {
                "file_id": str(fid),
                "name": file_name,
                "parquet": parquet_path,
                "description": "SAM2.1によるタイムラプス動画の疑似ラベル結果",
                "timelapse_plot": timelapse_plot,
                "timelapse_description": "SAM2.1の追跡結果を重畳したタイムラプス動画",
            }
        )

        start_step("dataset_export")
        try:
            datasets_base = Path("/workspace/src/datasets")
            dataset_name = f"t260_dataset_{Path(str(fid)).name}"
            dataset_root = datasets_base / dataset_name
            if dataset_root.exists():
                shutil.rmtree(dataset_root)
            dataset_root.mkdir(parents=True, exist_ok=True)
            temp_datasets.append(str(dataset_root))

            split_dirs = {
                "train": (dataset_root / "train" / "images", dataset_root / "train" / "labels"),
                "valid": (dataset_root / "valid" / "images", dataset_root / "valid" / "labels"),
                "test": (dataset_root / "test" / "images", dataset_root / "test" / "labels"),
            }
            for image_dir, label_dir in split_dirs.values():
                image_dir.mkdir(parents=True, exist_ok=True)
                label_dir.mkdir(parents=True, exist_ok=True)

            df = pd.read_parquet(parquet_path)
            seed_labels = [s.label for s in seeds if s.label] or ["object"]
            label_names = sorted(df["label"].dropna().unique().tolist()) if "label" in df.columns else []
            label_names = label_names or sorted(set(seed_labels))
            cls_map = {name: idx for idx, name in enumerate(label_names)}
            grouped = df.groupby("frame_index") if "frame_index" in df.columns else {}
            frame_files = sorted(Path(frames_dir).glob("*.jpg"))
            if not frame_files:
                raise RuntimeError("No frames extracted for RF-DETR dataset export")

            rnd = random.Random(abs(hash(str(fid))) % (2**32))
            for index, frame_path in enumerate(frame_files):
                try:
                    frame_index = int(frame_path.stem) - 1
                except Exception:
                    continue
                if len(frame_files) < 10:
                    split = "train"
                else:
                    roll = rnd.random()
                    split = "train" if roll < 0.8 else "valid" if roll < 0.9 else "test"
                image_dir, label_dir = split_dirs[split]
                shutil.copy2(frame_path, image_dir / frame_path.name)

                rows: List[str] = []
                try:
                    frame_rows = grouped.get_group(frame_index)
                    for _, row in frame_rows.iterrows():
                        label_row = _make_yolo_label_row(row, width=width, height=height, cls_map=cls_map, fallback_label=label_names[0])
                        if label_row:
                            rows.append(label_row)
                except Exception:
                    pass
                (label_dir / f"{frame_index + 1:08d}.txt").write_text("\n".join(rows), encoding="utf-8")

            data_yaml = dataset_root / "data.yaml"
            data_yaml.write_text(
                "\n".join(
                    [
                        f"path: {dataset_root}",
                        "train: train/images",
                        "val: valid/images",
                        "test: test/images",
                        f"names: {label_names}",
                    ]
                ),
                encoding="utf-8",
            )
            complete_step("dataset_export")
        except Exception:
            fail_step("dataset_export")
            raise

        start_step("rtdetr_train")
        try:
            train_out = work / f"rfdetr_train_{Path(str(fid)).name}"
            train_json = work / f"rfdetr_train_{Path(str(fid)).name}.json"
            epochs = _get_rtdetr_epochs(db_manager, str(job_id))
            variant = os.getenv("T260_RFDETR_VARIANT", "medium")
            batch_size = int(os.getenv("T260_RFDETR_BATCH_SIZE", "4"))
            grad_accum_steps = int(os.getenv("T260_RFDETR_GRAD_ACCUM_STEPS", "4"))
            lr = float(os.getenv("T260_RFDETR_LR", "0.0001"))
            rc = cmd_exec(
                [
                    "uv",
                    "run",
                    "-m",
                    "ml_module.cli_train_rfdetr",
                    "--dataset",
                    str(dataset_root),
                    "--out-dir",
                    str(train_out),
                    "--epochs",
                    str(epochs),
                    "--variant",
                    variant,
                    "--batch-size",
                    str(batch_size),
                    "--grad-accum-steps",
                    str(grad_accum_steps),
                    "--lr",
                    str(lr),
                    "--result",
                    str(train_json),
                ]
            )
            print(f"[t260] job={job_id} train rc={rc} out={train_json}")
            if rc != 0:
                raise RuntimeError("RF-DETR training failed.")
            train_res = json.loads(train_json.read_text(encoding="utf-8"))
            checkpoint = train_res.get("checkpoint")
            if not checkpoint or not osp.exists(str(checkpoint)):
                raise RuntimeError("RF-DETR training did not produce a checkpoint.")
            complete_step("rtdetr_train")
        except Exception:
            fail_step("rtdetr_train")
            raise

        start_step("trt_export")
        complete_step("trt_export")

        start_step("rtdetr_infer")
        final_parquet = str(work / "group_infer.parquet")
        final_video = str(work / "group_infer.mp4")
        infer_json = str(work / "group_infer.json")
        try:
            conf = float(os.getenv("T260_RFDETR_INFER_CONF", "0.25"))
            infer_cmd = [
                "uv",
                "run",
                "-m",
                "ml_module.cli_infer_rfdetr",
                "--model",
                str(checkpoint),
                "--video",
                str(source_video),
                "--out-parquet",
                final_parquet,
                "--out-video",
                final_video,
                "--variant",
                variant,
                "--conf",
                str(conf),
                "--result",
                infer_json,
            ]
            if inference_backend == INFERENCE_BACKEND_PYTORCH_FP16:
                infer_cmd.append("--half")
            rc = cmd_exec(infer_cmd)
            print(f"[t260] job={job_id} infer rc={rc} input={source_video}")
            if rc != 0 or not osp.exists(final_parquet):
                raise RuntimeError("RF-DETR inference failed.")
            complete_step("rtdetr_infer")
        except Exception:
            fail_step("rtdetr_infer")
            raise

        start_step("aggregate")
        complete_step("aggregate")

        return {
            "output_path": final_video if osp.exists(final_video) else None,
            "labels": sorted(set(label_names)),
            "results_artifacts": results_artifacts,
            "group_parquet": final_parquet,
            "temp_datasets": sorted(set(temp_datasets)),
            "video_description": "T260-ULRでRF-DETRの推論結果をプロットした動画",
            "group_parquet_description": "T260-ULRによる最終推論結果",
        }
