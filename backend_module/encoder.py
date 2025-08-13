import subprocess
import tempfile
import json
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
from backend_module.uuid_tools import get_uuid
from concurrent.futures import ThreadPoolExecutor, as_completed


class EncodeError(Exception):
    """Raised when ffmpeg encoding fails."""


@dataclass
class NvencQuality:
    # 目標/上限/バッファ（kbps）
    target_kbps: int
    max_kbps: int
    buf_kbps: int
    # 品質関連
    preset: str = "p5"  # p1..p7（大きいほど高品質・低速）
    profile: str = "high"
    rc: str = "vbr_hq"  # vbr_hq が高品質
    rc_lookahead: int = 32
    spatial_aq: int = 1
    temporal_aq: int = 1
    aq_strength: int = 8  # 0..15 推奨8-12
    b_frames: int = 3
    gop: int = 240


def _parse_fps(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    try:
        if "/" in s:
            n, d = s.split("/", 1)
            return float(n) / float(d)
        return float(s)
    except Exception:
        return None


def _recommend_quality(meta: dict) -> NvencQuality:
    width = meta.get("width") or 1920
    fps = _parse_fps(meta.get("avg_frame_rate")) or 30.0
    # 高動き前提でやや高めのビットレートを推奨
    if width >= 3800 or fps >= 50:
        return NvencQuality(target_kbps=50000, max_kbps=65000, buf_kbps=130000, preset="p5")
    if width >= 2560:
        return NvencQuality(target_kbps=24000, max_kbps=32000, buf_kbps=64000, preset="p5")
    if width >= 1920:
        return NvencQuality(target_kbps=12000, max_kbps=18000, buf_kbps=36000, preset="p5")
    if width >= 1280:
        return NvencQuality(target_kbps=7000, max_kbps=10000, buf_kbps=20000, preset="p5")
    return NvencQuality(target_kbps=4000, max_kbps=6000, buf_kbps=12000, preset="p5")


def encode_to_segments(input_path: str, out_dir: Optional[str] = None, *, nvenc_quality: Optional[NvencQuality] = None) -> List[str]:
    """
    指定された動画ファイルをffmpegで分割エンコードし、生成されたファイルの絶対パスを返す。

    実行コマンド（NVENC使用）:
    ffmpeg -y -nostdin -i INPUT -an -c:v h264_nvenc -preset p4 \
        -f segment -segment_time 180 -reset_timestamps 1 OUT_DIR/out_%03d.mp4

    例外は呼び出し側のtry/exceptで扱う想定。
    """
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input not found: {src}")

    # 出力ディレクトリは実行ごとに UUID サブディレクトリで分離
    # デフォルト: 入力と同階層に out/<uuid>/
    run_id = get_uuid(16)
    base_out = Path(out_dir) if out_dir else (src.parent / "out")
    out_dir_path = base_out / run_id
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # 出力テンプレート
    out_tpl = str(out_dir_path / "out_%03d.mp4")

    # 入力メタから品質推奨を決定
    try:
        meta = probe_video(str(src))
    except Exception:
        meta = {}
    q = nvenc_quality or _recommend_quality(meta)

    # まず NVENC で試行。失敗したら libx264 にフォールバック。
    nvenc_cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-i",
        str(src),
        "-an",
        "-c:v",
        "h264_nvenc",
        "-preset",
        q.preset,
        "-rc",
        q.rc,
        "-b:v",
        f"{q.target_kbps}k",
        "-maxrate",
        f"{q.max_kbps}k",
        "-bufsize",
        f"{q.buf_kbps}k",
        "-profile:v",
        q.profile,
        "-rc-lookahead",
        str(q.rc_lookahead),
        "-spatial_aq",
        str(q.spatial_aq),
        "-temporal_aq",
        str(q.temporal_aq),
        "-aq-strength",
        str(q.aq_strength),
        "-bf",
        str(q.b_frames),
        "-g",
        str(q.gop),
        "-tune",
        "hq",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "segment",
        "-segment_time",
        "180",
        "-reset_timestamps",
        "1",
        out_tpl,
    ]

    proc = subprocess.run(nvenc_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        # NVENC が使えない場合（10bitやGPU非搭載）はCPUエンコードへ切替
        print("CHANGED CPU ENCODE MODE")
        x264_cmd = [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-i",
            str(src),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-bf",
            "3",
            "-g",
            "240",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "segment",
            "-segment_time",
            "180",
            "-reset_timestamps",
            "1",
            out_tpl,
        ]
        proc2 = subprocess.run(x264_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc2.returncode != 0:
            # 両方失敗した場合のみ失敗を返す（NVENC のログを含める）
            raise EncodeError(
                "ffmpeg failed with NVENC and libx264 fallback:\n"
                + "--- NVENC stderr ---\n" + (proc.stderr or "")
                + "\n--- libx264 stderr ---\n" + (proc2.stderr or "")
            )

    # 生成ファイルを列挙
    outputs = sorted(str(p.resolve()) for p in out_dir_path.glob("out_*.mp4"))
    if not outputs:
        raise EncodeError("ffmpeg completed but no output segments were found")
    return outputs


def encode_to_segments_links(input_path: str, out_dir: Optional[str] = None) -> List[str]:
    """
    エンコード済みファイルのリンク（file://）を返す。実体はローカルファイル。
    """
    paths = encode_to_segments(input_path, out_dir)
    return [f"file://{p}" for p in paths]


def probe_video(path: str) -> dict:
    """ffprobeで動画情報を取得し、必要メタを返す。"""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        data = json.loads(proc.stdout or "{}")
    except Exception:
        data = {}

    streams = data.get("streams") or []
    v = None
    for s in streams:
        if s.get("codec_type") == "video":
            v = s
            break
    fmt = data.get("format") or {}

    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    duration = _to_float(fmt.get("duration"))
    if duration is None and v is not None:
        duration = _to_float(v.get("duration"))

    info = {
        "durationSec": duration,
        "width": v.get("width") if v else None,
        "height": v.get("height") if v else None,
        "nb_frames": None,
        "avg_frame_rate": None,
        "codec_name": v.get("codec_name") if v else None,
    }

    if v is not None:
        try:
            info["nb_frames"] = int(v.get("nb_frames")) if v.get("nb_frames") is not None else None
        except Exception:
            info["nb_frames"] = None
        afr = v.get("avg_frame_rate") or v.get("r_frame_rate")
        info["avg_frame_rate"] = afr

    return info


def transcode_video(
    input_path: str,
    output_path: Optional[str] = None,
    *,
    nvenc_quality: Optional[NvencQuality] = None,
    x264_crf: int = 20,
) -> str:
    """
    Re-encode a single video file to H.264 to reduce size.

    - Prefers NVENC (h264_nvenc) with recommended quality; falls back to libx264 CRF.
    - Returns absolute path to the output.
    """
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input not found: {src}")

    # Determine output path
    if output_path is None:
        dst = src.with_name(src.stem + "_enc.mp4")
    else:
        dst = Path(output_path)
    if dst.parent:
        dst.parent.mkdir(parents=True, exist_ok=True)

    # Pick quality params
    try:
        meta = probe_video(str(src))
    except Exception:
        meta = {}
    q = nvenc_quality or _recommend_quality(meta)

    # NVENC attempt
    nvenc_cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-i", str(src),
        "-c:v", "h264_nvenc",
        "-preset", q.preset,
        "-rc", q.rc,
        "-b:v", f"{q.target_kbps}k",
        "-maxrate", f"{q.max_kbps}k",
        "-bufsize", f"{q.buf_kbps}k",
        "-profile:v", q.profile,
        "-rc-lookahead", str(q.rc_lookahead),
        "-spatial_aq", str(q.spatial_aq),
        "-temporal_aq", str(q.temporal_aq),
        "-aq-strength", str(q.aq_strength),
        "-bf", str(q.b_frames),
        "-g", str(q.gop),
        "-tune", "hq",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        str(dst),
    ]
    proc = subprocess.run(nvenc_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        # libx264 fallback
        x264_cmd = [
            "ffmpeg", "-y", "-nostdin",
            "-i", str(src),
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", str(x264_crf),
            "-bf", "3",
            "-g", "240",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            str(dst),
        ]
        proc2 = subprocess.run(x264_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc2.returncode != 0 or not dst.exists():
            raise EncodeError(
                "ffmpeg transcode failed with NVENC and libx264 fallback:\n"
                + "--- NVENC stderr ---\n" + (proc.stderr or "")
                + "\n--- libx264 stderr ---\n" + (proc2.stderr or "")
            )

    return str(dst.resolve())


def create_thumbnail(
    input_path: str,
    output_path: str,
    *,
    timestamp_sec: float = 1.0,
    width: int = 640,
    quality: int = 2,
) -> str:
    """
    指定の動画から単一フレームを切り出してサムネイルを生成する。

    - ffmpeg を使用して `timestamp_sec` で 1 フレーム取得
    - 横幅は `width` を上限にし、アスペクト比を維持（高さは -2）
    - 失敗時は EncodeError を送出

    Returns: 生成されたサムネイルの絶対パス
    """
    src = Path(input_path)
    dst = Path(output_path)
    if not src.exists():
        raise FileNotFoundError(f"Input not found: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)

    vf = f"scale=min({width}\\,iw):-2"
    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-ss",
        str(timestamp_sec),
        "-i",
        str(src),
        "-frames:v",
        "1",
        "-vf",
        vf,
        "-q:v",
        str(quality),
        str(dst),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not dst.exists():
        raise EncodeError(
            "ffmpeg thumbnail generation failed\n"
            + (proc.stderr[-200:] if proc.stderr else "")
        )
    return str(dst.resolve())


def concat_videos(inputs: List[str], output_path: str) -> str:
    """
    Concatenate multiple video files losslessly using ffmpeg concat demuxer.

    inputs: ordered list of file paths with identical codec/parameters.
    output_path: destination file path (e.g., .mp4). Returns absolute path.
    """
    if not inputs:
        raise ValueError("No inputs provided for concatenation")
    out = Path(output_path)
    if out.parent:
        out.parent.mkdir(parents=True, exist_ok=True)

    # Write a temporary list file for ffmpeg concat demuxer
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        for p in inputs:
            f.write(f"file '{str(Path(p).resolve())}'\n")
        list_path = f.name

    cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-f", "concat", "-safe", "0", "-i", list_path,
        "-c", "copy",
        str(out),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not out.exists():
        raise EncodeError(
            "ffmpeg concat failed\n" + (proc.stderr[-300:] if proc.stderr else "")
        )
    return str(out.resolve())


# ---------------- Timelapse utilities ----------------

def _approx_total_frames_by_probe(path: str) -> Optional[int]:
    """Approximate total frames using ffprobe metadata.

    Uses duration * fps when possible, falling back to None when unknown.
    """
    try:
        meta = probe_video(path)
        dur = meta.get("durationSec")
        afr = meta.get("avg_frame_rate")
        fps = _parse_fps(afr) if afr else None
        if dur is not None and fps is not None:
            # safety margin
            return max(0, int(dur * fps + 0.5))
        return None
    except Exception:
        return None


def _make_timelapse_single(input_path: str, output_path: str, step: int) -> str:
    """Create a timelapse by selecting every `step`-th frame and normalizing PTS.

    Keeps visual smoothness by resetting PTS to match display at input FPS.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # ffmpeg filter: select every Nth frame; reset PTS to consecutive frames
    # Using vfr to allow frame dropping and let muxer write variable frame intervals; setpts normalizes timestamps
    vf = f"select='not(mod(n,{step}))',setpts=N/FRAME_RATE/TB"
    cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-i", str(input_path),
        "-vf", vf,
        "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        str(out),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not out.exists():
        raise EncodeError("ffmpeg timelapse failed\n" + (proc.stderr[-300:] if proc.stderr else ""))
    return str(out.resolve())


def timelapse_merge(segments: List[str], merged_out: str, *, max_frames: int = 27000, max_workers: int = 4) -> Tuple[str, int]:
    """
    Create per-segment timelapse videos in parallel and merge them into one.

    - Computes a global decimation step so that the approximate total frames across
      all segments is <= max_frames.
    - Runs per-segment timelapse encoding concurrently.
    - Concatenates the timelapsed segments losslessly.

    Returns: (merged_timelapse_path, step)
    """
    if not segments:
        raise ValueError("No segments provided for timelapse_merge")

    # Estimate total frames across all segments
    approx_frames: int = 0
    unknown = False
    for s in segments:
        n = _approx_total_frames_by_probe(s)
        if n is None:
            unknown = True
            break
        approx_frames += n

    if unknown or approx_frames <= 0:
        # Fallback: assume 30fps and use duration by ffprobe format when available
        approx_frames = 0
        for s in segments:
            try:
                meta = probe_video(s)
                dur = meta.get("durationSec") or 0.0
                fps = _parse_fps(meta.get("avg_frame_rate")) or 30.0
                approx_frames += int(dur * fps + 0.5)
            except Exception:
                # Worst case: assume 30fps * 180s (~5400 frames) per segment
                approx_frames += 5400

    step = max(1, (approx_frames + max_frames - 1) // max_frames)

    # Timelapse each segment concurrently
    tmp_dir = Path(Path(merged_out).parent or ".") / f"tl_{get_uuid(8)}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    args_list = []
    out_paths: List[str] = []
    for i, seg in enumerate(segments):
        out_path = str(tmp_dir / f"tl_{i:03d}.mp4")
        out_paths.append(out_path)
        args_list.append((seg, out_path, step))

    def _worker(a):
        return _make_timelapse_single(*a)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_worker, a) for a in args_list]
        for _ in as_completed(futures):
            pass

    # Merge timelapsed segments
    merged_path = concat_videos(out_paths, merged_out)

    # Best-effort cleanup of temp timelapse parts
    try:
        for p in out_paths:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass
        tmp_dir.rmdir()
    except Exception:
        pass

    return merged_path, step
