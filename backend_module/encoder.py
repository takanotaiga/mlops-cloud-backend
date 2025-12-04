import subprocess
import tempfile
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend_module.command_executer import cmd_exec
from backend_module.uuid_tools import get_uuid


class EncodeError(Exception):
    """Raised when ffmpeg encoding fails."""


def _run_with_executor(cmd: List[str]) -> Tuple[int, str]:
    """Run command via shared executor, capturing combined output."""
    res = cmd_exec(cmd, capture_output=True)
    if isinstance(res, tuple) and len(res) == 2:
        rc, out = res
    else:
        rc = res if isinstance(res, int) else -1
        out = ""
    return int(rc) if rc is not None else -1, out or ""


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


GOP_DEFAULT = 240


def encode_to_hls(
    input_path: str,
    out_dir: Optional[str] = None,
    *,
    segment_time: int = 6,
    backend: Literal["auto", "gpu", "cpu"] = "auto",
) -> Dict[str, List[str] | str]:
    """
    Encode input video to HLS (VOD) with fMP4 segments.

    - Prefers NVENC (hevc_nvenc) with HEVC sample-like settings; falls back to libx265.
    - Produces: playlist.m3u8, init.mp4, and seg_XXXXX.m4s files in an isolated run directory.
    - Returns a dict with absolute paths:
        { 'playlist': <m3u8>, 'segments': [<m4s/... including init.mp4>], 'out_dir': <dir> }
    """
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input not found: {src}")

    run_id = get_uuid(16)
    base_out = Path(out_dir) if out_dir else (src.parent / "out")
    out_dir_path = base_out / run_id / "hls"
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Output names
    playlist_name = "index.m3u8"
    init_name = "init.mp4"
    seg_tpl = str(out_dir_path / "seg_%05d.m4s")
    playlist_path = str(out_dir_path / playlist_name)

    # Fixed GOP from sample presets
    gop = GOP_DEFAULT

    common_hls = [
        "-an",
        "-pix_fmt", "yuv420p",
        "-g", str(gop),
        "-keyint_min", str(gop),
        "-sc_threshold", "0",
        "-f", "hls",
        "-hls_time", str(segment_time),
        "-hls_playlist_type", "vod",
        "-hls_segment_type", "fmp4",
        "-hls_fmp4_init_filename", init_name,
        "-hls_flags", "independent_segments+append_list",
        "-hls_segment_filename", seg_tpl,
        playlist_path,
    ]

    nvenc_cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-i", str(src),
        "-c:v", "hevc_nvenc",
        # Align with provided HEVC NVENC sample
        "-preset", "p1",
        "-rc", "vbr",
        "-crf", "24",
        "-rc-lookahead", "20",
        "-spatial_aq", "1",
        "-temporal_aq", "1",
        "-aq-strength", "8",
        "-bf", "0",
        "-tune", "hq",
    ] + common_hls

    if backend == "gpu":
        rc, out = _run_with_executor(nvenc_cmd)
        if rc != 0:
            raise EncodeError("ffmpeg HLS failed on GPU NVENC (forced)\n" + (out or ""))
    elif backend == "cpu":
        x264_cmd = [
            "ffmpeg", "-y", "-nostdin",
            "-i", str(src),
            "-c:v", "libx265",
            "-preset", "ultrafast",
            "-crf", "24",
            "-bf", "0",
        ] + common_hls
        rc2, out2 = _run_with_executor(x264_cmd)
        if rc2 != 0:
            raise EncodeError("ffmpeg HLS failed on CPU libx265 (forced)\n" + (out2 or ""))
    else:
        rc, out = _run_with_executor(nvenc_cmd)
        if rc != 0:
            # Fallback to libx265
            x264_cmd = [
                "ffmpeg", "-y", "-nostdin",
                "-i", str(src),
                "-c:v", "libx265",
                "-preset", "ultrafast",
                "-crf", "24",
                "-bf", "0",
            ] + common_hls
            rc2, out2 = _run_with_executor(x264_cmd)
            if rc2 != 0:
                raise EncodeError(
                    "ffmpeg HLS failed with NVENC(hevc_nvenc) and libx265 fallback:\n"
                    + "--- NVENC stderr ---\n" + (out or "")
                    + "\n--- libx265 stderr ---\n" + (out2 or "")
                )

    # Collect outputs
    # Include init.mp4 and all .m4s segments
    segs = []
    init_path = out_dir_path / init_name
    if init_path.exists():
        segs.append(str(init_path.resolve()))
    segs.extend(sorted(str(p.resolve()) for p in out_dir_path.glob("seg_*.m4s")))

    if not Path(playlist_path).exists() or len(segs) == 0:
        raise EncodeError("HLS generation completed but outputs are missing")

    return {"playlist": str(Path(playlist_path).resolve()), "segments": segs, "out_dir": str(out_dir_path.resolve())}


def hls_to_mp4(playlist_path: str, output_path: str) -> str:
    """
    Repackage an HLS playlist (fMP4 segments) into a single MP4 using stream copy.

    - Does not re-encode; uses `-c copy`.
    - Assumes playlist and segments are locally accessible.
    """
    playlist = Path(playlist_path)
    if not playlist.exists():
        raise FileNotFoundError(f"HLS playlist not found: {playlist}")

    out = Path(output_path)
    if out.parent:
        out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-allowed_extensions",
        "ALL",
        "-i",
        str(playlist),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(out),
    ]
    rc, out_log = _run_with_executor(cmd)
    if rc != 0 or not out.exists():
        raise EncodeError("ffmpeg HLS->MP4 repack failed\n" + ((out_log or "")[-300:] if out_log else ""))
    return str(out.resolve())


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
    x264_crf: int = 24,
) -> str:
    """
    Re-encode a single video file to HEVC to reduce size.

    - Prefers NVENC (hevc_nvenc) with sample-like settings; falls back to libx265 CRF.
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

    # Fixed GOP from sample presets
    gop = GOP_DEFAULT

    # NVENC attempt
    nvenc_cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-i", str(src),
        "-c:v", "hevc_nvenc",
        # Align with provided HEVC NVENC sample
        "-preset", "p1",
        "-rc", "vbr",
        "-crf", "24",
        "-rc-lookahead", "20",
        "-spatial_aq", "1",
        "-temporal_aq", "1",
        "-aq-strength", "8",
        "-bf", "0",
        "-g", str(gop),
        "-tune", "hq",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        str(dst),
    ]
    rc, out = _run_with_executor(nvenc_cmd)
    if rc != 0:
        # libx265 fallback
        x264_cmd = [
            "ffmpeg", "-y", "-nostdin",
            "-i", str(src),
            "-c:v", "libx265",
            "-preset", "ultrafast",
            "-crf", str(x264_crf),
            "-bf", "0",
            "-g", str(gop),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            str(dst),
        ]
        rc2, out2 = _run_with_executor(x264_cmd)
        if rc2 != 0 or not dst.exists():
            raise EncodeError(
                "ffmpeg transcode failed with NVENC(hevc_nvenc) and libx265 fallback:\n"
                + "--- NVENC stderr ---\n" + (out or "")
                + "\n--- libx265 stderr ---\n" + (out2 or "")
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
    rc, out = _run_with_executor(cmd)
    if rc != 0 or not dst.exists():
        raise EncodeError(
            "ffmpeg thumbnail generation failed\n"
            + ((out or "")[-200:] if out else "")
        )
    return str(dst.resolve())


def concat_videos(inputs: List[str], output_path: str) -> str:
    """
    Concatenate multiple video files losslessly using ffmpeg concat demuxer.

    inputs: ordered list of file paths with identical codec/parameters.
    output_path: destination file path (e.g., .mp4). Returns absolute path.
    NOTE: This uses stream copy and assumes identical bitstreams; for safety across
    heterogeneous inputs use concat_videos_safe which re-encodes.
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
    rc, out_log = _run_with_executor(cmd)
    if rc != 0 or not out.exists():
        raise EncodeError(
            "ffmpeg concat failed\n" + ((out_log or "")[-300:] if out_log else "")
        )
    return str(out.resolve())


def concat_videos_safe(inputs: List[str], output_path: str, *, backend: Literal["auto", "gpu", "cpu"] = "auto") -> str:
    """
    Concatenate multiple video files by decoding and re-encoding to HEVC for robust output.

    - Avoids stream copy issues like "Zero refs for a frame with P or B slices" at boundaries.
    - Produces CFR, yuv420p, fixed GOP, no B-frames for predictable downstream decode.
    """
    if not inputs:
        raise ValueError("No inputs provided for concatenation")
    out = Path(output_path)
    if out.parent:
        out.parent.mkdir(parents=True, exist_ok=True)

    # Write list for concat demuxer
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        for p in inputs:
            f.write(f"file '{str(Path(p).resolve())}'\n")
        list_path = f.name

    gop = GOP_DEFAULT
    # Common re-encode args (CFR, no B, fixed GOP, pix_fmt yuv420p)
    common = [
        "-f", "concat", "-safe", "0", "-i", list_path,
        "-an",
        "-vsync", "cfr",
        "-pix_fmt", "yuv420p",
        "-g", str(gop),
        "-keyint_min", str(gop),
        "-bf", "0",
        "-sc_threshold", "0",
    ]
    # Attempt chain: NVDec HEVC -> NVDec H264 -> normal decode -> CPU
    hevc_nv = [
        "ffmpeg", "-y", "-nostdin",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-c:v", "hevc_cuvid",
    ] + common + [
        "-c:v", "hevc_nvenc", "-preset", "p1", "-rc", "vbr", "-crf", "24",
        "-rc-lookahead", "20", "-spatial_aq", "1", "-temporal_aq", "1", "-aq-strength", "8",
        "-tune", "hq",
        str(out),
    ]
    h264_nv = [
        "ffmpeg", "-y", "-nostdin",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-c:v", "h264_cuvid",
    ] + common + [
        "-c:v", "hevc_nvenc", "-preset", "p1", "-rc", "vbr", "-crf", "24",
        "-rc-lookahead", "20", "-spatial_aq", "1", "-temporal_aq", "1", "-aq-strength", "8",
        "-tune", "hq",
        str(out),
    ]
    nv = [
        "ffmpeg", "-y", "-nostdin",
    ] + common + [
        "-c:v", "hevc_nvenc", "-preset", "p1", "-rc", "vbr", "-crf", "24",
        "-rc-lookahead", "20", "-spatial_aq", "1", "-temporal_aq", "1", "-aq-strength", "8",
        "-tune", "hq",
        str(out),
    ]
    x = [
        "ffmpeg", "-y", "-nostdin",
    ] + common + [
        "-c:v", "libx265", "-preset", "ultrafast", "-crf", "24",
        str(out),
    ]
    rc, log = _run_with_executor(hevc_nv)
    if rc != 0:
        rc2, log2 = _run_with_executor(h264_nv)
        if rc2 != 0:
            rc3, log3 = _run_with_executor(nv)
            if rc3 != 0:
                rc4, log4 = _run_with_executor(x)
                if rc4 != 0 or not out.exists():
                    msg = "ffmpeg concat(re-encode) failed across NVDec(hevc/h264) and NVEnc, CPU fallback also failed\n"
                    detail = log4 or log3 or log2 or log
                    raise EncodeError(msg + (detail or ""))
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


def _count_frames_precise(path: str) -> Optional[int]:
    """Fast frame count without full decode.

    Prefers stream nb_frames when present; else approximates via duration * fps.
    Avoids expensive `-count_frames` which can fully decode long videos.
    """
    try:
        meta = probe_video(path)
        # Try direct nb_frames first (lightweight)
        nb = meta.get("nb_frames")
        if isinstance(nb, int):
            return nb
        # Approximate from duration and fps
        dur = meta.get("durationSec") or 0.0
        fps = _parse_fps(meta.get("avg_frame_rate")) or None
        if dur and fps:
            return max(0, int(dur * fps + 0.5))
    except Exception:
        pass
    return None

def _count_frames_decode(path: str) -> Optional[int]:
    """Strict frame count by decoding metadata via ffprobe -count_frames.

    This is slower for long videos but reliable when deciding threshold actions.
    """
    import subprocess
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames,nb_frames",
            "-of", "default=nokey=1:noprint_wrappers=1",
            path,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out = (proc.stdout or "").strip().splitlines()
        vals = []
        for line in out:
            try:
                vals.append(int(line.strip()))
            except Exception:
                pass
        if vals:
            return max(vals)
    except Exception:
        return None
    return None


def count_frames(path: str) -> Optional[int]:
    """Public helper to count frames (precise when possible)."""
    return _count_frames_precise(path)

def count_frames_strict(path: str) -> Optional[int]:
    """Public helper to count frames strictly (decoding when needed)."""
    n = _count_frames_precise(path)
    if n is not None:
        return n
    return _count_frames_decode(path)


def timelapse_single(
    input_path: str,
    output_path: str,
    step: int,
    *,
    backend: Literal["auto", "gpu", "cpu"] = "auto",
) -> str:
    """Public helper to create a timelapse for a single input.

    Encodes in H.265 using GPU when available, falling back to CPU.
    Returns the absolute path to the output.
    """
    return _make_timelapse_single(input_path, output_path, step, backend=backend)

def _make_timelapse_single(
    input_path: str,
    output_path: str,
    step: int,
    *,
    backend: Literal["auto", "gpu", "cpu"] = "auto",
) -> str:
    """Create a timelapse by selecting every `step`-th frame and encoding to H.265.

    - Prefers NVENC (hevc_nvenc) for speed and parity with video_manager's hybrid strategy.
    - Falls back to libx265 on CPU when NVENC is unavailable.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # ffmpeg filter: select every Nth frame; normalize PTS for smooth playback
    vf = f"select='not(mod(n,{step}))',setpts=N/FRAME_RATE/TB,fps=30"

    gop = GOP_DEFAULT
    print(f"[timelapse] start file={input_path} step={step} backend={backend}")
    # Build attempt chain for faster decode: NVDec HEVC -> NVDec H264 -> normal -> CPU
    hevc_cuvid = [
        "ffmpeg", "-y", "-nostdin",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-c:v", "hevc_cuvid",
        "-i", str(input_path),
        "-vf", f"hwdownload,format=nv12,{vf}",
        "-an", "-vsync", "cfr",
        "-c:v", "hevc_nvenc",
        "-preset", "p1", "-rc", "vbr", "-crf", "24",
        "-rc-lookahead", "20", "-spatial_aq", "1", "-temporal_aq", "1", "-aq-strength", "8",
        "-bf", "0", "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
        "-tune", "hq", "-pix_fmt", "yuv420p",
        str(out),
    ]
    h264_cuvid = [
        "ffmpeg", "-y", "-nostdin",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-c:v", "h264_cuvid",
        "-i", str(input_path),
        "-vf", f"hwdownload,format=nv12,{vf}",
        "-an", "-vsync", "cfr",
        "-c:v", "hevc_nvenc",
        "-preset", "p1", "-rc", "vbr", "-crf", "24",
        "-rc-lookahead", "20", "-spatial_aq", "1", "-temporal_aq", "1", "-aq-strength", "8",
        "-bf", "0", "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
        "-tune", "hq", "-pix_fmt", "yuv420p",
        str(out),
    ]
    nvenc_cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-i", str(input_path),
        "-vf", vf,
        "-an", "-vsync", "cfr",
        "-c:v", "hevc_nvenc",
        "-preset", "p1", "-rc", "vbr", "-crf", "24",
        "-rc-lookahead", "20", "-spatial_aq", "1", "-temporal_aq", "1", "-aq-strength", "8",
        "-bf", "0", "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
        "-tune", "hq", "-pix_fmt", "yuv420p",
        str(out),
    ]
    x265_cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-i", str(input_path),
        "-vf", vf,
        "-an", "-vsync", "cfr",
        "-c:v", "libx265",
        "-preset", "ultrafast", "-crf", "24",
        "-bf", "0", "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
        "-pix_fmt", "yuv420p",
        str(out),
    ]
    def _try_chain() -> None:
        # Try NVDec HEVC
        rc, log = _run_with_executor(hevc_cuvid)
        if rc == 0:
            return
        # Try NVDec H264
        rc2, log2 = _run_with_executor(h264_cuvid)
        if rc2 == 0:
            return
        # Try normal decode -> NVEnc
        rc3, log3 = _run_with_executor(nvenc_cmd)
        if rc3 == 0:
            return
        # CPU fallback
        rc4, log4 = _run_with_executor(x265_cmd)
        if rc4 != 0:
            msg = "ffmpeg timelapse failed across all decode paths (hevc_cuvid, h264_cuvid, normal) and CPU fallback\n"
            detail = log4 or log3 or log2 or log
            raise EncodeError(msg + (detail or ""))

    if backend == "cpu":
        rc, log = _run_with_executor(x265_cmd)
        if rc != 0:
            raise EncodeError("ffmpeg timelapse failed on CPU libx265 (forced)\n" + (log or ""))
    else:
        _try_chain()

    if not out.exists():
        raise EncodeError("ffmpeg timelapse reported success but output missing")
    p = str(out.resolve())
    print(f"[timelapse] done  file={input_path} out={p}")
    return p


def timelapse_merge(
    segments: List[str],
    merged_out: str,
    *,
    max_frames: int = 27000,
    max_workers: int = 4,
    backend: Literal["auto", "gpu", "cpu"] = "auto",
    cpu_workers: int = 6,
    gpu_workers: int = 2,
) -> Tuple[str, int]:
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

    print(f"[tl-merge] start segments={len(segments)} out={merged_out} max_frames={max_frames}")
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

    # Add a small safety margin to avoid exceeding the cap after concat rounding
    approx_frames = int(approx_frames * 1.03)
    step = max(1, (approx_frames + max_frames - 1) // max_frames)
    print(f"[tl-merge] approx_total_frames={approx_frames} -> step={step}")

    # Timelapse each segment concurrently
    tmp_dir = Path(Path(merged_out).parent or ".") / f"tl_{get_uuid(8)}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    out_paths: List[str] = []
    for i, seg in enumerate(segments):
        out_paths.append(str(tmp_dir / f"tl_{i:03d}.mp4"))

    # Wrapper that retries CPU when GPU forced fails
    def _do_task(seg: str, out_path: str, step: int, pref: str) -> str:
        try:
            return _make_timelapse_single(seg, out_path, step, backend=pref)
        except Exception as e:
            if pref == "gpu":
                print(f"[tl-merge] gpu task failed; retry on cpu: {seg}")
                return _make_timelapse_single(seg, out_path, step, backend="cpu")
            raise

    # Two pools: GPU and CPU, with requested parallelism
    gpu_workers = max(0, int(gpu_workers))
    cpu_workers = max(0, int(cpu_workers))
    total_slots = max(1, gpu_workers + cpu_workers)

    futs = []
    ex_gpu = ThreadPoolExecutor(max_workers=gpu_workers) if gpu_workers > 0 else None
    ex_cpu = ThreadPoolExecutor(max_workers=cpu_workers) if cpu_workers > 0 else None
    try:
        for i, seg in enumerate(segments):
            out_path = out_paths[i]
            # Assign approximately gpu:cpu by slots proportion
            assign_gpu = gpu_workers > 0 and ((i % total_slots) < gpu_workers)
            if assign_gpu and ex_gpu is not None:
                futs.append(ex_gpu.submit(_do_task, seg, out_path, step, "gpu"))
            elif ex_cpu is not None:
                futs.append(ex_cpu.submit(_do_task, seg, out_path, step, "cpu"))
            else:
                # Single-threaded fallback
                _do_task(seg, out_path, step, backend)
        # Wait and surface exceptions
        for fu in as_completed(futs):
            fu.result()
    finally:
        if ex_gpu:
            ex_gpu.shutdown(wait=True)
        if ex_cpu:
            ex_cpu.shutdown(wait=True)

    # Merge timelapsed segments with safe re-encode to avoid HEVC ref errors
    merged_path = concat_videos_safe(out_paths, merged_out, backend=backend)
    print(f"[tl-merge] concatenated parts -> {merged_path}")

    # Enforce frame cap with tolerance: allow slight overruns (effort target)
    # Light-weight post check using fast estimate
    actual_est = _count_frames_precise(merged_path) or 0
    if max_frames and actual_est and actual_est > max_frames:
        tol_ratio = 1.10  # allow up to +10%
        # If near the threshold, perform a strict count to avoid unnecessary extra decimation
        actual = actual_est
        if actual_est <= int(max_frames * 1.5):
            n_strict = _count_frames_decode(merged_path)
            if n_strict:
                actual = n_strict
        if actual <= int(max_frames * tol_ratio):
            print(f"[tl-merge] post-check frames={actual} within tolerance (cap={max_frames}); skipping extra decimation")
        else:
            extra = (actual + max_frames - 1) // max_frames
            extra = max(2, int(extra))
            print(f"[tl-merge] post-check frames={actual} > cap; applying extra={extra}")
            tmp_fixed = str(Path(merged_out).with_suffix("").with_name(Path(merged_out).stem + "_fixed.mp4"))
            _make_timelapse_single(merged_path, tmp_fixed, extra, backend=backend)
            Path(merged_path).unlink(missing_ok=True)
            Path(tmp_fixed).rename(merged_path)
            step *= extra
            print(f"[tl-merge] post-fix complete -> {merged_path}, total_step={step}")

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

    print(f"[tl-merge] done -> {merged_path} step={step}")
    return merged_path, step


def _make_speedup_single(
    input_path: str,
    output_path: str,
    speed: float,
    *,
    backend: Literal["auto", "gpu", "cpu"] = "auto",
    target_fps: int = 30,
) -> str:
    """Encode a time-scaled video to reach a desired common speed factor.

    speed > 1 accelerates, < 1 slows down. Produces CFR yuv420p HEVC.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if speed <= 0:
        speed = 1.0
    gop = GOP_DEFAULT
    vf = f"setpts=(PTS-STARTPTS)/{speed},fps={target_fps}"
    # Attempt chain for faster decode
    hevc_nv = [
        "ffmpeg", "-y", "-nostdin",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-c:v", "hevc_cuvid",
        "-i", str(input_path),
        "-vf", f"hwdownload,format=nv12,{vf}",
        "-an", "-vsync", "cfr",
        "-c:v", "hevc_nvenc",
        "-preset", "p1", "-rc", "vbr", "-crf", "24",
        "-rc-lookahead", "20", "-spatial_aq", "1", "-temporal_aq", "1", "-aq-strength", "8",
        "-bf", "0", "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
        "-tune", "hq", "-pix_fmt", "yuv420p",
        str(out),
    ]
    h264_nv = [
        "ffmpeg", "-y", "-nostdin",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-c:v", "h264_cuvid",
        "-i", str(input_path),
        "-vf", f"hwdownload,format=nv12,{vf}",
        "-an", "-vsync", "cfr",
        "-c:v", "hevc_nvenc",
        "-preset", "p1", "-rc", "vbr", "-crf", "24",
        "-rc-lookahead", "20", "-spatial_aq", "1", "-temporal_aq", "1", "-aq-strength", "8",
        "-bf", "0", "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
        "-tune", "hq", "-pix_fmt", "yuv420p",
        str(out),
    ]
    nv = [
        "ffmpeg", "-y", "-nostdin",
        "-i", str(input_path),
        "-vf", vf,
        "-an", "-vsync", "cfr",
        "-c:v", "hevc_nvenc",
        "-preset", "p1", "-rc", "vbr", "-crf", "24",
        "-rc-lookahead", "20", "-spatial_aq", "1", "-temporal_aq", "1", "-aq-strength", "8",
        "-bf", "0", "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
        "-tune", "hq", "-pix_fmt", "yuv420p",
        str(out),
    ]
    x = [
        "ffmpeg", "-y", "-nostdin",
        "-i", str(input_path),
        "-vf", vf,
        "-an", "-vsync", "cfr",
        "-c:v", "libx265", "-preset", "ultrafast", "-crf", "24",
        "-bf", "0", "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
        "-pix_fmt", "yuv420p",
        str(out),
    ]
    rc, log = _run_with_executor(hevc_nv)
    if rc != 0:
        rc2, log2 = _run_with_executor(h264_nv)
        if rc2 != 0:
            rc3, log3 = _run_with_executor(nv)
            if rc3 != 0:
                rc4, log4 = _run_with_executor(x)
                if rc4 != 0 or not out.exists():
                    detail = log4 or log3 or log2 or log
                    raise EncodeError(
                        "ffmpeg speedup failed across NVDec(hevc/h264), NVEnc, and CPU fallback\n" + (detail or "")
                    )
    return str(out.resolve())


def timelapse_merge_to_duration(
    segments: List[str],
    merged_out: str,
    *,
    target_frames: int = 27000,
    backend: Literal["auto", "gpu", "cpu"] = "auto",
    cpu_workers: int = 6,
    gpu_workers: int = 2,
    target_fps: int = 30,
) -> Tuple[str, float]:
    """
    Create a timelapse whose total frames match `target_frames` at `target_fps` (± a small tolerance).

    - Computes total input duration, derives a common speed factor S = total_dur / target_duration
      where target_duration = target_frames / target_fps (but will not slow down clips when the
      source has fewer frames than requested).
    - Per-segment time-scale with CFR re-encode, then safe-concat with re-encode.
    - Returns: (merged_path, speed_factor)
    """
    if not segments:
        raise ValueError("No segments provided for timelapse_merge_to_duration")

    target_frames = max(1, int(target_frames))
    target_fps = max(1, int(target_fps))

    target_duration_sec = target_frames / float(target_fps)

    # Estimate total frames to decide whether to stretch; prefer frame count over duration
    est_total_frames = 0
    frames_unknown = False
    for s in segments:
        try:
            n = count_frames(s)
            if n is None:
                frames_unknown = True
                break
            est_total_frames += n
        except Exception:
            frames_unknown = True
            break

    # Sum durations
    total_dur = 0.0
    for s in segments:
        try:
            meta = probe_video(s)
            d = float(meta.get("durationSec") or 0.0)
            total_dur += max(0.0, d)
        except Exception:
            pass
    if total_dur <= 0.0:
        # Fallback: assume 30min to avoid divide by zero
        total_dur = 1800.0

    # Avoid stretching when total frames (or duration) are already below the target
    shorter_than_target = False
    if not frames_unknown and est_total_frames > 0 and est_total_frames < target_frames:
        shorter_than_target = True
    elif frames_unknown and total_dur < target_duration_sec:
        shorter_than_target = True

    S = 1.0 if shorter_than_target else max(0.0001, total_dur / float(target_duration_sec))

    tmp_dir = Path(Path(merged_out).parent or ".") / f"tl_{get_uuid(8)}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_paths: List[str] = []
    for i, seg in enumerate(segments):
        out_paths.append(str(tmp_dir / f"tl_{i:03d}.mp4"))

    def _task(seg: str, out_path: str, pref: str) -> str:
        try:
            return _make_speedup_single(seg, out_path, S, backend=pref, target_fps=target_fps)
        except Exception as e:
            if pref == "gpu":
                return _make_speedup_single(seg, out_path, S, backend="cpu", target_fps=target_fps)
            raise

    # Parallel encode per segment
    gpu_workers = max(0, int(gpu_workers))
    cpu_workers = max(0, int(cpu_workers))
    total_slots = max(1, gpu_workers + cpu_workers)
    futs = []
    ex_gpu = ThreadPoolExecutor(max_workers=gpu_workers) if gpu_workers > 0 else None
    ex_cpu = ThreadPoolExecutor(max_workers=cpu_workers) if cpu_workers > 0 else None
    try:
        for i, seg in enumerate(segments):
            out_path = out_paths[i]
            assign_gpu = gpu_workers > 0 and ((i % total_slots) < gpu_workers)
            if assign_gpu and ex_gpu is not None:
                futs.append(ex_gpu.submit(_task, seg, out_path, "gpu"))
            elif ex_cpu is not None:
                futs.append(ex_cpu.submit(_task, seg, out_path, "cpu"))
            else:
                _task(seg, out_path, backend)
        for fu in as_completed(futs):
            fu.result()
    finally:
        if ex_gpu:
            ex_gpu.shutdown(wait=True)
        if ex_cpu:
            ex_cpu.shutdown(wait=True)

    # Merge with safe re-encode
    merged_path = concat_videos_safe(out_paths, merged_out, backend=backend)

    # Final frame-count correction to be close to target_frames
    actual_frames = count_frames(merged_path)
    if actual_frames is not None:
        tol_frames = max(3, int(target_frames * 0.01))  # allow small drift (>=3 frames or 1%)
        # Only tighten when we exceed the requested frames; do not stretch shorter clips
        if actual_frames > target_frames + tol_frames:
            corr = max(0.0001, actual_frames / float(target_frames))
            corrected = str(Path(merged_out).with_suffix("").with_name(Path(merged_out).stem + "_corr.mp4"))
            _make_speedup_single(merged_path, corrected, corr, backend=backend, target_fps=target_fps)
            Path(merged_path).unlink(missing_ok=True)
            Path(corrected).rename(merged_path)
            S *= corr
    else:
        # Fallback to duration-based correction when frame count is unavailable
        try:
            meta = probe_video(merged_path)
            dur = float(meta.get("durationSec") or 0.0)
        except Exception:
            dur = 0.0
        if dur > target_duration_sec and abs(dur - float(target_duration_sec)) > 0.05:
            corr = max(0.0001, dur / float(target_duration_sec))
            corrected = str(Path(merged_out).with_suffix("").with_name(Path(merged_out).stem + "_corr.mp4"))
            _make_speedup_single(merged_path, corrected, corr, backend=backend, target_fps=target_fps)
            Path(merged_path).unlink(missing_ok=True)
            Path(corrected).rename(merged_path)
            S *= corr

    # Cleanup temp parts
    try:
        for p in out_paths:
            Path(p).unlink(missing_ok=True)
        tmp_dir.rmdir()
    except Exception:
        pass

    return merged_path, S
