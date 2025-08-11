import subprocess
import json
from pathlib import Path
from typing import List, Optional
from backend_module.uuid_tools import get_uuid


class EncodeError(Exception):
    """Raised when ffmpeg encoding fails."""


def encode_to_segments(input_path: str, out_dir: Optional[str] = None) -> List[str]:
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

    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-i",
        str(src),
        "-an",
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-f",
        "segment",
        "-segment_time",
        "180",
        "-reset_timestamps",
        "1",
        out_tpl,
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise EncodeError(f"ffmpeg failed (code {proc.returncode}):\n{proc.stderr}")

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
