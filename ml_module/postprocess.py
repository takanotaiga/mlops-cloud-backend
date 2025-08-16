from __future__ import annotations

from typing import List
from pathlib import Path

from backend_module.encoder import transcode_video


def postprocess_video_paths(paths: List[str], work_dir: str) -> List[str]:
    """Postprocess inference outputs.

    - For now, transcodes each input video to a normalized MP4 using
      NVENC if available with CPU fallback, referencing existing encoder.
    - Returns a list of output paths in the same order as inputs.

    If a transcode fails, returns the original path for that item.
    """
    out_paths: List[str] = []
    base = Path(work_dir)
    base.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(paths, start=1):
        try:
            dst = base / f"result_{i:03d}_enc.mp4"
            final_p = transcode_video(p, output_path=str(dst))
            out_paths.append(final_p)
        except Exception:
            # Fallback to original path on failure
            out_paths.append(p)
    return out_paths

