import shutil
from pathlib import Path

import pytest

import backend_module.encoder as enc


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe are required",
)

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"


@pytest.fixture(scope="session")
def sample_video() -> Path:
    src = ASSETS_DIR / "demo.mp4"
    if not src.exists():
        pytest.skip("demo asset is missing")
    return src


def test_parse_fps() -> None:
    assert enc._parse_fps("30") == 30.0
    assert enc._parse_fps("30000/1001") == pytest.approx(29.97, rel=1e-3)
    assert enc._parse_fps("bad") is None


def test_probe_video_and_frame_helpers(sample_video: Path) -> None:
    meta = enc.probe_video(str(sample_video))
    assert meta["width"] == 1280
    assert meta["height"] == 720
    assert meta["codec_name"] in ("h264", "hevc")
    assert meta["durationSec"] and meta["durationSec"] > 0
    assert meta["avg_frame_rate"]

    precise = enc.count_frames(str(sample_video))
    assert precise is not None and precise > 0

    strict = enc.count_frames_strict(str(sample_video))
    assert strict is not None and strict >= precise

    approx = enc._approx_total_frames_by_probe(str(sample_video))
    assert approx is not None and approx > 0

    decoded = enc._count_frames_decode(str(sample_video))
    assert decoded is not None and decoded > 0


def test_encode_to_segments_cpu(sample_video: Path, tmp_path: Path) -> None:
    outputs = enc.encode_to_segments(str(sample_video), out_dir=str(tmp_path / "segs"), backend="cpu")
    assert outputs
    for out in outputs:
        assert Path(out).exists()


def test_encode_to_segments_auto_fallback(sample_video: Path, tmp_path: Path) -> None:
    outputs = enc.encode_to_segments(str(sample_video), out_dir=str(tmp_path / "segs_auto"), backend="auto")
    assert outputs
    for out in outputs:
        assert Path(out).exists()


def test_encode_to_hls_cpu(sample_video: Path, tmp_path: Path) -> None:
    result = enc.encode_to_hls(str(sample_video), out_dir=str(tmp_path / "hls"), backend="cpu", segment_time=6)
    assert Path(result["playlist"]).exists()
    assert result["segments"]
    for s in result["segments"]:
        assert Path(s).exists()


def test_transcode_video_and_thumbnail(sample_video: Path, tmp_path: Path) -> None:
    transcoded = enc.transcode_video(str(sample_video), output_path=str(tmp_path / "transcoded.mp4"))
    assert Path(transcoded).exists()

    thumb = enc.create_thumbnail(transcoded, str(tmp_path / "thumb.jpg"), timestamp_sec=1.0, width=320)
    assert Path(thumb).exists()


def test_create_thumbnail_errors(sample_video: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        enc.create_thumbnail("missing.mp4", str(tmp_path / "thumb.jpg"))

    # Force error by pointing to non-video (use empty file)
    dummy = tmp_path / "empty.bin"
    dummy.write_bytes(b"")
    with pytest.raises(enc.EncodeError):
        enc.create_thumbnail(str(dummy), str(tmp_path / "thumb2.jpg"))


def test_concat_videos_and_safe(sample_video: Path, tmp_path: Path) -> None:
    copy1 = tmp_path / "c1.mp4"
    copy2 = tmp_path / "c2.mp4"
    shutil.copy(sample_video, copy1)
    shutil.copy(sample_video, copy2)

    out_concat = enc.concat_videos([str(copy1), str(copy2)], str(tmp_path / "concat.mp4"))
    assert Path(out_concat).exists()

    out_safe = enc.concat_videos_safe([str(copy1), str(copy2)], str(tmp_path / "concat_safe.mp4"), backend="cpu")
    assert Path(out_safe).exists()

    with pytest.raises(ValueError):
        enc.concat_videos([], str(tmp_path / "none.mp4"))


def test_make_speedup_single(sample_video: Path, tmp_path: Path) -> None:
    out = enc._make_speedup_single(str(sample_video), str(tmp_path / "speed.mp4"), speed=2.0, backend="cpu")
    assert Path(out).exists()


def test_timelapse_single(sample_video: Path, tmp_path: Path) -> None:
    out = enc.timelapse_single(str(sample_video), str(tmp_path / "tl.mp4"), step=8, backend="cpu")
    assert Path(out).exists()


def test_timelapse_merge(sample_video: Path, tmp_path: Path) -> None:
    seg1 = tmp_path / "seg1.mp4"
    seg2 = tmp_path / "seg2.mp4"
    shutil.copy(sample_video, seg1)
    shutil.copy(sample_video, seg2)

    merged, step = enc.timelapse_merge(
        [str(seg1), str(seg2)],
        str(tmp_path / "tl_merged.mp4"),
        max_frames=500,
        backend="cpu",
        cpu_workers=1,
        gpu_workers=0,
    )
    assert Path(merged).exists()
    assert step >= 1


def test_timelapse_merge_to_duration(sample_video: Path, tmp_path: Path) -> None:
    merged_out = tmp_path / "timelapse.mp4"
    target_frames = 60
    target_fps = 30

    merged_path, speed = enc.timelapse_merge_to_duration(
        [str(sample_video)],
        str(merged_out),
        target_frames=target_frames,
        target_fps=target_fps,
        backend="cpu",
        cpu_workers=1,
        gpu_workers=0,
    )

    assert Path(merged_path).exists()
    assert speed > 0

    frames = enc.count_frames_strict(merged_path)
    assert frames is not None
    assert frames == pytest.approx(target_frames, rel=0.2, abs=5)


def test_timelapse_merge_to_duration_errors() -> None:
    with pytest.raises(ValueError):
        enc.timelapse_merge_to_duration([], "out.mp4")

    with pytest.raises(FileNotFoundError):
        enc.encode_to_segments("missing.mp4")
