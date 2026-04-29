from __future__ import annotations

from .cli_infer_samrai import main, sam2_track_to_parquet

__all__ = ["main", "sam2_track_to_parquet"]


if __name__ == "__main__":
    raise SystemExit(main())
