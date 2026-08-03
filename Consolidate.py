#!/usr/bin/env python3
"""Collapse per-run Netatmo Parquet files into one de-duplicated file per day.

Run nightly. Any day directory older than --min-age-days is merged, written to
<output-dir>/netatmo_public_YYYY-MM-DD.parquet, and its per-run files removed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

DEDUPE_KEYS = ["station_id", "module_id", "data_type", "timestamp_unix"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("daily"))
    parser.add_argument("--min-age-days", type=int, default=1,
                        help="only consolidate days at least this old")
    parser.add_argument("--keep-run-files", action="store_true")
    args = parser.parse_args(argv)

    cutoff = dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=args.min_age_days)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    consolidated = 0

    for day_dir in sorted(p for p in args.input_dir.glob("*/*") if p.is_dir()):
        try:
            day = dt.date.fromisoformat(f"{day_dir.parent.name}-{day_dir.name}")
        except ValueError:
            print(f"Skipping unrecognised directory {day_dir}", file=sys.stderr)
            continue
        if day > cutoff:
            continue

        run_files = sorted(day_dir.glob("*.parquet"))
        if not run_files:
            continue

        target = args.output_dir / f"netatmo_public_{day.isoformat()}.parquet"
        frames = [pd.read_parquet(f) for f in run_files]
        if target.exists():
            frames.insert(0, pd.read_parquet(target))

        merged = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=DEDUPE_KEYS, keep="first")
            .sort_values(["station_id", "module_id", "timestamp_unix"])
            .reset_index(drop=True)
        )
        merged.to_parquet(target, index=False, compression="zstd")

        raw_rows = sum(len(f) for f in frames)
        print(f"{day}: {len(run_files)} runs, {raw_rows} rows -> {len(merged)} unique "
              f"({target.stat().st_size / 1024:.0f} kB)")

        if not args.keep_run_files:
            for f in run_files:
                f.unlink()
            try:
                day_dir.rmdir()
                if not any(day_dir.parent.iterdir()):
                    day_dir.parent.rmdir()
            except OSError:
                pass

        consolidated += 1

    print(f"Consolidated {consolidated} day(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
