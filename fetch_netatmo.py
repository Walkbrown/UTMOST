#!/usr/bin/env python3
"""Fetch Netatmo public weather-station data for a bounding box and write Parquet.

Designed to run unattended in CI:
  * credentials come from the environment, never from source
  * the OAuth2 refresh token is rotated on every call and written back to a
    GitHub Actions secret so the next run can still authenticate
  * one Parquet file is written per run; de-duplication happens downstream at
    consolidation time rather than by rewriting a growing daily file
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd
import requests

AUTH_URL = "https://api.netatmo.com/oauth2/token"
PUBLIC_DATA_URL = "https://api.netatmo.com/api/getpublicdata"
GITHUB_API = "https://api.github.com"

# lat_sw, lon_sw, lat_ne, lon_ne
DEFAULT_BBOX: Tuple[float, float, float, float] = (51.90992, 4.83181, 52.27075, 5.41334)

BBox = Tuple[float, float, float, float]


# --------------------------------------------------------------------------- #
# OAuth2
# --------------------------------------------------------------------------- #

def refresh_tokens(client_id: str, client_secret: str, refresh_token: str) -> Tuple[str, str]:
    """Exchange a refresh token for a fresh access token.

    Netatmo rotates refresh tokens: the response carries a new one and the old
    one is burned. Returns (access_token, refresh_token) -- always persist the
    second element before the next run.
    """
    response = requests.post(
        AUTH_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Token refresh failed [{response.status_code}]: {response.text[:400]}"
        )
    payload = response.json()
    if "access_token" not in payload:
        raise RuntimeError(f"No access_token in refresh response: {payload}")
    return payload["access_token"], payload.get("refresh_token", refresh_token)


def update_github_secret(repo: str, name: str, value: str, gh_token: str) -> None:
    """Encrypt and store a value as an Actions secret on `repo` (owner/name)."""
    from nacl import encoding, public  # imported late so --dry-run works without it

    headers = {
        "Authorization": f"Bearer {gh_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    base = f"{GITHUB_API}/repos/{repo}/actions/secrets"

    key_response = requests.get(f"{base}/public-key", headers=headers, timeout=30)
    if key_response.status_code >= 400:
        raise RuntimeError(
            f"Could not read repo public key [{key_response.status_code}]: "
            f"{key_response.text[:300]}"
        )
    key_data = key_response.json()

    sealed_box = public.SealedBox(
        public.PublicKey(key_data["key"].encode(), encoding.Base64Encoder)
    )
    encrypted = base64.b64encode(sealed_box.encrypt(value.encode())).decode()

    put_response = requests.put(
        f"{base}/{name}",
        headers=headers,
        json={"encrypted_value": encrypted, "key_id": key_data["key_id"]},
        timeout=30,
    )
    if put_response.status_code not in (201, 204):
        raise RuntimeError(
            f"Could not write secret {name} [{put_response.status_code}]: "
            f"{put_response.text[:300]}"
        )


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #

def split_bbox(bbox: BBox, nx: int, ny: int) -> Iterator[BBox]:
    """Split a bounding box into an nx-by-ny grid of sub-boxes.

    getpublicdata truncates the station list for large areas, so tiling recovers
    stations that a single wide query silently drops.
    """
    lat_sw, lon_sw, lat_ne, lon_ne = bbox
    dlat = (lat_ne - lat_sw) / ny
    dlon = (lon_ne - lon_sw) / nx
    for i in range(nx):
        for j in range(ny):
            yield (
                lat_sw + j * dlat,
                lon_sw + i * dlon,
                lat_sw + (j + 1) * dlat,
                lon_sw + (i + 1) * dlon,
            )


def fetch_tile(token: str, bbox: BBox, retries: int = 3) -> List[Dict[str, Any]]:
    """Fetch every public station within one bounding box."""
    lat_sw, lon_sw, lat_ne, lon_ne = bbox
    params = {
        "lat_ne": lat_ne,
        "lon_ne": lon_ne,
        "lat_sw": lat_sw,
        "lon_sw": lon_sw,
        "filter": "false",
    }
    headers = {"Authorization": f"Bearer {token}"}

    for attempt in range(retries):
        response = requests.get(PUBLIC_DATA_URL, params=params, headers=headers, timeout=60)
        if response.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
            time.sleep(2 ** attempt * 5)
            continue
        if response.status_code >= 400:
            raise RuntimeError(
                f"getpublicdata failed [{response.status_code}]: {response.text[:400]}"
            )
        payload = response.json()
        if "body" not in payload:
            raise RuntimeError(f"Unexpected API response: {str(payload)[:400]}")
        return payload["body"]

    raise RuntimeError("getpublicdata exhausted retries")


def fetch_all(token: str, bbox: BBox, nx: int, ny: int, pause: float = 1.0) -> List[Dict[str, Any]]:
    """Fetch across a tiled bounding box, de-duplicating stations by id."""
    stations: Dict[str, Dict[str, Any]] = {}
    tiles = list(split_bbox(bbox, nx, ny))
    for index, tile in enumerate(tiles):
        for station in fetch_tile(token, tile):
            station_id = station.get("_id")
            if station_id:
                stations[station_id] = station
        if index < len(tiles) - 1:
            time.sleep(pause)  # stay clear of the 50-requests-per-10s app limit
    return list(stations.values())


# --------------------------------------------------------------------------- #
# Flattening
# --------------------------------------------------------------------------- #

# Wind and rain modules do not use the "res" envelope; they expose flat keys.
WIND_FIELDS = ("wind_strength", "wind_angle", "gust_strength", "gust_angle")
RAIN_FIELDS = ("rain_60min", "rain_24h", "rain_live")


def _station_context(station: Dict[str, Any]) -> Dict[str, Any]:
    place = station.get("place") or {}
    location = place.get("location") or [None, None]
    return {
        "station_id": station.get("_id"),
        "station_type": station.get("station_type"),
        "station_altitude": place.get("altitude"),
        "station_city": place.get("city"),
        "station_country": place.get("country"),
        "station_timezone": place.get("timezone"),
        "station_location_lat": location[1],
        "station_location_lon": location[0],
    }


def iter_rows(stations: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """Flatten station + module payloads into long-format rows.

    Handles all three module shapes getpublicdata returns: the `res`/`type`
    envelope used by temperature, humidity and pressure modules, and the flat
    `wind_*` and `rain_*` shapes. The original loader only handled the first,
    so wind and rain were being dropped silently.
    """
    for station in stations:
        context = _station_context(station)
        measures = station.get("measures") or {}

        for module_id, module in measures.items():
            if not isinstance(module, dict):
                continue

            # --- res / type envelope (temperature, humidity, pressure) ---
            if "res" in module:
                module_type = module.get("type")
                data_types = module_type if isinstance(module_type, list) else []
                for measure_time, values in (module.get("res") or {}).items():
                    timestamp = int(measure_time)
                    if not isinstance(values, list):
                        values = [values]
                    for index, value in enumerate(values):
                        yield {
                            **context,
                            "module_id": module_id,
                            "module_type": ",".join(data_types) if data_types else None,
                            "data_type": data_types[index] if index < len(data_types) else None,
                            "value": value,
                            "timestamp_unix": timestamp,
                        }
                continue

            # --- flat wind module ---
            if "wind_timeutc" in module:
                timestamp = module.get("wind_timeutc")
                if timestamp is None:
                    continue
                for field in WIND_FIELDS:
                    if field in module:
                        yield {
                            **context,
                            "module_id": module_id,
                            "module_type": "wind",
                            "data_type": field,
                            "value": module[field],
                            "timestamp_unix": int(timestamp),
                        }
                continue

            # --- flat rain module ---
            if "rain_timeutc" in module:
                timestamp = module.get("rain_timeutc")
                if timestamp is None:
                    continue
                for field in RAIN_FIELDS:
                    if field in module:
                        yield {
                            **context,
                            "module_id": module_id,
                            "module_type": "rain",
                            "data_type": field,
                            "value": module[field],
                            "timestamp_unix": int(timestamp),
                        }


def rows_to_frame(rows: Iterable[Dict[str, Any]], fetched_at: dt.datetime) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return frame
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_unix"], unit="s", utc=True)
    frame["fetched_at_utc"] = fetched_at
    frame = frame.drop_duplicates(
        subset=["station_id", "module_id", "data_type", "timestamp_unix"]
    )
    return frame.sort_values(["station_id", "module_id", "timestamp_unix"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_grid(value: str) -> Tuple[int, int]:
    try:
        nx_text, ny_text = value.lower().split("x")
        nx, ny = int(nx_text), int(ny_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("grid must look like '3x3'") from exc
    if nx < 1 or ny < 1:
        raise argparse.ArgumentTypeError("grid dimensions must be >= 1")
    return nx, ny


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data", type=Path,
                        help="root directory for Parquet output")
    parser.add_argument("--prefix", default="netatmo_public",
                        help="output filename prefix")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("LAT_SW", "LON_SW", "LAT_NE", "LON_NE"),
                        default=list(DEFAULT_BBOX), help="bounding box corners")
    parser.add_argument("--tiles", type=parse_grid, default=(1, 1),
                        help="split the bbox into an NxM grid, e.g. 3x3")
    parser.add_argument("--no-token-writeback", action="store_true",
                        help="skip writing the rotated refresh token back to GitHub")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    client_id = os.environ.get("CLIENT_ID")
    client_secret = os.environ.get("CLIENT_SECRET")
    refresh_token = os.environ.get("REFRESH_TOKEN")
    missing = [n for n, v in [("CLIENT_ID", client_id), ("CLIENT_SECRET", client_secret),
                              ("REFRESH_TOKEN", refresh_token)] if not v]
    if missing:
        print(f"Missing environment variables: {', '.join(missing)}", file=sys.stderr)
        return 2

    fetched_at = dt.datetime.now(dt.timezone.utc)

    try:
        access_token, new_refresh_token = refresh_tokens(client_id, client_secret, refresh_token)
        print("Token refreshed.")

        # Persist the rotated token *before* doing anything that might fail.
        if not args.no_token_writeback and new_refresh_token != refresh_token:
            repo = os.environ.get("GITHUB_REPOSITORY")
            gh_token = os.environ.get("GH_PAT")
            if repo and gh_token:
                update_github_secret(repo, "REFRESH_TOKEN", new_refresh_token, gh_token)
                print("Rotated REFRESH_TOKEN written back to repository secrets.")
            else:
                print("WARNING: refresh token rotated but GH_PAT/GITHUB_REPOSITORY are "
                      "unset -- the next run will fail with invalid_grant.", file=sys.stderr)

        nx, ny = args.tiles
        bbox: BBox = tuple(args.bbox)  # type: ignore[assignment]
        stations = fetch_all(access_token, bbox, nx, ny)
        frame = rows_to_frame(iter_rows(stations), fetched_at)

        if frame.empty:
            print(f"{len(stations)} stations returned but no measurements; nothing written.")
            return 0

        day_dir = args.output_dir / fetched_at.strftime("%Y") / fetched_at.strftime("%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        path = day_dir / f"{args.prefix}_{fetched_at.strftime('%Y%m%dT%H%M%SZ')}.parquet"
        frame.to_parquet(path, index=False, compression="zstd")

        size_kb = path.stat().st_size / 1024
        print(f"{len(stations)} stations -> {len(frame)} rows -> {path} ({size_kb:.0f} kB)")
        print("Variables: " + ", ".join(sorted(frame["data_type"].dropna().unique())))
        return 0

    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
