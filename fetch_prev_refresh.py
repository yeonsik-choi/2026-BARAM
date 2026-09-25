"""external_*_prev.csv의 빈 칸을 Previous Runs API에서 다시 받아 채운다.

처음 받았을 때(2026-07-12) UKMO 쪽 결측이 많아서 2026-07-15에 한 번 더 받았다.
기존 값은 그대로 두고 NaN인 칸만 채운다. 원본은 external_*_prev_backup0712.csv로 남긴다.
UKMO 2025년 1~8월 중 약 88일은 API에도 값이 없어서 NaN으로 남아 있다.

실행: python fetch_prev_refresh.py  (fetch_prev_runs.py 실행 후)
"""
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
LAT, LON = 37.28, 128.963

CHUNKS = [("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31"),
          ("2025-01-01", "2025-06-30"), ("2025-07-01", "2025-12-31"),
          ("2026-01-01", "2026-01-01")]

JOBS = [
    ("icon", "icon_seamless",
     ["wind_speed_100m", "wind_speed_10m", "wind_direction_100m",
      "wind_gusts_10m", "temperature_2m", "surface_pressure"]),
    ("gem", "gem_seamless",
     ["wind_speed_120m", "wind_speed_80m", "wind_speed_10m", "wind_gusts_10m",
      "wind_direction_120m"]),
    ("ukmo", "ukmo_seamless",
     ["wind_speed_10m", "wind_gusts_10m", "wind_direction_10m"]),
    ("jma", "jma_seamless",
     ["wind_speed_10m", "wind_direction_10m"]),
]


def fetch(model, start, end, hourly):
    url = ("https://previous-runs-api.open-meteo.com/v1/forecast"
           f"?latitude={LAT}&longitude={LON}&start_date={start}&end_date={end}"
           f"&hourly={','.join(hourly)}&models={model}"
           "&wind_speed_unit=ms&timezone=Asia%2FSeoul")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                d = json.loads(r.read())
            df = pd.DataFrame(d["hourly"])
            df["time"] = pd.to_datetime(df["time"])
            return df
        except Exception as e:
            print(f"  재시도 {attempt + 1}: {str(e)[:80]}")
            time.sleep(8)
    raise RuntimeError(f"fetch 실패: {model} {start}")


for name, model, base_vars in JOBS:
    path = ROOT / f"external_{name}_prev.csv"
    old = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["time"]).set_index("time")

    hourly = [f"{v}_previous_day{n}" for v in base_vars for n in (1, 2)]
    frames = []
    for start, end in CHUNKS:
        frames.append(fetch(model, start, end, hourly))
        time.sleep(2)
    new = (pd.concat(frames, ignore_index=True)
           .drop_duplicates("time").sort_values("time").set_index("time"))

    merged = old.combine_first(new.reindex(old.index.union(new.index)))
    merged = merged.sort_index()

    backup = ROOT / f"external_{name}_prev_backup0712.csv"
    if not backup.exists():
        old.reset_index().to_csv(backup, index=False, encoding="utf-8-sig")
    merged.reset_index().rename(columns={"index": "time"}).to_csv(
        path, index=False, encoding="utf-8-sig")

    print(f"\n=== external_{name}_prev.csv 병합 결과 ===")
    for c in old.columns:
        before = old[c].notna().sum()
        after = merged[c].notna().sum()
        m25 = merged[merged.index >= "2025-01-01"]
        na25 = m25[c].isna().sum()
        print(f"  {c}: {before} -> {after} (+{after - before}), 2025+ 잔여결측 {na25}")
