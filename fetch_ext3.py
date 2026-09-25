"""Open-Meteo Historical Forecast API로 GEM, UKMO, JMA 예보를 받고 ICON 앞 구간을 채운다.

1) ICON   2022-11-24 ~ 2023-02-28 -> external_icon.csv에 합침
2) GEM    2022-11-24 ~ -> external_gem.csv   wind_speed_120m/80m/10m, wind_gusts_10m, wind_direction_120m
3) UKMO   2022-01-01 ~ -> external_ukmo.csv  wind_speed_10m, wind_gusts_10m, wind_direction_10m
4) JMA    2022-01-01 ~ -> external_jma.csv   wind_speed_10m, wind_direction_10m

지점과 주의사항은 fetch_icon.py와 같다. 수집일 2026-07-08.

실행: python fetch_ext3.py  (fetch_icon.py 실행 후)
"""
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
LAT, LON = 37.28, 128.963


def fetch(model, start, end, hourly):
    url = ("https://historical-forecast-api.open-meteo.com/v1/forecast"
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


# 1) ICON 앞 구간
ICON_VARS = ["wind_speed_100m", "wind_speed_10m", "wind_direction_100m",
             "wind_gusts_10m", "temperature_2m", "surface_pressure"]
bf = fetch("icon_seamless", "2022-11-24", "2023-02-28", ICON_VARS)
old = pd.read_csv(ROOT / "external_icon.csv", encoding="utf-8-sig", parse_dates=["time"])
merged = (pd.concat([bf, old], ignore_index=True)
          .drop_duplicates("time", keep="last").sort_values("time"))
merged.to_csv(ROOT / "external_icon.csv", index=False, encoding="utf-8-sig")
print(f"ICON 백필: +{len(merged) - len(old)}행 → {len(merged)}행 "
      f"({merged['time'].min()} ~ {merged['time'].max()})")

# 2~4) GEM, UKMO, JMA
JOBS = [
    ("gem", "gem_seamless",
     ["wind_speed_120m", "wind_speed_80m", "wind_speed_10m", "wind_gusts_10m",
      "wind_direction_120m"],
     [("2022-11-24", "2023-12-31"), ("2024-01-01", "2024-12-31"),
      ("2025-01-01", "2025-12-31"), ("2026-01-01", "2026-01-01")]),
    ("ukmo", "ukmo_seamless",
     ["wind_speed_10m", "wind_gusts_10m", "wind_direction_10m"],
     [("2022-01-01", "2022-12-31"), ("2023-01-01", "2023-12-31"),
      ("2024-01-01", "2024-12-31"), ("2025-01-01", "2025-12-31"),
      ("2026-01-01", "2026-01-01")]),
    ("jma", "jma_seamless",
     ["wind_speed_10m", "wind_direction_10m"],
     [("2022-01-01", "2022-12-31"), ("2023-01-01", "2023-12-31"),
      ("2024-01-01", "2024-12-31"), ("2025-01-01", "2025-12-31"),
      ("2026-01-01", "2026-01-01")]),
]
for name, model, hourly, chunks in JOBS:
    frames = []
    for start, end in chunks:
        frames.append(fetch(model, start, end, hourly))
        time.sleep(1.5)
    out = (pd.concat(frames, ignore_index=True)
           .drop_duplicates("time").sort_values("time"))
    out.to_csv(ROOT / f"external_{name}.csv", index=False, encoding="utf-8-sig")
    main_var = hourly[0]
    valid = out[main_var].notna()
    first_valid = out.loc[valid, "time"].min() if valid.any() else None
    print(f"external_{name}.csv: {len(out)}행, {main_var} 유효 {valid.sum()} (첫 유효 {first_valid})")
