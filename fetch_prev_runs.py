"""Open-Meteo Previous Runs API에서 previous_day1 / previous_day2 값을 수집.

previous_day1은 대상 시각 24시간 전에 나온 예보, previous_day2는 48시간 전 예보다.
최신 실행분이 섞인 Historical Forecast 대신 추론 기간(2025)에 쓰려고 받았다.

- 모델: ICON, GEM, UKMO, JMA (변수는 fetch_icon.py, fetch_ext3.py와 같음)
- 기간: 2024-01-01 ~ 2026-01-01 (2024년은 검증과 매퍼 학습에 사용)
- 출력: external_{icon,gem,ukmo,jma}_prev.csv, 컬럼명은 <변수>_previous_day1/2
- 수집일: 2026-07-12

day1을 그대로 쓰면 오후 시각에는 기준 시점(전날 13시)보다 늦게 나온 예보가 들어간다.
build_external_a1.py에서 13시 이후 시각의 day1은 버리고 day2로 채운다.

실행: python fetch_prev_runs.py
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
    hourly = [f"{v}_previous_day{n}" for v in base_vars for n in (1, 2)]
    frames = []
    for start, end in CHUNKS:
        frames.append(fetch(model, start, end, hourly))
        time.sleep(2)
    out = (pd.concat(frames, ignore_index=True)
           .drop_duplicates("time").sort_values("time"))
    out.to_csv(ROOT / f"external_{name}_prev.csv", index=False, encoding="utf-8-sig")
    for n in (1, 2):
        col = f"{base_vars[0]}_previous_day{n}"
        valid = out[col].notna()
        first = out.loc[valid, "time"].min() if valid.any() else None
        v25 = out.loc[(out["time"] >= "2025-01-01") & valid, "time"].count()
        print(f"external_{name}_prev.csv day{n}: 유효 {valid.sum()}/{len(out)} "
              f"(첫 유효 {first}, 2025+ 유효 {v25})")
