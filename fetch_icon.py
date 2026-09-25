"""Open-Meteo Historical Forecast API에서 DWD ICON 예보를 받아 external_icon.csv로 저장.

- 지점: 단지 중심 (37.28N, 128.963E), 1시간 간격
- 변수: wind_speed_100m, wind_speed_10m, wind_direction_100m, wind_gusts_10m,
        temperature_2m, surface_pressure
- 기간: 2023-03-01 ~ 2026-01-01 (2022-11-24 ~ 2023-02-28은 fetch_ext3.py에서 추가)
- 수집일: 2026-07-08

이 API는 최신 실행분을 이어붙인 값이라 대상 시각 직전에 나온 예보가 섞여 있다.
그래서 2024-03 이후 구간은 build_external_a1.py에서 previous-run 기반 값으로 바꿔서 쓴다.

실행: python fetch_icon.py
"""
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

LAT, LON = 37.28, 128.963
VARS = ["wind_speed_100m", "wind_speed_10m", "wind_direction_100m",
        "wind_gusts_10m", "temperature_2m", "surface_pressure"]
OUT = Path(__file__).parent / "external_icon.csv"

frames = []
for start, end in [("2023-03-01", "2023-12-31"), ("2024-01-01", "2024-12-31"),
                   ("2025-01-01", "2025-12-31"), ("2026-01-01", "2026-01-01")]:
    url = ("https://historical-forecast-api.open-meteo.com/v1/forecast"
           f"?latitude={LAT}&longitude={LON}&start_date={start}&end_date={end}"
           f"&hourly={','.join(VARS)}&models=icon_seamless"
           "&wind_speed_unit=ms&timezone=Asia%2FSeoul")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                d = json.loads(r.read())
            break
        except Exception as e:
            print(f"{start} 재시도 {attempt + 1}: {e}")
            time.sleep(5)
    else:
        raise RuntimeError(f"fetch 실패: icon_seamless {start}")
    df = pd.DataFrame(d["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    frames.append(df)
    print(f"{start} ~ {end}: {len(df)}행, ws100 유효 {df['wind_speed_100m'].notna().sum()}")
    time.sleep(1)

out = pd.concat(frames, ignore_index=True).drop_duplicates("time").sort_values("time")
out.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"저장: {OUT}  ({len(out)}행, {out['time'].min()} ~ {out['time'].max()})")
