"""교토대 RISH 아카이브에서 JMA MSM 지상풍(10m)을 받아 external_msm.csv로 저장.

대상일 D(01~24시)에는 D-1 00UTC(09시 KST) 실행의 FH16~39를 쓴다.
아카이브 파일의 Last-Modified가 02:15 UTC(11:15 KST) 전후라서 전날 13시 전에 공개된 자료다.
단지 중심(37.282N, 128.962E) 주변 격자의 u10/v10을 평균한 뒤 풍속, 풍향으로 바꿔 저장한다.

- 기간: 2022-01-01 ~ 2026-01-02
- 수집일: 2026-07-20 ~ 21
- 추가 패키지: xarray, cfgrib

실행: PYTHONIOENCODING=utf-8 python fetch_msm.py [시작일] [종료일] [출력 csv]

이미 받은 날짜는 건너뛰므로 중간에 멈춰도 다시 실행하면 된다. GRIB 파일은 추출 후 바로 지운다.
여러 개를 동시에 돌릴 때는 기간과 출력 파일을 나눠서 실행하고 나중에 합친다.
"""
import gc
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / (sys.argv[3] if len(sys.argv) > 3 else "external_msm.csv")
TMP = ROOT / f"_msm_tmp{sys.argv[3] if len(sys.argv) > 3 else ''}"
TMP.mkdir(exist_ok=True)
LAT, LON = 37.282, 128.962
BASE = "http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original"
FILES = ["FH16-33", "FH34-39"]


def fetch(url, dst, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r, open(dst, "wb") as f:
                shutil.copyfileobj(r, f)
            return True
        except Exception as e:
            if i == tries - 1:
                print(f"  FAIL {url}: {e}", flush=True)
                return False
            time.sleep(15 * (i + 1))  # 서버 제한이 있어서 간격을 넉넉히


def extract_child(path):
    # 별도 프로세스에서 GRIB을 읽어 stdout으로 넘긴다 (ecCodes 메모리 누수 때문)
    import xarray as xr
    out = {}
    for short in ("10u", "10v"):
        ds = xr.open_dataset(path, engine="cfgrib",
                             backend_kwargs={"indexpath": "",
                                             "filter_by_keys": {"shortName": short}})
        v = list(ds.data_vars)[0]
        sel = ds[v].sel(latitude=slice(LAT + 0.11, LAT - 0.11),
                        longitude=slice(LON - 0.14, LON + 0.14))
        out[short] = sel.mean(dim=("latitude", "longitude")).to_series().to_numpy()
        vt = pd.to_datetime(ds["valid_time"].to_series().values)
        ds.close()
    for t, u, v in zip(vt, out["10u"], out["10v"]):
        print(f"{t.isoformat()},{u},{v}")


def extract(path):
    r = subprocess.run([sys.executable, __file__, "--extract", str(path)],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "extract child failed")
    vt, us, vs = [], [], []
    for line in r.stdout.strip().splitlines():
        t, u, v = line.split(",")
        vt.append(pd.Timestamp(t))
        us.append(float(u))
        vs.append(float(v))
    return pd.DatetimeIndex(vt), np.array(us), np.array(vs)


def run_one(init):
    rows = []
    for f in FILES:
        name = f"Z__C_RJTD_{init:%Y%m%d}000000_MSM_GPV_Rjp_Lsurf_{f}_grib2.bin"
        url = f"{BASE}/{init:%Y/%m/%d}/{name}"
        dst = TMP / name
        if not fetch(url, dst):
            continue
        try:
            vt, u, v = extract(dst)
            ws = np.sqrt(u ** 2 + v ** 2)
            wd = (np.rad2deg(np.arctan2(-u, -v)) + 360) % 360  # 바람이 불어오는 방향
            t_kst = vt + pd.Timedelta(hours=9)
            rows.append(pd.DataFrame({"time": t_kst, "msm_ws10": ws, "msm_wd10": wd}))
        except Exception as e:
            print(f"  EXTRACT FAIL {name}: {e}", flush=True)
        finally:
            if dst.exists():
                os.remove(dst)
        time.sleep(2)
    gc.collect()
    if not rows:
        return None
    df = pd.concat(rows, ignore_index=True)
    return df if len(df) == 24 else None  # 파일 하나만 받은 날은 버리고 다음 실행 때 다시 받는다


def main():
    start = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp("2021-12-31")
    end = pd.Timestamp(sys.argv[2]) if len(sys.argv) > 2 else pd.Timestamp("2025-12-31")
    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT, encoding="utf-8-sig", parse_dates=["time"])
        day = (prev["time"] - pd.Timedelta(hours=1)).dt.normalize() - pd.Timedelta(days=1)
        cnt = day.value_counts()
        done = set(cnt[cnt >= 24].index)  # 24시간이 다 있는 날만 완료로 본다
    t0 = time.time()
    n = 0
    for init in pd.date_range(start, end, freq="D"):
        if init in done:
            continue
        df = run_one(init)
        if df is not None:
            hdr = not OUT.exists()
            df.to_csv(OUT, mode="a", header=hdr, index=False, encoding="utf-8-sig")
        n += 1
        if n % 25 == 0:
            el = time.time() - t0
            print(f"{init:%Y-%m-%d} 완료 {n}런, {el/60:.0f}분 경과 ({el/n:.0f}s/런)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--extract":
        extract_child(sys.argv[2])
    else:
        main()
