"""외부 예보 전처리. external_{icon,gem,ukmo,jma}_a1.csv를 만든다.

Historical Forecast 값은 최신 실행분을 이어붙인 것이라 2025년 추론에 그대로 쓸 수 없다.
그래서 구간을 나눠서 처리한다.

- 2024-03-01 이후: previous-run 값과 KMA 풍속으로 원래 값을 예측하는 LightGBM 매퍼를 만들어
  그 출력으로 바꾼다. 매퍼는 2024-03-01 ~ 2025-01-01 데이터로만 학습하고(주차 홀짝 2-fold OOF),
  2025년 구간에는 예측만 한다.
- 2024-02 이전(학습 구간): 원래 값에, 겹치는 구간에서 본 "previous-run - 원래 값" 차이를
  날짜 단위로 뽑아서 더한다. 학습과 추론 피처의 분포를 맞추려는 것.

previous_day1은 대상 시각 24시간 전 예보라서 13시가 넘는 시각은 기준 시점(전날 13시)보다
늦게 나온 값이 된다. 그런 시각은 day1을 버리고 day2(48시간 전)를 쓴다. 00시는 24시로 본다.

입력: external_{icon,gem,ukmo,jma}.csv, external_{icon,gem,ukmo,jma}_prev.csv, external_kma_wsd.csv
출력: external_{icon,gem,ukmo,jma}_a1.csv

실행: python build_external_a1.py [출력 폴더]
      기본 출력 폴더는 _a1_rebuild/ 이고, 다 만든 뒤 기존 파일과 sha256을 비교해서 보여준다.
      약 2~3분. lightgbm 버전이 다르면 결과가 조금 달라질 수 있다.
"""
import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

ROOT = Path(__file__).parent

MODELS = {
    "icon": ["wind_speed_100m", "wind_speed_10m", "wind_direction_100m",
             "wind_gusts_10m", "temperature_2m", "surface_pressure"],
    "gem": ["wind_speed_120m", "wind_speed_80m", "wind_speed_10m", "wind_gusts_10m",
            "wind_direction_120m"],
    "ukmo": ["wind_speed_10m", "wind_gusts_10m", "wind_direction_10m"],
    "jma": ["wind_speed_10m", "wind_direction_10m"],
}
SWAP_START, SWAP_END = "2024-03-01", "2025-01-01"
LGBM = dict(n_estimators=500, learning_rate=0.05, num_leaves=63, min_child_samples=20,
            colsample_bytree=0.8, subsample=0.8, subsample_freq=1, verbose=-1, n_jobs=-1)


def compliant_series(pv, var, scheme):
    # scheme A: day2만 사용
    # scheme B: 발행 블록 기준 13시 이하 시각은 day1, 나머지는 day2
    # 00시는 전날 블록의 마지막 시각이라 hour=0이 아니라 24로 봐야 한다.
    #
    # UKMO day1 돌풍에 60 m/s가 넘는 이상값이 있는데, 걸러서 NaN으로 두면 오히려 점수가 떨어져서 그대로 둔다.
    d1, d2 = pv[f"{var}_previous_day1"], pv[f"{var}_previous_day2"]
    if scheme == "A":
        return d2
    blk_hour = np.where(pv.index.hour == 0, 24, pv.index.hour)
    return d1.where(blk_hour <= 13).fillna(d2)


def make_noise_dir(scheme="B", seed=0, swap_start=SWAP_START, swap_end=SWAP_END,
                   pool_end=None, noise_before="2024-01-01", outdir=None, suffix="",
                   comp_frames=None, ws_match=0):
    # swap 구간은 previous-run 기반 값으로 바꾸고, noise_before 이전 구간에는
    # 겹침 구간의 예보 오차(delta = previous-run - 원래 값)를 같은 달 날짜 하나를 골라 더한다.
    #
    # ws_match > 0 이면 같은 달 후보 중 일평균 풍속이 가까운 날짜 ws_match개 안에서 고른다.
    # 오차 크기가 풍속과 상관이 있어서 넣어 본 옵션인데 최종본은 0(사용 안 함)이다.
    rng = np.random.default_rng(seed)
    tmp = Path(outdir) if outdir else Path(tempfile.mkdtemp(prefix=f"prev_noise_{scheme}_"))
    for name, vars_ in MODELS.items():
        sp = pd.read_csv(ROOT / f"external_{name}.csv", encoding="utf-8-sig",
                         parse_dates=["time"]).set_index("time")
        pv = pd.read_csv(ROOT / f"external_{name}_prev.csv", encoding="utf-8-sig",
                         parse_dates=["time"]).set_index("time")
        out = sp.copy()
        mask = (out.index >= swap_start) & (out.index <= swap_end)
        comp = {v: (comp_frames[name][v] if comp_frames is not None
                    else compliant_series(pv, v, scheme)).reindex(out.index) for v in vars_}
        for v in vars_:
            out.loc[mask, v] = comp[v][mask]

        # 오차 풀 (날짜 x 시각). 모든 변수에 같은 날짜를 써서 변수 간 상관을 유지한다.
        pool = (out.index >= swap_start) & (out.index <= (pool_end or swap_end))
        piv = {}
        for v in vars_:
            d = comp[v][pool] - sp.loc[pool, v]
            if "direction" in v:
                d = (d + 180) % 360 - 180
            idx = d.index
            piv[v] = pd.DataFrame({"date": idx.date, "hour": idx.hour, "d": d.to_numpy()}) \
                .pivot_table(index="date", columns="hour", values="d", dropna=False)
        pool_dates = piv[vars_[0]].index.to_numpy()
        pool_months = pd.Series([d.month for d in pool_dates])
        by_month = {m: pool_dates[(pool_months == m).to_numpy()] for m in range(1, 13)}

        daily_ws = None
        if ws_match:
            wsv = "wind_speed_10m" if "wind_speed_10m" in vars_ else vars_[0]
            s_ws = sp[wsv]
            daily_ws = s_ws.groupby(s_ws.index.date).mean()

        all_hours = out.index.hour
        all_dates = out.index.date
        tr_pos = np.flatnonzero(out.index < pd.Timestamp(noise_before))
        pos_by_date = pd.Series(tr_pos).groupby(pd.Series(all_dates[tr_pos])).groups
        vals = {v: out[v].to_numpy(copy=True) for v in vars_}
        for d, positions in pos_by_date.items():
            m = d.month
            cands = by_month.get(m)
            if cands is None or len(cands) == 0:  # 해당 월이 풀에 없으면 가장 가까운 월
                dist = sorted(range(1, 13), key=lambda k: min(abs(k - m), 12 - abs(k - m)))
                cands = next(by_month[k] for k in dist if len(by_month.get(k, [])) > 0)
            if ws_match and len(cands) > ws_match:
                tw = daily_ws.get(d, np.nan)
                cw = np.array([daily_ws.get(c, np.nan) for c in cands], dtype=float)
                ok = np.isfinite(cw)
                if np.isfinite(tw) and ok.sum() >= ws_match:
                    near = cands[ok][np.argsort(np.abs(cw[ok] - tw))[:ws_match]]
                    cands = near
            src = cands[rng.integers(len(cands))]
            pos = np.asarray(positions)
            hrs = all_hours[pos]
            for v in vars_:
                base = vals[v][pos]
                if np.isnan(base).all():
                    continue  # 데이터가 없는 구간은 NaN 그대로
                delta = piv[v].loc[src].reindex(hrs).to_numpy()
                if "direction" in v:
                    vals[v][pos] = (base + delta) % 360
                elif v.startswith("wind_"):
                    vals[v][pos] = np.maximum(base + delta, 0)
                else:
                    vals[v][pos] = base + delta
        for v in vars_:
            out[v] = vals[v]
        out.reset_index().to_csv(tmp / f"external_{name}{suffix}.csv", index=False, encoding="utf-8-sig")
    return tmp


def build_inputs(idx, lags=True, ecmwf=False):
    X = pd.DataFrame(index=idx)
    # compliant_series와 같은 규칙: day1은 발행 블록 기준 13시 이하 시각만 쓴다.
    # comp_frames 경로는 compliant_series를 거치지 않으므로 여기서 한 번 더 막는다.
    blk_hour = np.where(idx.hour == 0, 24, idx.hour)
    d1_ok = blk_hour <= 13
    splice = {}
    for name, vars_ in MODELS.items():
        sp = pd.read_csv(ROOT / f"external_{name}.csv", encoding="utf-8-sig",
                         parse_dates=["time"]).set_index("time").reindex(idx)
        pv = pd.read_csv(ROOT / f"external_{name}_prev.csv", encoding="utf-8-sig",
                         parse_dates=["time"]).set_index("time").reindex(idx)
        for c in pv.columns:
            s = pv[c].where(d1_ok) if c.endswith("_previous_day1") else pv[c]
            if "direction" in c:
                rad = np.deg2rad(s)
                X[f"{name}_{c}_sin"] = np.sin(rad)
                X[f"{name}_{c}_cos"] = np.cos(rad)
            else:
                X[f"{name}_{c}"] = s
        splice[name] = sp
    km = pd.read_csv(ROOT / "external_kma_wsd.csv", encoding="utf-8-sig",
                     parse_dates=["time"]).set_index("time").reindex(idx)
    X["kma_wsd"] = km["kma_wsd"]
    if ecmwf:
        # ECMWF previous-run을 입력에 추가해 본 실험 (사용 안 함, 데이터 파일도 없음)
        ec = pd.read_csv(ROOT / "external_ecmwf_prev.csv", encoding="utf-8-sig",
                         parse_dates=["time"]).set_index("time").reindex(idx)
        for c in ec.columns:
            X[f"ecmwf_{c}"] = ec[c].where(d1_ok) if c.endswith("_previous_day1") else ec[c]
    if lags:
        # 시차/롤링 피처 실험. 점수가 떨어져서 최종본은 lags=False로 호출한다.
        keys = ["icon_wind_speed_100m_previous_day1", "icon_wind_speed_100m_previous_day2",
                "gem_wind_speed_120m_previous_day1", "jma_wind_speed_10m_previous_day1",
                "kma_wsd"]
        for k in keys:
            s = X[k]
            for lag in (-3, -2, -1, 1, 2, 3):
                X[f"{k}_lag{lag}"] = s.shift(lag)
            X[f"{k}_roll6_mean"] = s.rolling(6, center=True, min_periods=1).mean()
            X[f"{k}_roll6_std"] = s.rolling(6, center=True, min_periods=1).std()
    X["hod_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    X["hod_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    X["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365)
    X["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365)
    return X, splice


def build_mapped_frames(lags=True, mapper_seeds=(0,), ecmwf=False):
    idx = pd.date_range("2024-01-01", "2026-01-01 23:00", freq="h")
    X, splice = build_inputs(idx, lags=lags, ecmwf=ecmwf)

    ov = (idx >= SWAP_START) & (idx <= SWAP_END)
    wpar = idx.isocalendar().week.to_numpy() % 2

    def fit_target(y):
        acc = np.zeros(len(idx))
        ok = y.notna().to_numpy() & ov
        for rs in mapper_seeds:
            out = np.full(len(idx), np.nan)
            for p in (0, 1):
                m = LGBMRegressor(random_state=rs, **LGBM).fit(X[ok & (wpar == p)], y[ok & (wpar == p)])
                te = ov & (wpar != p)
                out[te] = m.predict(X[te])
            mfull = LGBMRegressor(random_state=rs, **LGBM).fit(X[ok], y[ok])
            out[~ov] = mfull.predict(X[~ov])
            acc += out
        return acc / len(mapper_seeds)

    # UKMO 10m 풍속/풍향은 매퍼 출력보다 previous-run 값이 더 나아서 그대로 쓴다
    raw_keep = {("ukmo", "wind_speed_10m"), ("ukmo", "wind_direction_10m")}
    frames = {}
    for name, vars_ in MODELS.items():
        f = pd.DataFrame(index=idx)
        if any((name, v) in raw_keep for v in vars_):
            pv = pd.read_csv(ROOT / f"external_{name}_prev.csv", encoding="utf-8-sig",
                             parse_dates=["time"]).set_index("time").reindex(idx)
        for v in vars_:
            if (name, v) in raw_keep:
                f[v] = compliant_series(pv, v, "B")
                continue
            y = splice[name][v]
            if "direction" in v:
                rad = np.deg2rad(y)
                s = fit_target(pd.Series(np.sin(rad), index=idx))
                c = fit_target(pd.Series(np.cos(rad), index=idx))
                f[v] = (np.rad2deg(np.arctan2(s, c)) + 360) % 360
            elif v.startswith("wind_"):
                f[v] = np.maximum(fit_target(y), 0)
            else:
                f[v] = fit_target(y)
        frames[name] = f
    return frames


def main():
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "_a1_rebuild"
    outdir.mkdir(parents=True, exist_ok=True)
    frames = build_mapped_frames(lags=False, mapper_seeds=(0, 1, 2))
    make_noise_dir("B", seed=0, swap_start="2024-03-01", swap_end="2026-01-01",
                   pool_end="2024-12-30", noise_before="2024-03-01",
                   outdir=outdir, suffix="_a1", comp_frames=frames)
    for name in MODELS:
        p = outdir / f"external_{name}_a1.csv"
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        ref = ROOT / f"external_{name}_a1.csv"
        note = ""
        if ref.exists() and ref.resolve() != p.resolve():
            rh = hashlib.sha256(ref.read_bytes()).hexdigest()
            note = "  기존 파일과 일치" if rh == h else f"  기존 파일과 다름 ({rh[:16]})"
        print(f"{p.name}  sha256 {h[:16]}{note}")


if __name__ == "__main__":
    main()
