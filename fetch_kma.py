"""기상청 API허브 단기예보 격자 자료에서 단지 격자(nx=94, ny=121)의 풍속(WSD)을 수집.

대상일 D의 01~24시 값은 D-1 11시 발표분에서 가져온다 (리드 14~37시간).
격자 배열 방향은 동네예보 조회서비스 값과 비교해서 확인한다 (verify_index).

- 기간: 2022-01-01 ~ 2025-12-31 (대상일 기준)
- 출력: external_kma_wsd.csv (time, kma_wsd)
- 수집일: 2026-07-12

API 키가 필요하다. https://apihub.kma.go.kr 에서 발급받아 환경변수로 넘긴다.

    set KMA_AUTH_KEY=발급받은키
    python fetch_kma.py [시작일 종료일]

이미 받은 시각은 건너뛰므로 호출 한도에 걸려 멈춰도 다시 실행하면 이어서 받는다.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "external_kma_wsd.csv"
KEY = os.environ.get("KMA_AUTH_KEY", "")
GRID = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-dfs_shrt_grd"
VILAGE = "https://apihub.kma.go.kr/api/typ02/openApi/VilageFcstInfoService_2.0/getVilageFcst"
NX, NY, PX, PY = 149, 253, 94, 121
UA = {"User-Agent": "Mozilla/5.0"}


def http(url, timeout=90):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def grid_tokens(tmfc, tmef, var="WSD"):
    txt = http(f"{GRID}?tmfc={tmfc}&tmef={tmef}&vars={var}&authKey={KEY}").decode("euc-kr", "replace")
    toks = [t.strip() for t in txt.replace("\n", ",").split(",")]
    vals = []
    for t in toks:
        if not t:
            continue
        try:
            vals.append(float(t))
        except ValueError:
            return None, txt[:200]
    if len(vals) != NX * NY:
        return None, f"토큰수 {len(vals)} != {NX * NY}: {txt[:120]}"
    return vals, None


def verify_index():
    # 어제 발표분을 조회서비스 값과 비교해서 격자 배열이 아래→위인지 위→아래인지 정한다
    base = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    tgt_date = datetime.now().strftime("%Y%m%d")
    url = (f"{VILAGE}?pageNo=1&numOfRows=1000&dataType=JSON&base_date={base}&base_time=1100"
           f"&nx={PX}&ny={PY}&authKey={KEY}")
    d = json.loads(http(url))
    items = d["response"]["body"]["items"]["item"]
    ref = {it["fcstTime"]: float(it["fcstValue"]) for it in items
           if it["category"] == "WSD" and it["fcstDate"] == tgt_date}
    hh = sorted(ref)[len(ref) // 2]
    vals, err = grid_tokens(f"{base}11", f"{tgt_date}{hh[:2]}")
    if vals is None:
        raise RuntimeError(f"격자 조회 실패: {err}")
    cands = {"bottom-up": (PY - 1) * NX + (PX - 1), "top-down": (NY - PY) * NX + (PX - 1)}
    print(f"대조 기준(조회서비스) {tgt_date} {hh}: WSD={ref[hh]}")
    best = None
    for name, idx in cands.items():
        diff = abs(vals[idx] - ref[hh])
        print(f"  {name} idx={idx}: {vals[idx]} (차이 {diff:.2f})")
        if best is None or diff < best[2]:
            best = (name, idx, diff)
    if best[2] > 0.51:
        raise RuntimeError("격자 인덱스 검증 실패 — 어느 방향도 일치하지 않음")
    print(f"확정: {best[0]} (idx={best[1]})")
    return best[1]


def main():
    if not KEY:
        sys.exit("환경변수 KMA_AUTH_KEY 가 필요합니다 (https://apihub.kma.go.kr 무료 발급).")
    start = sys.argv[1] if len(sys.argv) > 1 else "2022-01-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2025-12-31"  # 대상일 기준, 마지막 시각은 2026-01-01 00:00
    try:
        idx = verify_index()
    except Exception as e:
        sys.exit(f"격자 인덱스 검증 실패 ({type(e).__name__}: {str(e)[:120]}) — "
                 "KMA_AUTH_KEY 가 유효한지 확인하세요.")

    done = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8-sig").splitlines()[1:]:
            done.add(line.split(",")[0])
    else:
        OUT.write_text("time,kma_wsd\n", encoding="utf-8-sig")

    day = datetime.strptime(start, "%Y-%m-%d")
    end_day = datetime.strptime(end, "%Y-%m-%d")
    n_req = n_fail = 0
    t0 = time.time()
    f = OUT.open("a", encoding="utf-8-sig")
    try:
        while day <= end_day:
            tmfc = (day - timedelta(days=1)).strftime("%Y%m%d") + "11"
            for h in range(1, 25):
                stamp = (day + timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S") if h == 24 \
                    else day.strftime("%Y-%m-%d") + f" {h:02d}:00:00"
                if h == 24:
                    stamp = (day + timedelta(days=1)).strftime("%Y-%m-%d") + " 00:00:00"
                if stamp in done:
                    continue
                tmef = (day + timedelta(days=1)).strftime("%Y%m%d") + "00" if h == 24 \
                    else day.strftime("%Y%m%d") + f"{h:02d}"
                val = None
                for attempt in range(3):
                    try:
                        vals, err = grid_tokens(tmfc, tmef)
                        if vals is not None:
                            val = vals[idx]
                        break
                    except urllib.error.HTTPError as e:
                        body = e.read()[:200].decode("utf-8", "replace")
                        if e.code in (403, 429) or "한도" in body:
                            print(f"호출 한도/차단 (HTTP {e.code}): {body} — 중단, 재실행 시 이어서 수집")
                            return
                        time.sleep(3)
                    except Exception:
                        time.sleep(3)
                n_req += 1
                if val is None or val < -90:
                    n_fail += 1
                else:
                    f.write(f"{stamp},{val}\n")
                time.sleep(0.12)
            if day.day == 1 or (day - end_day).days == 0:
                f.flush()
                rate = n_req / max(time.time() - t0, 1)
                print(f"{day:%Y-%m-%d} 완료 (누적 {n_req}콜, 실패 {n_fail}, {rate:.1f}콜/s)", flush=True)
            day += timedelta(days=1)
    finally:
        f.close()
    print(f"수집 종료: {n_req}콜, 실패 {n_fail}")


if __name__ == "__main__":
    main()
