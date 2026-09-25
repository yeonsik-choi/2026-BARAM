# 제3회 풍력발전량 예측 AI 경진대회

태백 가덕산 풍력단지의 2025년 시간별 발전량을 전날 나온 기상 예보로 예측하는 대회 코드입니다.
대회에서 제공한 LDAPS, GFS 예보에 외부 수치예보 몇 가지를 더해서 LightGBM으로 학습했습니다.

- 팀명: 연식2
- Public 점수: 0.67430
- 예측 대상: `kpx_group_1` ~ `kpx_group_3` (설비용량 21,600 / 21,600 / 21,000 kWh)
- 평가: NMAE와 FiCR(오차가 설비용량의 6%, 8% 이내인 시간의 발전량 가중 비율)을 반씩 반영하고, 실측 발전량이 설비용량 10% 이상인 시간만 채점

## 구성

```
.
├── submit_train.ipynb        학습, model_artifacts/ 저장
├── submit_inference.ipynb    추론, open/final_submission.csv 생성
├── build_external_a1.py      외부 예보 전처리 (external_*_a1.csv 생성)
├── fetch_*.py                외부 데이터 수집
├── external_*.csv            수집한 외부 데이터
├── model_artifacts/          학습된 모델 (187MB)
└── requirements.txt
```

## 실행

Windows 10, Python 3.10.9에서 작업했습니다.

```bash
pip install -r requirements.txt jupyter

# 학습 (15~25분, model_artifacts/를 덮어씀)
jupyter nbconvert --to notebook --execute --inplace submit_train.ipynb

# 추론 (10~15초)
jupyter nbconvert --to notebook --execute --inplace submit_inference.ipynb
```

- `model_artifacts/`가 있으면 추론만 돌려도 됩니다.
- 노트북은 `open/`이 있는 폴더에서 실행해야 합니다.
- 시드를 전부 고정해 두어서 같은 환경이면 같은 결과가 나옵니다. lightgbm은 버전에 따라 결과가 조금 달라질 수 있어서 4.6.0을 권장합니다.

## 방법

### 피처

- LDAPS, GFS 격자 예보에서 격자별로 풍속을 계산한 뒤 전체 평균, 단지 인근 격자 평균, 격자 간 표준편차를 구함
- 허브 높이 풍속의 세제곱, 100m 풍향, 850hPa와 지상 기온차, 하루 풍속 평균/최대/최소
- lag, lead, diff, rolling 피처 (lead와 중심 rolling은 같은 발행 블록 안에서만 계산)
- 외부 예보(ICON, GEM, UKMO, JMA, MSM)의 풍속, 풍향과 모델 간 풍속 차이

### 모델

- LightGBM, L1 loss
- 그룹별 모델과 3개 그룹을 합친 통합 모델(`group_id` 피처 추가)을 각각 시드 10개로 학습하고 0.3 : 0.7로 섞음
- 발전량 비율을 샘플 가중치로 사용 (FiCR이 발전량 가중이라서)
- SCADA로 터빈 정지나 고장이 의심되는 시간을 찾아 학습에서 제외
  - 그룹은 발전 중인데 특정 터빈 출력만 거의 0인 시간
  - 나셀 풍속은 충분한데 출력이 파워커브 기대치의 35%가 안 되는 시간

### 보정

- 5-fold OOF 예측으로 isotonic 회귀 보정
- 보정한 OOF를 8개 분위 구간으로 나누고, 구간마다 점수가 가장 좋아지는 오프셋(±2.5%cap 이내)을 찾아 절반만 적용

### 후처리

`submit_inference.ipynb`의 `post_transform`에서 순서대로 적용합니다. 고정 상수만 쓰고 학습은 하지 않습니다.

| 순서 | 내용 |
|---|---|
| 1 | 그룹별 레벨 배율 |
| 2 | g3 겨울 구간(1/19 ~ 2/10) 하향 |
| 3 | 발행 블록 안에서 3시간 롤링 중앙값 |
| 4 | 저출력 구간(4 ~ 14%cap)에 4.5%cap 가산 |
| 5 | 정격 근처 예측을 고원값까지 올림 |
| 6 ~ 7 | g3 1 ~ 11월 상한 80%cap |
| 8 ~ 10 | 발행 블록 시각(13시 이하 / 이후)별 배율 |
| 11 | 하한 10%cap |
| 12 | 그룹별 달력 구간 배율 (`CAL_WINDOWS`) |

- g3 상한은 학습 데이터 마지막 달(2024-12)에 5기 중 1기(`unison_wtg01`)가 계속 멈춰 있던 것을 보고 넣었습니다.
- 하한은 채점 방식에서 나온 값입니다. 실측이 설비용량 10% 이상인 시간만 채점하므로 예측을 10%보다 낮게 둘 이유가 없습니다.
- 나머지 상수는 학습 기간 홀드아웃으로 방향을 잡고 Public 리더보드 점수를 보면서 크기를 정했습니다.

## 외부 데이터

| 파일 | 출처 | 기간 | 수집 코드 |
|---|---|---|---|
| `external_{icon,gem,ukmo,jma}.csv` | Open-Meteo Historical Forecast API | 2022 ~ 2025 | `fetch_icon.py`, `fetch_ext3.py` |
| `external_{icon,gem,ukmo,jma}_prev.csv` | Open-Meteo Previous Runs API | 2024 ~ 2025 | `fetch_prev_runs.py`, `fetch_prev_refresh.py` |
| `external_kma_wsd.csv` | 기상청 API허브 단기예보 격자 | 2022 ~ 2025 | `fetch_kma.py` |
| `external_msm.csv` | 교토대 RISH 아카이브 (JMA MSM) | 2022 ~ 2025 | `fetch_msm.py` |
| `external_{icon,gem,ukmo,jma}_a1.csv` | 위 파일로 만든 전처리 결과 | 2022 ~ 2025 | `build_external_a1.py` |

모델이 직접 읽는 파일은 `_a1` 4개와 `external_msm.csv`이고, 나머지는 `_a1`을 만들 때 쓰는 원본입니다.

- 수집 지점은 단지 중심(37.28N, 128.963E)이고, 수집 시기는 2026년 7월입니다.
- Open-Meteo는 과거 값이 나중에 바뀌기도 해서 다시 받으면 값이 조금 다를 수 있습니다. 그래서 받은 CSV를 그대로 올려 두었습니다.
- `fetch_kma.py`는 기상청 API허브 인증키가 필요합니다 (`KMA_AUTH_KEY` 환경변수).
- `fetch_msm.py`는 `xarray`, `cfgrib`이 추가로 필요합니다.
- `_a1` 파일은 `python build_external_a1.py`로 다시 만들 수 있습니다 (2~3분, 결과는 `_a1_rebuild/`에 저장).

### 예보 시점 맞추기

대회 데이터의 `data_available_kst_dtm`에 맞춰 대상일 전날 13시를 예측 시점으로 보고, 외부 예보도 그 전에 나온 것만 쓰도록 했습니다.

- Open-Meteo Historical Forecast는 최신 실행분을 이어붙인 값이라 그대로 쓰면 더 늦게 나온 예보가 섞입니다. 그래서 2024-03 이후 구간은 previous-run 값으로 원래 값을 예측하는 LightGBM 매퍼를 만들어 그 출력으로 바꿨습니다. 매퍼는 2024년 3~12월 데이터로만 학습했습니다.
- previous_day1(24시간 전 예보)은 13시 이하 시각에만 쓰고, 그 뒤 시각은 previous_day2(48시간 전)로 채웠습니다.
- 학습 구간(2024-02 이전)에는 겹치는 기간에서 본 예보 오차를 날짜 단위로 뽑아 더해서 학습과 추론의 입력 분포를 맞췄습니다.
- 기상청 단기예보는 전날 11시 발표분, MSM은 전날 00UTC(09시 KST) 실행분만 씁니다.
- 모델 안의 lead, 중심 rolling 피처도 같은 발행 블록을 넘지 않게 했습니다.