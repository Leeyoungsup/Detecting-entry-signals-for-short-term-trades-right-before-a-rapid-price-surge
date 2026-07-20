# AGENT.md

## 1. 프로젝트 목표

이 프로젝트의 목적은 미국 주식 1분봉 데이터를 이용해 **단타 매수 기회 판단 AI 모델**을 개발하는 것이다.

현재 단계에서는 다음만 구현한다.

1. `*_enriched.csv` 데이터 적재 및 검증
2. 수집 universe와 종목 선정 편향 감사
3. **Toss 차트에서도 동일하게 재현 가능한 가격 기반 컬럼만 선별**
4. 최근 60개 1분봉 기반 입력 시퀀스 생성
5. Dual-path 기반 확정 라벨과 거래비용 반영 순수익 라벨 생성
6. 학습·검증·테스트 데이터셋 생성
7. Gradient Boosting baseline 학습 및 순차 백테스트
8. 진입가·수수료·슬리피지 민감도 분석
9. baseline 검증 후 OHLC 조건부 합성틱 Monte Carlo 보조 라벨 실험
10. GRU·TCN 시퀀스 모델 학습 및 비교
11. 모델과 전처리 artifact 저장

`enriched.csv`를 읽더라도 모델 입력은 차트에서 직접 확인 가능한 가격 정보와
그 가격으로부터 계산 가능한 기술지표로 제한한다.

현재 단계에서 구현하지 않는다.

- 실시간 주문
- 실제 매수·매도 API 연동
- 주문 체결 관리
- 포지션 관리
- 실시간 익절·손절
- AI 매도 모델
- 계좌 위험관리
- 자동매매 운영 시스템

---

# 2. 핵심 모델 정의

## 2.1 입력

각 샘플은 동일 종목의 연속된 최근 60개 enriched 1분봉으로 구성한다.

```text
입력 시점: t
입력 범위: t-59, ..., t
입력 형태: [60, feature_count]
```

`t`는 완전히 확정된 1분봉이다.

모델은 `t`분까지 관측 가능한 정보만 입력으로 사용해야 한다.

## 2.2 모델 목표

단순히 다음 1분봉이 상승하는지를 예측하지 않는다.

모델은 다음을 예측한다.

```text
현재 시점에 매수 주문을 시도했을 때,
향후 제한된 시간 안에 손절보다 익절이 먼저 발생하고
거래 수수료 차감 후 순수익이 발생할 가능성
```

기본 primary horizon은 10분으로 한다.

이 프로젝트는 단타 급등 직전 신호 탐지가 목적이므로 라벨 horizon은 10분으로 고정한다.
TP 도달률을 높이기 위해 20분·30분으로 보유시간을 늘리지 않는다.

## 2.3 기본 출력

최소 출력:

```text
P(TP first within 10 minutes)
expected_net_return_10m
strong_buy_probability
```

권장 멀티태스크 출력:

```text
p_tp_first_10m
expected_net_return_10m
entry_fill_probability
```

---

# 3. 절대 원칙

## 3.1 미래정보 누수 금지

판단 시점이 `t`라면 모델 입력에는 `t`까지의 데이터만 포함한다.

금지:

- `t+1` 이후 OHLCV를 입력 특징으로 사용
- 미래 봉으로 계산한 지표를 현재 입력에 병합
- 라벨 생성용 합성틱 통계를 입력에 포함
- 전체 데이터셋으로 스케일러를 fit
- 랜덤 row split
- 동일 시간 구간의 인접 샘플을 train과 validation에 섞기

허용:

- `t`까지의 최근 60개 확정봉
- `t`분까지 집계된 quote 및 trade 특징
- `t+1` 이후 데이터는 라벨 생성에만 사용

## 3.2 합성틱은 라벨 생성에만 사용

OHLC 조건부 합성틱은 미래 1분봉 내부의 가격 순서를 추정하기 위한 도구다.

합성틱으로 생성한 다음 값은 target 또는 sample weight로만 사용한다.

```text
p_entry_fill
p_tp_first
p_sl_first
p_timeout
expected_net_return
expected_mfe
expected_mae
label_confidence
```

합성된 미래 틱 가격은 모델 입력으로 사용하지 않는다.

## 3.3 순수익 기준

모든 라벨은 거래비용 차감 후 계산한다.

```text
net_pnl
= sell_notional
- buy_notional
- buy_commission
- sell_commission
- SEC fee
- TAF
```

Gross return만으로 positive label을 만들지 않는다.


## 3.4 데이터 공급자 독립성

과거 데이터는 Alpaca API에서 생성되었고 실사용 플랫폼은 Toss API를 사용한다.

따라서 API 공급자, 거래소 통합 방식, 체결 분류 방식에 따라 값이 달라질 수 있는
데이터는 모델 입력에서 전부 제외한다.

### 모델 입력에서 반드시 제외할 컬럼

```text
volume
trade_count
notional_usd
volume_ma20
relative_volume 관련 전 컬럼
relative_notional 관련 전 컬럼
avg_trade_size
alpaca_vwap

quote_count
first_bid
first_ask
last_bid
last_ask
avg_bid
avg_ask
min_bid
max_bid
min_ask
max_ask
avg_spread_pct
min_spread_pct
max_spread_pct
avg_bid_size
avg_ask_size
bid_ask_imbalance
bid_share
signed_book_imbalance
quote_valid
호가 변화 및 잔량 관련 전 파생 컬럼

trade_tick_count
trade_volume_from_ticks
aggressive_buy_volume
aggressive_sell_volume
unknown_trade_volume
trade_notional_from_ticks
trade_strength
sell_pressure
unknown_trade_ratio
known_trade_ratio
directional_buy_share
directional_sell_share
net_aggressive_volume_ratio
aggressive_buy_ratio_total
aggressive_sell_ratio_total
effective_trade_strength
체결 방향 및 체결 흐름 관련 전 파생 컬럼
```

위 컬럼은 데이터 감사와 원본 품질 확인에는 사용할 수 있으나,
모델의 `X` 입력이나 스케일러 대상에는 포함하지 않는다.

### 모델 입력에 허용되는 데이터

```text
open
high
low
close
```

그리고 위 네 가격만으로 계산 가능한 다음 정보:

```text
수익률
캔들 몸통
윗꼬리·아랫꼬리
고저폭
종가 위치
가격 이동평균
가격 EMA
가격 Bollinger Band
가격 MACD
가격 기반 변동성
최근 고점·저점과의 거리
시간대 정보
```

`session_vwap`도 거래량 가중값이므로 입력에서 제외한다.

핵심 원칙:

```text
Toss 실시간 시스템에서 같은 60개 OHLC만으로 재계산할 수 없는 값은
학습 입력에 넣지 않는다.
```

---

# 4. 입력 데이터

## 4.1 데이터 위치

데이터 루트는 프로젝트 루트 기준 다음 상대 경로다.

```text
../../data/stock_data
```

실제 원본 데이터는 날짜별 세션 디렉터리 아래에 저장되어 있다.

```text
../../data/stock_data/
└── raw/
    └── session_YYYY-MM-DD/
        ├── *_enriched.csv
        ├── *_bars.csv
        └── collection_manifest.json
```

학습 대상 파일 탐색 패턴:

```text
../../data/stock_data/raw/session_*/*_enriched.csv
```

경로는 현재 실행 디렉터리가 아니라 **프로젝트 루트 기준으로 해석**해야 한다.
코드에서는 프로젝트 루트를 먼저 결정한 뒤 절대 경로로 변환하여 사용한다.

`../../data/stock_data` 아래의 파일은 원본 데이터이므로 읽기 전용으로 취급한다.
정제 데이터, 캐시, 라벨, 데이터셋 분할 결과 및 모델 artifact를 이 경로에
덮어쓰거나 새로 저장하지 않는다.

기본 학습 파일은 `*_enriched.csv`다.

예시:

```text
lcid_2026-07-14_1700-0800_kst_bfaf1ad5_enriched.csv
```

`*_bars.csv`는 enriched 파일 검증용으로만 사용한다.

## 4.2 예상 컬럼

### 식별 및 시간

```text
symbol
timestamp_kst
timestamp_utc
```

### OHLCV

```text
open
high
low
close
volume
trade_count
alpaca_vwap
notional_usd
```

### 기술지표

```text
volume_ma20
ma5
ma20
ma60
ma120
ema9
ema20
bb_mid20
bb_upper20
bb_lower20
bb_position_0_1
session_vwap
macd_12_26
macd_signal_9
macd_histogram
candle_color
return_pct
range_pct
```

### Quote 집계

```text
quote_count
first_bid
first_ask
last_bid
last_ask
avg_bid
avg_ask
min_bid
max_bid
min_ask
max_ask
avg_spread_pct
min_spread_pct
max_spread_pct
avg_bid_size
avg_ask_size
bid_ask_imbalance
```

### Trade tick 집계

```text
trade_tick_count
trade_volume_from_ticks
aggressive_buy_volume
aggressive_sell_volume
unknown_trade_volume
trade_notional_from_ticks
trade_strength
sell_pressure
unknown_trade_ratio
```

Codex는 실제 CSV를 먼저 읽고 컬럼 존재 여부와 dtype을 검증해야 한다.

---

# 5. 데이터 검증

## 5.1 정렬

```python
sort_values(["symbol", "timestamp_utc"])
```

동일 `symbol + timestamp_utc` 중복은 허용하지 않는다.

## 5.2 OHLC 무결성

각 행에서 검사한다.

```text
high >= max(open, close)
low <= min(open, close)
high >= low
open > 0
high > 0
low > 0
close > 0
volume >= 0
trade_count >= 0
```

위반 행은 제거하고 보고서에 기록한다.

## 5.3 시간 연속성

각 종목 내 인접 timestamp 차이를 계산한다.

```text
delta_minutes
is_consecutive_minute
data_gap
gap_minutes
```

초기 학습에서는 60봉 시퀀스 내부에 1분 초과 gap이 하나라도 있으면 해당 샘플을 제외한다.

무거래 분인지 수집 누락인지 구분되지 않은 상태에서 임의 forward fill을 하지 않는다.

## 5.4 Quote 검증

파생값:

```python
mid_last = (last_bid + last_ask) / 2
spread_abs_last = last_ask - last_bid
spread_pct_last = spread_abs_last / mid_last
```

유효 조건:

```text
quote_count >= 1
last_bid > 0
last_ask > 0
last_ask >= last_bid
mid_last > 0
```

다음 컬럼을 생성한다.

```text
quote_valid
spread_pct_last
```

quote 이상치는 무조건 제거하기보다 `quote_valid=0`으로 표시하고 관련 feature를 masking한다.

## 5.5 호가 불균형 재계산

현재 `bid_ask_imbalance`는 signed imbalance가 아닐 가능성이 있으므로 원본을 보존하고 새 값을 계산한다.

```python
denom = avg_bid_size + avg_ask_size

bid_share = avg_bid_size / denom

signed_book_imbalance = (
    avg_bid_size - avg_ask_size
) / denom
```

분모가 0이면 NaN이다.

원본 컬럼은 다음 이름으로 복사한다.

```text
source_bid_ask_imbalance
```

## 5.6 Unknown trade 보정

다음 파생값을 생성한다.

```python
tick_volume = trade_volume_from_ticks

known_volume = (
    aggressive_buy_volume
    + aggressive_sell_volume
)

known_trade_ratio = known_volume / tick_volume

directional_buy_share = (
    aggressive_buy_volume / known_volume
)

directional_sell_share = (
    aggressive_sell_volume / known_volume
)

net_aggressive_volume_ratio = (
    aggressive_buy_volume
    - aggressive_sell_volume
) / tick_volume

aggressive_buy_ratio_total = (
    aggressive_buy_volume / tick_volume
)

aggressive_sell_ratio_total = (
    aggressive_sell_volume / tick_volume
)
```

0으로 나누는 경우 NaN으로 처리한다.

`trade_strength`는 단독으로 사용하지 않는다.

반드시 `known_trade_ratio`와 함께 사용하거나 다음처럼 보정한다.

```python
effective_trade_strength = (
    normalized_trade_strength
    * known_trade_ratio
)
```

---

# 6. 모델 입력 특징

모델 입력은 **OHLC 가격 및 가격만으로 계산 가능한 특징**으로 제한한다.

Alpaca와 Toss 사이에서 달라질 수 있는 거래량, 거래대금, VWAP, 체결,
호가 및 잔량 관련 값은 사용하지 않는다.

## 6.1 원본 가격 특징

입력에 사용할 수 있는 원본값:

```text
open
high
low
close
```

다만 종목별 절대 가격 차이를 줄이기 위해 raw OHLC만 넣기보다
시점별 기준가격으로 정규화한 값을 함께 사용한다.

```python
open_rel = open / close - 1
high_rel = high / close - 1
low_rel = low / close - 1
close_rel = 0.0
```

또는 각 60봉 시퀀스의 첫 종가나 마지막 종가를 기준으로 정규화할 수 있다.
실험별 정규화 방식을 config와 feature version에 기록한다.

## 6.2 캔들 구조 특징

```python
log_return_1 = log(close / close.shift(1))
log_return_2 = log(close / close.shift(2))
log_return_3 = log(close / close.shift(3))
log_return_5 = log(close / close.shift(5))
log_return_10 = log(close / close.shift(10))
log_return_20 = log(close / close.shift(20))

body_return = (close - open) / open
range_return = (high - low) / open

upper_wick = (
    high - maximum(open, close)
) / open

lower_wick = (
    minimum(open, close) - low
) / open

close_location = (
    close - low
) / (high - low)
```

`high == low`이면 `close_location=0.5`.

추가:

```text
gap_from_prev_close
body_to_range_ratio
upper_wick_to_range
lower_wick_to_range
is_bullish
is_bearish
```

## 6.3 가격 추세 특징

모든 이동평균과 EMA는 OHLC의 `close`만으로 다시 계산한다.
CSV에 이미 존재하는 값이 있더라도 동일한 파이프라인에서 재계산하는 것을 원칙으로 한다.

```text
SMA 3, 5, 10, 20, 30, 60
EMA 3, 5, 9, 12, 20, 26
```

상대 특징:

```python
close_to_sma5 = close / sma5 - 1
close_to_sma10 = close / sma10 - 1
close_to_sma20 = close / sma20 - 1
close_to_sma30 = close / sma30 - 1
close_to_sma60 = close / sma60 - 1

close_to_ema5 = close / ema5 - 1
close_to_ema9 = close / ema9 - 1
close_to_ema12 = close / ema12 - 1
close_to_ema20 = close / ema20 - 1
close_to_ema26 = close / ema26 - 1

sma5_to_sma20 = sma5 / sma20 - 1
sma10_to_sma30 = sma10 / sma30 - 1
ema9_to_ema20 = ema9 / ema20 - 1
ema12_to_ema26 = ema12 / ema26 - 1
```

기울기:

```text
sma5_slope_3
sma20_slope_5
ema9_slope_3
ema20_slope_5
```

## 6.4 가격 변동성 특징

거래량을 사용하지 않고 OHLC만으로 계산한다.

```text
rolling_std_return_3
rolling_std_return_5
rolling_std_return_10
rolling_std_return_20
rolling_std_return_60
```

True Range:

```python
true_range = maximum(
    high - low,
    abs(high - prev_close),
    abs(low - prev_close),
)

atr_5 = rolling_mean(true_range, 5)
atr_10 = rolling_mean(true_range, 10)
atr_14 = rolling_mean(true_range, 14)
atr_20 = rolling_mean(true_range, 20)

atr_pct_5 = atr_5 / close
atr_pct_10 = atr_10 / close
atr_pct_14 = atr_14 / close
atr_pct_20 = atr_20 / close
```

Parkinson volatility 등 OHLC 기반 변동성 추정치도 보조 실험으로 허용한다.

## 6.5 Bollinger Band

가격만으로 직접 재계산한다.

```python
bb_mid20 = rolling_mean(close, 20)
bb_std20 = rolling_std(close, 20)
bb_upper20 = bb_mid20 + 2 * bb_std20
bb_lower20 = bb_mid20 - 2 * bb_std20

bb_width = (
    bb_upper20 - bb_lower20
) / bb_mid20

bb_position = (
    close - bb_lower20
) / (bb_upper20 - bb_lower20)
```

## 6.6 MACD 및 모멘텀

가격만으로 재계산한다.

```python
macd = ema12 - ema26
macd_signal = ema(macd, 9)
macd_hist = macd - macd_signal

macd_scaled = macd / close
macd_signal_scaled = macd_signal / close
macd_hist_scaled = macd_hist / close
```

추가 가격 모멘텀:

```text
roc_3
roc_5
roc_10
roc_20
rsi_6
rsi_14
stochastic_k_14
stochastic_d_3
```

RSI와 stochastic은 가격만 사용하므로 허용한다.

## 6.7 가격 위치 특징

```python
rolling_high_5 = rolling_max(high, 5)
rolling_high_10 = rolling_max(high, 10)
rolling_high_20 = rolling_max(high, 20)
rolling_high_60 = rolling_max(high, 60)

rolling_low_5 = rolling_min(low, 5)
rolling_low_10 = rolling_min(low, 10)
rolling_low_20 = rolling_min(low, 20)
rolling_low_60 = rolling_min(low, 60)

distance_from_high_5 = close / rolling_high_5 - 1
distance_from_high_10 = close / rolling_high_10 - 1
distance_from_high_20 = close / rolling_high_20 - 1
distance_from_high_60 = close / rolling_high_60 - 1

distance_from_low_5 = close / rolling_low_5 - 1
distance_from_low_10 = close / rolling_low_10 - 1
distance_from_low_20 = close / rolling_low_20 - 1
distance_from_low_60 = close / rolling_low_60 - 1
```

추가:

```text
range_position_5
range_position_10
range_position_20
range_position_60
breakout_high_5
breakout_high_20
breakdown_low_5
breakdown_low_20
```

## 6.8 시간 특징

US Eastern Time 기준으로 생성한다.

```text
is_premarket
is_regular
is_afterhours
minutes_from_regular_open
minutes_to_regular_close
is_opening_5m
is_opening_30m
is_power_hour
minute_of_day_sin
minute_of_day_cos
day_of_week_sin
day_of_week_cos
```

DST는 `zoneinfo.ZoneInfo("America/New_York")`로 처리한다.

## 6.9 최종 입력 금지 검증

feature pipeline 완료 후 다음 assertion을 수행한다.

```python
FORBIDDEN_PATTERNS = [
    "volume",
    "notional",
    "trade_count",
    "vwap",
    "quote",
    "bid",
    "ask",
    "spread",
    "imbalance",
    "aggressive",
    "sell_pressure",
    "trade_strength",
    "tick_count",
]

for feature in model_features:
    assert not any(
        pattern in feature.lower()
        for pattern in FORBIDDEN_PATTERNS
    )
```

단, `is_premarket` 등의 문자열에 `market`이 포함되는 것은 허용한다.
금지 패턴은 실제 feature naming convention에 맞춰 테스트에서 관리한다.

# 7. 결측값 처리

## 7.1 기술지표 warm-up

MA60, MA120 등은 초반 NaN이 존재할 수 있다.

초기 baseline에서는 다음 중 하나를 선택한다.

### 권장 V1

60개 입력 전체에서 안정적으로 존재하는 feature만 사용한다.

긴 warm-up이 필요한 `ma120` 관련 feature는 우선 제외할 수 있다.

### V2

더 긴 과거 데이터를 확보한 뒤 MA60·MA120을 포함한다.

원본 NaN을 무조건 0으로 바꾸지 않는다.

## 7.2 스케일링

스케일러는 train split에만 fit한다.

권장:

```text
RobustScaler 또는 StandardScaler
```

스케일러, feature 순서, dtype을 모델과 함께 저장한다.

---

# 8. 입력 시퀀스 생성

각 샘플은 정확히 60개 연속봉이어야 한다.

```python
X.shape == [60, num_features]
```

시퀀스 제외 조건:

- 종목 변경
- 1분 초과 gap
- 필수 OHLC 결측
- 필수 feature가 기준 이상 결측
- 미래 라벨 구간 부족
- timestamp 역전

metadata:

```text
symbol
input_start_timestamp
input_end_timestamp
label_horizon
entry_price
label_version
feature_version
```

---

# 9. 가상 진입가격

## 9.1 실사용과 학습의 구분

실사용 주문 정책은 다음과 같다.

```text
현재 Toss best ask - one tick
```

하지만 학습용 과거 데이터의 Alpaca quote는 Toss와 동일하지 않으므로
라벨 생성에 Alpaca `last_ask`, `last_bid`, spread를 사용하지 않는다.

## 9.2 학습용 진입가 proxy

판단 시점 `t`의 확정 종가를 현재 시장가격 proxy로 사용한다.

```python
reference_price = close_t
buy_limit = reference_price - tick_size
```

이는 실제 Toss `best ask - one tick`을 정확히 재현하는 것이 아니라,
차트 가격만으로 일관되게 생성 가능한 보수적 proxy다.

다른 진입가 실험:

```text
A. close_t - one tick
B. close_t
C. next_open
```

Primary 설정은 A로 한다.

`next_open`은 미래 가격이므로 모델 입력에는 절대 넣지 않고
라벨링용 체결 proxy 실험에만 사용할 수 있다.

```text
entry_price_source = close_minus_tick
```

## 9.3 Tick size

Tick size 계산은 별도 함수로 구현한다.

초기 기본:

```yaml
tick_size:
  price_below_1: 0.0001
  price_at_or_above_1: 0.01
```

이는 학습용 체결 proxy의 가격 양자화 기준이며 모델 입력 feature가 아니다.
실제 Toss 주문 가능 단위가 다르면 배포 전 해당 정책으로 교체한다.

## 9.4 체결 가정

주문 유효시간은 첫 미래 1분봉으로 제한한다.

합성틱 경로가 다음 가격 이하로 내려가면 체결로 본다.

```python
fill_threshold = buy_limit - tick_size
```

즉 지정가를 단순 터치한 경우는 미체결로 처리하고,
한 틱 이상 관통한 경우만 체결로 인정한다.

```text
synthetic price <= buy_limit - one tick
→ FILLED

그 외
→ NO_FILL
```

이 규칙은 실제 호가 대기열을 재현하지 않으며,
가격 차트만으로 가능한 보수적 체결 proxy다.

# 10. 수수료 모델

수수료는 설정 파일로 분리한다.

초기 baseline:

```yaml
fees:
  buy_commission_rate: 0.001
  sell_commission_rate: 0.001
  sec_fee_rate: 0.0000206
  taf_per_share: 0.000195
  taf_max_per_trade: 9.79
  commission_rounding: nearest_cent
  regulatory_fee_rounding: ceil_cent
```

계산은 `decimal.Decimal`을 사용한다.

```python
buy_notional = buy_price * shares
sell_notional = sell_price * shares

buy_commission = round_cent(
    buy_notional * buy_commission_rate
)

sell_commission = round_cent(
    sell_notional * sell_commission_rate
)

sec_fee = ceil_cent(
    sell_notional * sec_fee_rate
)

taf_fee = ceil_cent(
    min(
        shares * taf_per_share,
        taf_max_per_trade,
    )
)

net_pnl = (
    sell_notional
    - buy_notional
    - buy_commission
    - sell_commission
    - sec_fee
    - taf_fee
)

net_return = (
    net_pnl
    / (buy_notional + buy_commission)
)
```

주의:

- 수수료 값은 config로 관리한다.
- 실제 토스 계정의 우대 또는 이벤트 수수료가 있으면 쉽게 변경 가능해야 한다.
- 규제 수수료 반올림 방식은 실제 거래명세서 확보 후 수정 가능하게 설계한다.

---

# 11. Barrier 정의

## 11.1 초기 고정 Barrier

첫 모델은 해석이 쉬운 고정값으로 시작한다.

```yaml
barriers:
  take_profit_pct: 0.05
  stop_loss_pct: 0.03
  horizons_minutes: [10]
```

진입가 기준:

```python
take_profit_price = (
    entry_price * (1 + take_profit_pct)
)

stop_loss_price = (
    entry_price * (1 - stop_loss_pct)
)
```

## 11.2 종료 조건

각 Monte Carlo 경로에서 진입 후 다음 중 먼저 발생한 것을 기록한다.

```text
TAKE_PROFIT
STOP_LOSS
TIMEOUT
NO_FILL
```

## 11.3 시간 종료

horizon 종료 시 마지막 합성 가격을 exit price proxy로 사용한다.

필요하면 spread haircut을 적용한다.

---

# 12. OHLC 조건부 합성틱 Monte Carlo

## 12.1 목적

1분봉 OHLC만으로는 high와 low의 발생 순서를 알 수 없다.

따라서 각 미래 1분봉에 대해 OHLC를 만족하는 가능한 내부 경로를 여러 개 생성한다.

단일 경로를 정답으로 고정하지 않는다.

금지:

```text
모든 양봉 = O-L-H-C
모든 음봉 = O-H-L-C
```

초기에는 두 극값 순서를 동일 확률로 둔다.

```text
O-H-L-C: 50%
O-L-H-C: 50%
```

## 12.2 경로 필수 조건

각 합성 경로는 다음을 만족해야 한다.

```text
첫 가격 = Open
마지막 가격 = Close
최고값 = High
최저값 = Low
모든 가격은 tick grid에 위치
```

## 12.3 단계별 구현

### Stage 1: Dual-path 검증

먼저 다음 두 deterministic path를 구현한다.

```text
O-H-L-C
O-L-H-C
```

두 경로에서 TP/SL 결과가 같으면 확정성이 높은 샘플이다.

결과가 다르면 `ambiguous`.

이 단계로 barrier 코드와 fee 코드를 검증한다.

### Stage 2: Monte Carlo

권장 초기 설정:

```yaml
synthetic_ticks:
  paths_per_sample: 200
  min_ticks_per_bar: 20
  max_ticks_per_bar: 300
  seed: 42
```

1분봉 내 합성 tick 수는 거래량이나 체결 건수를 사용하지 않고 고정값 또는
OHLC 변동폭 기반으로 결정한다.

초기 권장:

```python
n_ticks = base_ticks_per_bar
```

보조 방식:

```python
range_ratio = (high - low) / open

n_ticks = clip(
    base_ticks_per_bar
    + int(range_ratio / range_step),
    min_ticks_per_bar,
    max_ticks_per_bar,
)
```

이는 실제 체결 수를 추정하는 값이 아니라 합성 경로의 수치적 해상도다.

## 12.4 경로 생성 방법

각 봉의 극값 순서를 먼저 선택한다.

```text
O -> H -> L -> C
또는
O -> L -> H -> C
```

각 구간 길이는 Dirichlet distribution으로 배분한다.

각 구간 내부는 Brownian bridge 또는 constrained random walk로 생성한다.

권장 절차:

1. 극값 순서 샘플링
2. 세 구간의 tick 수 할당
3. 시작점·종료점 고정 bridge 생성
4. OHLC 범위 밖 값 reject 또는 clipping
5. high와 low를 정확히 한 번 이상 포함
6. tick size 단위로 quantize
7. 최종적으로 OHLC 보존 검사

## 12.5 거래량·VWAP 사용 금지

합성 경로 생성 시 다음 값은 사용하지 않는다.

```text
volume
trade_count
trade_tick_count
notional_usd
alpaca_vwap
session_vwap
quote 및 trade-flow 컬럼
```

합성 경로는 OHLC와 tick size만 만족하도록 생성한다.
VWAP은 경로 생성, 경로 선택, 제약 조건, 보정, sample weight 및
라벨 신뢰도 계산의 어느 단계에서도 사용하지 않는다.

## 12.6 미래 여러 봉 연결

10분 horizon은 미래 1분봉별 합성 경로를 시간 순서대로 연결한다.

각 미래 봉 open과 이전 봉 close 사이 gap은 그대로 유지한다.

## 12.7 합성 청산가격

Alpaca quote/spread는 Toss와 다를 수 있으므로 라벨에도 사용하지 않는다.

TP·SL barrier 판정은 합성 가격 자체로 수행한다.

실제 체결 마찰은 다음 두 요소로 반영한다.

1. Toss 거래 수수료
2. 설정 가능한 고정 슬리피지 또는 가격대별 슬리피지 가정

```python
sell_fill_price = (
    trigger_price
    * (1 - sell_slippage_pct)
)
```

초기에는 여러 슬리피지 시나리오를 비교한다.

```yaml
labeling:
  sell_slippage_scenarios:
    - 0.0000
    - 0.0005
    - 0.0010
    - 0.0020
```

Primary 모델의 라벨에는 보수적인 기본값을 선택하되 config로 관리한다.

## 12.8 TP/SL 선도달

진입 체결 이후 합성 bid 기준으로 판정한다.

```python
for synthetic_bid in path_after_fill:

    if synthetic_bid <= stop_loss_price:
        result = "STOP_LOSS"
        break

    if synthetic_bid >= take_profit_price:
        result = "TAKE_PROFIT"
        break
```

합성틱은 순서가 존재하므로 동일 봉 내 TP/SL 충돌 문제는 발생하지 않는다.

---

# 13. Monte Carlo 라벨

각 시점의 10분 horizon에 대해 저장한다.

```text
entry_fill_probability
p_tp_first
p_sl_first
p_timeout
expected_gross_return
expected_net_return
expected_holding_minutes
expected_mfe
expected_mae
label_confidence
mc_paths
mc_seed
label_version
```

## 13.1 Hard label

초기 기준:

```yaml
labels:
  strong_buy_min_fill_probability: 0.70
  strong_buy_min_tp_probability: 0.65
  strong_buy_min_expected_net_return: 0.0
  avoid_min_sl_probability: 0.65
```

정의:

```text
STRONG_BUY:
- entry_fill_probability >= 0.70
- p_tp_first >= 0.65
- expected_net_return > 0

AVOID:
- p_sl_first >= 0.65
또는
- expected_net_return < 0 이면서 label confidence가 높음

UNCERTAIN:
- 나머지
```

초기 binary target:

```text
1 = STRONG_BUY
0 = AVOID + UNCERTAIN
```

별도 실험:

```text
1 = STRONG_BUY
0 = AVOID
UNCERTAIN 제외
```

## 13.2 Soft target

모델의 주 target으로 soft label을 권장한다.

```text
target_tp_probability = p_tp_first
target_expected_net_return = expected_net_return
```

## 13.3 Sample weight

```python
direction_confidence = abs(
    p_tp_first - p_sl_first
)

fill_confidence = (
    entry_fill_probability
)

sample_weight = (
    direction_confidence
    * fill_confidence
)
```

추가 데이터 품질 점수를 곱할 수 있다.

---

# 14. 데이터셋 분할

랜덤 row split은 금지한다.

## 14.1 시간 기반 분할

```text
Train: 과거 날짜
Validation: 이후 날짜
Test: 가장 최근 날짜
```

## 14.2 Purge 및 embargo

10분 미래 라벨을 사용하므로 인접 구간 누수를 막는다.

```yaml
split:
  method: purged_time_split
  embargo_minutes: 10
```

Train과 validation 경계에서 최소 10분을 제거한다.

## 14.3 종목 일반화 평가

가능하면 다음 두 평가를 분리한다.

```text
Seen-symbol future-date test
Unseen-symbol test
```

데이터가 충분하지 않으면 우선 future-date test부터 구현한다.

---

# 15. 학습 모델

## 15.1 구현 순서

### Baseline 1: PyTorch Stage-wise Gradient Boosting

최근 60봉을 통계 요약해 학습한다.

각 feature에 대해 예:

```text
last
mean
std
min
max
slope
first_to_last_change
```

첫 구현은 PyTorch 기반 얕은 weak learner를 stage-wise additive 방식으로 학습한다.
각 stage는 이전 logit을 고정하고 binary cross-entropy를 줄이는 새 learner를 추가한다.

```text
Direct: P(TP first within 10m)
Fill: P(fill)
Conditional: P(TP first within 10m | fill)
Two-stage: P(fill) × P(TP first within 10m | fill)
```

LightGBM, XGBoost, CatBoost는 이후 비교 후보이며 첫 baseline에는 사용하지 않는다.

목적:

- 파이프라인 검증
- 주요 feature 확인
- sequence model 비교 기준

### Baseline 2: GRU

첫 sequence model 권장 구조:

```yaml
model:
  architecture: gru
  hidden_size: 128
  num_layers: 2
  dropout: 0.2
  bidirectional: false
```

입력:

```text
[batch, 60, feature_count]
```

출력:

```text
tp_probability
expected_net_return
strong_buy_probability
```

### Baseline 3: TCN

GRU와 비교한다.

Transformer는 데이터량이 충분한지 확인한 뒤 후순위로 둔다.

## 15.2 손실 함수

권장 멀티태스크 loss:

```python
total_loss = (
    bce_probability_loss
    + lambda_return * huber_return_loss
    + lambda_class * focal_class_loss
)
```

sample weight를 적용한다.

---

# 16. 학습 설정

권장 초기값:

```yaml
training:
  batch_size: 256
  learning_rate: 0.001
  max_epochs: 100
  early_stopping_patience: 10
  gradient_clip_norm: 1.0
  seed: 42
```

Validation PR-AUC 또는 validation expected trading return을 기준으로 early stopping한다.

단, threshold 최적화는 validation에서만 수행한다.

---

# 17. 평가 지표

## 17.1 ML 지표

```text
PR-AUC
ROC-AUC
Precision
Recall
F1
Brier score
Calibration error
Precision at top 1%
Precision at top 5%
Precision at top 10%
```

단타 매수 모델에서는 accuracy보다 precision과 calibration을 우선한다.

## 17.2 거래 관점 지표

라벨 기반 가상 거래 결과로 계산한다.

```text
평균 expected net return
선택된 신호의 p_tp_first 평균
선택된 신호의 p_sl_first 평균
예상 승률
예상 profit factor
신호 수
종목별 성과
세션별 성과
spread 구간별 성과
가격대별 성과
```

## 17.3 Threshold 선정

0.5 고정 금지.

Validation set에서 다음 조건을 만족하는 threshold를 선택한다.

```text
minimum precision
minimum signal count
positive mean expected net return
acceptable loss probability
```

예시:

```yaml
decision:
  min_probability: 0.75
  min_expected_net_return: 0.003
```

Test에서는 validation에서 고정한 threshold를 그대로 사용한다.

---

# 18. 필수 Ablation

다음을 비교한다.

1. Raw OHLC relative features
2. OHLC + candle structure
3. OHLC + price trend indicators
4. OHLC + price volatility indicators
5. OHLC + momentum indicators
6. OHLC + time features
7. All provider-independent price features
8. hard label vs soft label
9. deterministic dual path vs Monte Carlo
10. close-minus-tick vs close vs next-open entry proxy
11. 슬리피지 시나리오별 라벨 안정성
12. 30봉 vs 60봉
13. 10분 horizon 고정 라벨의 날짜·종목별 안정성
14. GRU vs TCN vs boosting baseline

최종 모델은 AUC가 아니라 다음을 함께 기준으로 선택한다.

```text
out-of-sample precision
calibration
expected net return
signal count
종목 간 안정성
날짜 간 안정성
```

---

# 19. 저장 artifact

학습 완료 후 반드시 저장한다.

```text
model weights
model config
feature list
feature order
scaler
label config
fee config
training config
decision threshold
calibration model
dataset version
feature version
label version
git commit
random seed
```

권장 구조:

```text
artifacts/
└─ model_version/
   ├─ model.pt
   ├─ scaler.pkl
   ├─ feature_schema.json
   ├─ model_config.yaml
   ├─ labeling_config.yaml
   ├─ fee_config.yaml
   ├─ threshold.json
   ├─ calibration.pkl
   └─ metrics.json
```

---

# 20. 권장 프로젝트 구조

```text
project/
├─ AGENT.md
├─ README.md
├─ pyproject.toml
├─ configs/
│  ├─ base.yaml
│  ├─ features.yaml
│  ├─ labeling.yaml
│  └─ training.yaml
├─ data/
│  ├─ raw/
│  ├─ interim/
│  └─ processed/
├─ src/
│  ├─ data/
│  │  ├─ loader.py
│  │  ├─ schema.py
│  │  ├─ validator.py
│  │  └─ splits.py
│  ├─ features/
│  │  ├─ price.py
│  │  ├─ volume.py
│  │  ├─ quote.py
│  │  ├─ trades.py
│  │  ├─ temporal.py
│  │  └─ pipeline.py
│  ├─ labeling/
│  │  ├─ fees.py
│  │  ├─ tick_size.py
│  │  ├─ synthetic_path.py
│  │  ├─ monte_carlo.py
│  │  ├─ barriers.py
│  │  └─ labels.py
│  ├─ datasets/
│  │  ├─ sequence_dataset.py
│  │  └─ build_dataset.py
│  ├─ models/
│  │  ├─ boosting.py
│  │  ├─ gru.py
│  │  ├─ tcn.py
│  │  └─ losses.py
│  ├─ training/
│  │  ├─ train.py
│  │  ├─ evaluate.py
│  │  ├─ calibrate.py
│  │  └─ registry.py
│  └─ utils/
│     ├─ config.py
│     ├─ hashing.py
│     └─ reproducibility.py
├─ tests/
│  ├─ test_schema.py
│  ├─ test_features_no_leakage.py
│  ├─ test_fee_model.py
│  ├─ test_synthetic_ohlc.py
│  ├─ test_barrier_order.py
│  ├─ test_sequence_builder.py
│  └─ test_split_purge.py
├─ notebooks/
│  ├─ 01_data_audit.ipynb
│  ├─ 02_feature_audit.ipynb
│  ├─ 03_label_audit.ipynb
│  ├─ 04_gradient_boosting_baseline.ipynb
│  └─ 05_model_evaluation.ipynb
└─ scripts/
   ├─ audit_data.py
   ├─ build_labels.py
   ├─ build_dataset.py
   ├─ train_model.py
   └─ evaluate_model.py
```

---

# 21. 구현 순서

초기 모델은 Monte Carlo보다 데이터 편향 감사, Dual-path 확정 라벨,
단순 baseline 및 순차 백테스트를 우선한다. Monte Carlo 라벨은 baseline의
한계가 확인된 뒤 유효성을 별도로 검증하는 보조 실험으로 진행한다.

모든 Phase에서 거래량과 VWAP은 모델 입력 및 라벨 생성에 사용하지 않는다.

## Phase 1. 데이터 및 universe 감사

구현:

1. enriched CSV loader
2. schema 검증
3. timestamp 정렬
4. 중복 검사
5. OHLC 오류 검사
6. gap 검사
7. quote 이상치 검사
8. unknown trade 비율 분석
9. feature NaN 분석
10. 세션별 수집 종목과 종목 선정 기준 확인
11. 급등·거래대금 상위 종목 중심의 selection bias 확인
12. 실패한 급등 후보와 일반 종목 등 negative coverage 확인
13. 실시간 후보 종목 선별 방식과 학습 universe의 일치 여부 확인

산출물:

```text
reports/data_audit.md
```

## Phase 2. Feature pipeline 및 데이터셋 골격

구현:

1. 파생 feature 생성
2. feature schema 저장
3. 60봉 sequence 생성
4. 미래정보 누수 테스트
5. train-only scaler
6. metadata schema 저장
7. purged time split 구현
8. dataset version 저장

## Phase 3. Fee, barrier 및 Dual-path 확정 라벨

구현:

1. Decimal 수수료 계산
2. TP/SL 가격 계산
3. horizon timeout
4. net return 계산
5. `O-H-L-C`, `O-L-H-C` 두 경로 판정
6. 두 경로의 결과가 같은 확정 샘플 분리
7. 경로에 따라 결과가 다른 ambiguous 샘플 별도 저장
8. 라벨·수수료 단위 테스트

초기 baseline은 확정 샘플을 우선 사용한다. ambiguous 샘플을 임의로
positive 또는 negative로 강제 변환하지 않는다.

## Phase 4. PyTorch Stage-wise Gradient Boosting baseline

구현:

1. 확정 라벨과 60봉 요약 feature 연결
2. direct와 two-stage Torch boosting baseline 학습
3. validation calibration
4. validation threshold 선정
5. 고정 threshold로 test 평가
6. 날짜별·종목별·가격대별 성능 보고

GRU와 TCN을 시작하기 전에 데이터와 라벨 파이프라인이 baseline에서
정상적으로 동작하는지 검증한다.

## Phase 5. 순차 백테스트

구현:

1. 시간순 signal 생성
2. 중복 signal 및 재진입 규칙 적용
3. 주문 TTL과 미체결 처리
4. 동시 보유 수와 자금 제약 적용
5. 수수료와 슬리피지 반영
6. 손익곡선, MDD, profit factor 및 turnover 계산
7. 세션별·종목별 성과 안정성 분석

겹치는 1분 샘플을 모두 독립 거래로 간주해 성과를 부풀리지 않는다.

## Phase 6. 진입 및 비용 민감도 분석

비교:

1. `close_minus_tick`, `close`, `next_open` 진입 proxy
2. tick size 정책
3. 수수료 시나리오
4. 슬리피지 시나리오
5. 고정 barrier와 변동성 기반 barrier
6. 10분 horizon 고정 하의 비용·체결 민감도

10분 horizon은 프로젝트 목적으로 고정하며 validation이나 test 결과에 따라 늘리지 않는다.

## Phase 7. Monte Carlo 보조 라벨 검증

Dual-path baseline과 순차 백테스트가 완료된 후 진행한다.

구현:

1. constrained synthetic path
2. OHLC 보존
3. tick quantization
4. 거래량·VWAP 미사용 assertion
5. multi-bar 연결
6. 체결 판정
7. TP/SL 선도달
8. 확률 및 expected return 저장
9. Dual-path 확정 샘플과 결과 일치도 검사
10. 합성 경로 prior 및 seed 민감도 분석

Monte Carlo 라벨이 out-of-sample 성능과 calibration을 실제로 개선할 때만
primary 학습 라벨로 승격한다.

## Phase 8. 시퀀스 모델 및 Ablation

구현:

1. GRU
2. TCN
3. boosting baseline과 동일 split 비교
4. feature group ablation
5. hard label과 soft label 비교
6. Dual-path와 Monte Carlo 비교
7. 모델 구조별 calibration 및 순차 백테스트 비교

---

# 22. 필수 테스트

## 22.1 합성 경로

모든 경로에서:

```text
path[0] == open
path[-1] == close
max(path) == high
min(path) == low
모든 가격이 tick grid에 존재
```

## 22.2 Barrier

인공 경로로 검증한다.

```text
TP first
SL first
TIMEOUT
NO_FILL
gap up
gap down
```

## 22.3 수수료

다음 케이스를 테스트한다.

```text
소액 거래
고가주 거래
손실 거래
TAF 상한
SEC fee 1센트 미만
```

## 22.4 누수

원본 데이터의 `t+1` 이후 값을 변경해도 `t` 시점 feature가 변하지 않아야 한다.

## 22.5 시퀀스

```text
정확히 60행
동일 symbol
연속 1분
미래 행 없음
metadata timestamp와 마지막 봉 일치
```

## 22.6 Split

train label window와 validation label window가 겹치지 않아야 한다.

---

# 23. 초기 설정 예시

```yaml
project:
  seed: 42
  sequence_length: 60
  primary_horizon_minutes: 10

data:
  reject_sequence_with_gap: true

features:
  use_price: true
  use_price_technical: true
  use_time: true
  use_volume: false
  use_notional: false
  use_vwap: false
  use_quote: false
  use_trade_flow: false
  provider_independent_only: true

entry:
  live_price_rule: toss_best_ask_minus_one_tick
  labeling_price_rule: close_minus_one_tick
  fill_rule: penetrate_one_tick
  order_ttl_minutes: 1

tick_size:
  price_below_1: 0.0001
  price_at_or_above_1: 0.01

barriers:
  take_profit_pct: 0.05
  stop_loss_pct: 0.03
  horizons_minutes: [10]

synthetic_ticks:
  paths_per_sample: 200
  min_ticks_per_bar: 20
  max_ticks_per_bar: 300
  high_first_probability: 0.5
  low_first_probability: 0.5
  base_ticks_per_bar: 100
  range_step: 0.001
  seed: 42

labeling:
  sell_slippage_pct: 0.001
  strong_buy_min_fill_probability: 0.70
  strong_buy_min_tp_probability: 0.65
  strong_buy_min_expected_net_return: 0.0
  avoid_min_sl_probability: 0.65

fees:
  buy_commission_rate: 0.001
  sell_commission_rate: 0.001
  sec_fee_rate: 0.0000206
  taf_per_share: 0.000195
  taf_max_per_trade: 9.79
  commission_rounding: nearest_cent
  regulatory_fee_rounding: ceil_cent

split:
  method: purged_time_split
  embargo_minutes: 10

model:
  architecture: gru
  hidden_size: 128
  num_layers: 2
  dropout: 0.2

training:
  batch_size: 256
  learning_rate: 0.001
  max_epochs: 100
  early_stopping_patience: 10
  gradient_clip_norm: 1.0

decision:
  min_probability: 0.75
  min_expected_net_return: 0.003
```

위 값은 초기 실험용이며 최적값으로 간주하지 않는다.

---

# 24. Codex 작업 규칙

1. 현재 단계는 모델 개발까지만 진행한다.
2. 실시간 주문·자동매매 코드를 만들지 않는다.
3. 각 Phase별로 구현하고 테스트를 통과한 뒤 다음 단계로 이동한다.
4. 실제 CSV를 읽지 않고 스키마를 추측하지 않는다.
5. 원본 컬럼을 덮어쓰지 않는다.
6. 모든 파라미터를 YAML config로 관리한다.
7. 랜덤 seed를 고정한다.
8. 데이터셋, feature, label, model version을 기록한다.
9. 미래정보 누수 테스트를 반드시 작성한다.
10. 모델 선택은 accuracy가 아니라 out-of-sample precision, calibration, expected net return으로 한다.
11. Alpaca와 Toss 사이에 달라질 수 있는 거래량·거래대금·VWAP·호가·체결 데이터는 모델 입력에서 제외한다.
12. 최종 feature list는 Toss의 60개 OHLC만으로 동일하게 재계산 가능해야 한다.
13. provider-dependent 컬럼이 model feature에 포함되면 테스트를 실패시킨다.
