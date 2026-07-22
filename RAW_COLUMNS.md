# Enriched 1분봉 컬럼 설명

이 문서는 `*_enriched.csv`에 저장된 55개 컬럼을 설명한다. `enriched` 파일은
`bars` 파일의 29개 컬럼에 분봉별 호가 집계와 개별 체결 집계 26개를 추가한 데이터다.

> 주의: 파일 이름에 `raw`가 포함되어 있어도 모든 컬럼이 원천 데이터는 아니다.
> OHLCV·호가·체결 집계와 사전에 계산된 기술지표가 같은 파일에 들어 있다.

## 공통 단위와 시간 기준

- 가격과 거래대금의 통화는 USD다.
- `volume`, 호가 잔량, 체결량의 단위는 주식 수(shares)다.
- 이름이 `_pct`이거나 `strength`, `pressure`, `ratio`인 컬럼은 대부분 `0.01`이 아닌
  퍼센트 포인트 단위다. 예를 들어 `3.0`은 3%를 뜻한다.
- 한 행은 해당 timestamp에서 시작하는 1분 구간이다. 그 행의 전체 집계값은 1분봉이
  완성된 뒤에만 확정된다.
- 아래 산식은 실제 CSV 값으로 검증했다. 공격적 매수·매도 체결의 세부 분류 알고리즘은
  manifest에 명시되어 있지 않으므로 의미 기준으로 설명한다.

## 1. 식별자와 시간

| 컬럼 | 형식·단위 | 설명 |
|---|---|---|
| `symbol` | 문자열 | 미국 주식 티커. 예: `ALM`, `BMNR`. |
| `timestamp_kst` | KST datetime 문자열 | 해당 1분봉의 시작 시각을 한국 표준시로 표현한 값. CSV에는 timezone offset이 별도로 붙지 않는다. |
| `timestamp_utc` | UTC datetime 문자열 | 같은 1분봉 시작 시각의 UTC 표현. `timestamp_kst`와 동일한 순간이다. |

## 2. 1분봉과 거래량

| 컬럼 | 형식·단위 | 설명 |
|---|---|---|
| `open` | USD/주 | 해당 1분에 체결된 첫 거래 가격. |
| `high` | USD/주 | 해당 1분의 최고 체결 가격. |
| `low` | USD/주 | 해당 1분의 최저 체결 가격. |
| `close` | USD/주 | 해당 1분의 마지막 체결 가격. |
| `volume` | shares | Alpaca 1분봉의 총 체결량. |
| `trade_count` | 건 | Alpaca 1분봉에 집계된 체결 건수. 아래 `trade_tick_count`와 출처·집계 과정이 달라 드물게 차이가 날 수 있다. |
| `alpaca_vwap` | USD/주 | Alpaca가 제공한 해당 1분의 체결량 가중 평균 체결 가격(VWAP). |
| `notional_usd` | USD | 단순 거래대금 proxy. 실제 파일에서는 `close × volume`으로 계산된다. 개별 체결가격을 합산한 실제 거래대금은 `trade_notional_from_ticks`다. |

## 3. 사전 계산된 가격·거래량 지표

모든 rolling 지표는 현재 행을 포함한다. 따라서 `t`봉이 완성된 뒤에는 사용할 수 있지만,
봉이 완성되기 전에 완성값으로 사용하면 미래 정보가 섞인다. 필요한 과거 봉 수가 없으면
초기 행은 `NaN`이다.

| 컬럼 | 형식·단위 | 설명 |
|---|---|---|
| `volume_ma20` | shares | 현재 봉을 포함한 최근 20개 관측 봉 `volume`의 단순이동평균. |
| `ma5` | USD/주 | `close`의 5봉 단순이동평균(SMA). |
| `ma20` | USD/주 | `close`의 20봉 단순이동평균. |
| `ma60` | USD/주 | `close`의 60봉 단순이동평균. |
| `ma120` | USD/주 | `close`의 120봉 단순이동평균. |
| `ema9` | USD/주 | `close`의 9봉 지수이동평균. 첫 EMA는 최초 9개 종가의 SMA로 시작한다. |
| `ema20` | USD/주 | `close`의 20봉 지수이동평균. 첫 EMA는 최초 20개 종가의 SMA로 시작한다. |
| `bb_mid20` | USD/주 | 20봉 볼린저밴드 중심선. `close`의 20봉 SMA이며 `ma20`과 같다. |
| `bb_upper20` | USD/주 | 볼린저 상단. `bb_mid20 + 2 × 20봉 종가 모집단 표준편차(ddof=0)`. |
| `bb_lower20` | USD/주 | 볼린저 하단. `bb_mid20 - 2 × 20봉 종가 모집단 표준편차(ddof=0)`. |
| `bb_position_0_1` | 무단위 | 밴드 내 종가 위치. `(close - bb_lower20) / (bb_upper20 - bb_lower20)`. 하단은 0, 중심은 약 0.5, 상단은 1이다. 밴드 밖이면 0보다 작거나 1보다 클 수 있다. |
| `session_vwap` | USD/주 | 파일 수집 세션 시작부터 현재 봉까지의 누적 VWAP proxy. 각 봉의 typical price `(high + low + close) / 3`에 `volume`을 가중하여 계산한다. `alpaca_vwap`의 누적값은 아니다. |
| `macd_12_26` | USD/주 | `close`의 12봉 EMA에서 26봉 EMA를 뺀 MACD 선. |
| `macd_signal_9` | USD/주 | `macd_12_26`의 9봉 EMA인 signal 선. |
| `macd_histogram` | USD/주 | `macd_12_26 - macd_signal_9`. 양수면 단기 상승 모멘텀이 signal보다 강하다는 뜻이다. |
| `candle_color` | 문자열 | `close >= open`이면 `up`, `close < open`이면 `down`. 시가와 종가가 같아도 `up`으로 저장된다. |
| `return_pct` | % | 직전 관측 봉 종가 대비 현재 종가 수익률. `(close / previous_close - 1) × 100`. 파일의 첫 행은 보통 `NaN`이다. |
| `range_pct` | % | 현재 봉의 고저 변동폭. `(high - low) / close × 100`. 일반적인 open 또는 low 기준 변동폭과 분모가 다르다. |

## 4. 분봉 내 호가 집계

`bid`는 즉시 매도할 때 받을 수 있는 매수호가이고, `ask`는 즉시 매수할 때 지불하는
매도호가다. 해당 분에 유효한 quote가 없으면 아래 컬럼들이 `NaN`일 수 있다.

| 컬럼 | 형식·단위 | 설명 |
|---|---|---|
| `quote_count` | 건 | 해당 1분에 수집된 quote 레코드 수. 결측이면 그 분의 quote 집계가 없다는 뜻이다. |
| `first_bid` | USD/주 | 해당 분에 처음 관측된 bid 가격. |
| `first_ask` | USD/주 | 해당 분에 처음 관측된 ask 가격. 시장가 매수 체결 proxy로 사용할 수 있다. |
| `last_bid` | USD/주 | 해당 분에 마지막으로 관측된 bid 가격. 분 종료 시 시장가 매도 proxy로 사용할 수 있다. |
| `last_ask` | USD/주 | 해당 분에 마지막으로 관측된 ask 가격. |
| `avg_bid` | USD/주 | 해당 분에 수집된 bid 가격의 평균. |
| `avg_ask` | USD/주 | 해당 분에 수집된 ask 가격의 평균. |
| `min_bid` | USD/주 | 해당 분에 관측된 bid 가격 중 최솟값. 실제 한 번의 매도 체결가가 아니라 분봉 내 최악의 bid다. |
| `max_bid` | USD/주 | 해당 분에 관측된 bid 가격 중 최댓값. |
| `min_ask` | USD/주 | 해당 분에 관측된 ask 가격 중 최솟값. |
| `max_ask` | USD/주 | 해당 분에 관측된 ask 가격 중 최댓값. |
| `avg_spread_pct` | % | 각 quote의 midpoint 기준 상대 스프레드 `2 × (ask - bid) / (ask + bid) × 100`을 계산한 뒤 해당 분에서 평균낸 값. |
| `min_spread_pct` | % | 해당 분의 quote별 midpoint 기준 상대 스프레드 최솟값. |
| `max_spread_pct` | % | 해당 분의 quote별 midpoint 기준 상대 스프레드 최댓값. |
| `avg_bid_size` | shares | 해당 분 quote의 표시 bid 잔량 평균. 전체 호가창 깊이가 아니라 수집된 최우선 bid size다. |
| `avg_ask_size` | shares | 해당 분 quote의 표시 ask 잔량 평균. 전체 호가창 깊이가 아니라 수집된 최우선 ask size다. |
| `bid_ask_imbalance` | 0~1 | 최우선 호가 잔량 비율. `avg_bid_size / (avg_bid_size + avg_ask_size)`. 0.5보다 크면 표시 bid 잔량이 상대적으로 많다. |

`min_bid`와 `max_ask`처럼 서로 다른 시점의 극값을 조합해 실제 spread 또는 실제 체결쌍으로
해석하면 안 된다. 같은 시점의 bid/ask로 계산된 spread 컬럼을 사용해야 한다.

## 5. 개별 체결(tick) 집계와 체결 방향

| 컬럼 | 형식·단위 | 설명 |
|---|---|---|
| `trade_tick_count` | 건 | 개별 trade tick 데이터에서 해당 분에 집계한 체결 레코드 수. `trade_count`와 대부분 같지만 데이터 수집·정렬 차이로 일치하지 않을 수 있다. |
| `trade_volume_from_ticks` | shares | 개별 trade tick의 체결량 합계. 아래 매수·매도·미분류 체결량의 합과 같다. `volume`과 대부분 같지만 feed 누락이나 정렬 차이가 있으면 달라질 수 있다. |
| `aggressive_buy_volume` | shares | 매수자가 ask를 받아 체결한 것으로 분류된 buyer-initiated 체결량. 시장가성 매수 압력의 proxy다. |
| `aggressive_sell_volume` | shares | 매도자가 bid를 때려 체결한 것으로 분류된 seller-initiated 체결량. 시장가성 매도 압력의 proxy다. |
| `unknown_trade_volume` | shares | 당시 quote와 안정적으로 매칭되지 않아 매수·매도 방향을 결정하지 못한 체결량. |
| `trade_notional_from_ticks` | USD | 개별 체결마다 `trade_price × trade_size`를 계산해 합산한 실제 tick 기반 거래대금. |
| `trade_strength` | % | 방향이 분류된 거래 중 공격적 매수 비중. `aggressive_buy_volume / (aggressive_buy_volume + aggressive_sell_volume) × 100`. 분모가 0이면 `NaN`일 수 있다. |
| `sell_pressure` | % | 방향이 분류된 거래 중 공격적 매도 비중. `aggressive_sell_volume / (aggressive_buy_volume + aggressive_sell_volume) × 100`. 유효한 행에서는 `trade_strength + sell_pressure = 100`. |
| `unknown_trade_ratio` | % | 전체 tick 체결량 중 미분류 체결량 비율. `unknown_trade_volume / trade_volume_from_ticks × 100`. 높을수록 `trade_strength`와 `sell_pressure`의 신뢰도가 낮다. |

## 모델링 시 특히 주의할 점

1. **완성 시점**: 한 행의 `high`, `low`, `close`, `volume`, `last_*`, `min_*`, `max_*`는
   그 1분이 끝나기 전에는 확정되지 않는다.
2. **절대값 일반화**: 가격, 거래량, 거래대금은 종목마다 규모가 크게 다르다. 모델 입력에는
   수익률, 로그 변환, 과거 rolling 기준 비율 등을 함께 검토한다.
3. **호가 결측**: quote가 없는 행의 호가 컬럼을 0으로 채우면 실제 0달러 호가로 오인된다.
   결측 여부를 별도 mask로 보존하는 것이 안전하다.
4. **체결 방향 신뢰도**: `unknown_trade_ratio`가 높으면 aggressive buy/sell 비율의 의미가
   약해진다. 방향성 feature와 함께 품질 feature로 사용한다.
5. **중복 파일**: `enriched`는 `bars` 컬럼을 이미 포함한다. 두 파일을 행 방향으로 합치면
   같은 분봉이 중복된다.
6. **파일 완전성**: 현재 원본에는 `bars` 파일 211개 중 대응 `enriched`가 없는 파일이
   4개 있다. 전체 적재 전에 파일 쌍 존재 여부를 검사해야 한다.
