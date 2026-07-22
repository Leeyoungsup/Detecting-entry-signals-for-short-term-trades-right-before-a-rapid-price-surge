# AGENT.md

## 프로젝트 목표

미국 주식 1분봉 데이터로 **초단타 스캘핑 진입 시점**을 판단하는 AI 모델을 개발한다.

완료된 `t`분봉 직후 시장가로 진입했을 때, 앞으로 1~5분 안에 모든 비용을 제외하고
3% 이상의 순수익을 실현할 수 있는지를 예측하는 것이 첫 번째 목표다.

## 프로젝트 경로

모든 상대 경로는 프로젝트 루트 기준이다.

```text
원본 데이터: ../../data/stock_data/raw
모델:        ../../model
결과:        ../../results
가상환경:    conda urban
```

- 원본 `raw` 데이터는 수정하지 않는다.
- 새 전처리 결과와 학습 결과를 원본 데이터 폴더에 저장하지 않는다.
- 개발은 Jupyter notebook 순서로 진행한다.

## 판단 시점과 시간 정렬

- 원본 1분봉 timestamp는 해당 1분 구간의 시작 시각으로 취급한다.
- 예를 들어 `19:23`봉의 OHLCV는 봉이 끝나는 약 `19:24`에 사용할 수 있다.
- 모델은 `t`봉이 완전히 종료된 직후 한 번 실행한다.
- 모델 입력은 `t-29`부터 `t`까지 정확히 연속된 30개 완성 봉이다.
- `t+1`의 완성 OHLCV, 거래량, VWAP 또는 고가·저가는 feature로 사용하지 않는다.
- 역사적 진입 체결 proxy는 `t+1`분의 첫 executable ask인 `first_ask[t+1]`이다.
- quote는 진입·청산 체결과 label 계산에 사용할 수 있지만 미래 quote를 feature로
  사용하지 않는다.
- 입력 30봉이나 미래 5분 사이에 1분 초과 gap이 있으면 해당 표본을 만들지 않는다.

## 모델 입력

기본 sequence 길이는 60봉이 아니라 **30봉**이다.

### 원본 입력

- `open`, `high`, `low`, `close`
- `volume`, `trade_count`
- `alpaca_vwap`

### 과거 데이터로 계산할 feature

- 1분 수익률, 봉 몸통·전체 범위·윗꼬리·아랫꼬리
- `log1p(volume)`, 과거 5/10/20봉 대비 상대 거래량, 거래량 변화율
- 직전 20봉 통상 거래대금 대비 현재 총 거래대금과 공격적 순매수 거래대금
- EMA 5/9/20과 현재가의 거리
- 당일 누적 VWAP과 현재가의 거리
- 판단 시점까지의 당일 전고점·전저점과 현재가의 거리
- 최근 5/10/20봉 고점·저점과 현재가의 거리
- 가격 변화와 거래량 변화의 상호작용

당일 VWAP, 전고점, 전저점은 반드시 `t`까지의 정보만 사용해 다시 계산한다.
원본의 사전 계산 컬럼을 사용할 때도 미래 정보가 섞이지 않았는지 먼저 검증한다.

## 첫 번째 label과 실행 규칙

- 진입 가격: `first_ask[t+1]`
- 탐색 구간: 진입한 `t+1`분부터 최대 5분
- 보수적 TP 판정 가격: 각 미래 완성 1분봉의 executable `min_bid`
- positive label: 미래 1~5분 중 한 번이라도 **수수료와 규제 비용을 제외한
  순수익률이 +3% 이상**이면 1, 아니면 0
- gross +3% 도달이 아니라 비용 차감 후 net +3% 도달 여부를 계산한다.
- 함께 저장할 값: 최초 도달 분, 1~5분 중 가장 높은 `min_bid` 기준 순수익률,
  `t+5` timeout 순수익률
- 확률 학습 target은 `target_net3_by_1m`부터 `target_net3_by_5m`까지 누적으로 저장한다.
- 같은 진입 분의 `min_bid`에는 진입 직후 bid가 포함되므로 보수적 1분 target은 구조적으로
  0이다. 실제 probability head는 2·3·4·5분 discrete-time hazard를 학습한다.
- 동일 급등이 만든 인접 positive도 각 시점의 진입 ask와 미래 quote로 독립 검증된 경우
  전부 유지한다. episode ID는 진단용이며 제거·역가중에 사용하지 않는다.
- 다음 봉 하락 여부, 하락봉 선행 조건, TP_FIRST/SL_FIRST 같은 조건은 사용하지 않는다.
- 첫 단계에는 별도 SL label을 두지 않는다.

백테스트에서는 신호 발생 시 즉시 진입한다. TP label은 미래 완성봉의 `min_bid`로도
net +3%가 유지되는 첫 분에만 성립하며, 순간적인 고가 터치는 인정하지 않는다.
5분 안에 달성하지 못하면 `t+5`의 마지막 bid로 청산한다.

## 평가 원칙

- 날짜 기준 walk-forward 방식으로만 모델을 선택한다.
- 같은 날짜의 인접 30봉 중복이 랜덤 split을 통해 Train/Test에 섞이지 않게 한다.
- 핵심 지표는 거래 단위 평균·중앙 순수익률, 승률, 총 순손익, 날짜별 일관성,
  최대 낙폭이다.
- PR-AUC는 보조 분류 지표이며 실제 체결 순수익보다 우선하지 않는다.
- TP 확률 head에는 class weight를 적용하지 않고 자연 발생 비율의 proper loss를 사용한다.
- 확률은 날짜 기반 OOF에서 Brier score, calibration error와 reliability를 확인한 뒤 보정한다.
- 데이터 split, 모델, threshold를 확정하기 전까지 별도의 최종 Test 날짜를 소비하지 않는다.

## AI 진입·청산 확장 원칙

- 진입 AI는 완료된 `t`봉까지의 정보로 `BUY/SKIP`을 판단한다.
- 청산 AI는 진입 후 매분 봉이 완료될 때마다 `HOLD/EXIT`를 다시 판단한다.
- 청산 입력에는 해당 시점까지 갱신한 30봉 feature, 진입 ask, 현재 executable bid,
  미실현 순수익률, 보유 시간이 포함된다.
- 청산 target은 미래를 feature로 쓰지 않고, 과거 데이터에서 비용 차감 후 위험조정
  순수익을 최대화한 행동을 supervised optimal-stopping label로 만든다.
- 데이터가 충분해지기 전에는 offline RL보다 supervised policy를 먼저 사용한다.
- AI 청산을 사용하더라도 최대 5분 강제 청산 규칙은 안전장치로 유지한다.

## 현재 상태

- 현재 새 baseline은 `notebooks/01_multi_horizon_tp_probability_preprocessing.ipynb`이다.
- 새 전처리 artifact는
  `../../results/preprocessing/scalp_30m_ohlcv_net3_multihorizon_5m_v2_*`에 있다.
- 72,181개 표본과 기존 2,847개 positive를 모두 유지했다. positive는 1,128개 연속
  episode이며, 여러 행 episode에 속한 positive 2,420개도 제거하지 않았다.
- 누적 TP 비율은 2분 0.57%, 3분 1.63%, 4분 2.80%, 5분 3.94%다.
- 새 확률 모델은 2·3·4·5분 hazard를 출력하고 누적 TP 확률을 계산해야 한다.
- `notebooks/02_multihorizon_hazard_probability_tcn.ipynb`에서 class weight 없이 50,648개
  parameter TCN으로 hazard와 분별 TP 유지 확률을 학습했다.
- OOF 5분 TP PR-AUC는 0.1259(양성률 4.32%), 마지막 날짜 Test PR-AUC는
  0.0591(양성률 1.98%)이다. Test ROC-AUC는 0.7886이다.
- OOF에서는 raw 5분 확률 평균 4.15%와 실제 4.32%가 가까웠고 Platt 보정이 Brier를
  개선하지 못해 raw 확률을 선택했다. Test 평균 확률은 2.70%로 실제 1.98%보다 높았다.
- 확률이 높을수록 TP 빈도는 증가했지만 실패 시 손실도 커졌다. OOF 최상위 decile은
  TP 14.64%인데 정책 평균 수익률은 -1.83%, Test 최상위 decile은 TP 6.66%인데
  평균 -2.16%였다.
- 모든 절대 확률 threshold의 OOF 손익이 음수다. 연구용 1% threshold의 Test 결과도
  686건, TP precision 3.94%, 평균 -1.01%, 총 -$696.26이므로 checkpoint의
  `authorized_probability_threshold`는 `None`, `deployment_eligible=false`다.
- 아래 v1과 모델 02~06 결과는 실패 원인 비교용 기록이며 새 pipeline 학습에는 사용하지 않는다.

- 기존 60봉 OHLC-only 전처리, 4-class barrier, 동적 SL, ModernTCN/TimeMixer 실험과
  관련 artifact는 모두 폐기했다.
- 원본 `*_enriched.csv`에는 OHLCV, 실제 VWAP, quote 정보가 존재함을 확인했다.
- `notebooks/01_scalping_ohlcv_preprocessing.ipynb`에서 30봉 feature와 보수적
  `min_bid` net +3% label을 생성한다.
- 현재 전처리 결과는 9거래일, 111종목, 72,181표본이며 positive는 2,847개(3.94%)다.
- 전체 39개 feature 중 절대 `log_volume`은 ablation 전용이다. 기본 학습은 상대 거래량,
  상대 거래대금, 공격적 순매수 거래대금 등 38개 feature를 사용한다.
- 전처리 artifact는 `../../results/preprocessing/scalp_30m_ohlcv_net3_minbid_5m_v1_*`에 있다.
- `notebooks/02_compact_mptsnet_weighted_oof.ipynb`에서 AAAI 2025 MPTSNet의 소형화
  구조와 Train fold 기준 positive class weight를 평가했다.
- 마지막 거래일 이전 3일 expanding OOF PR-AUC는 0.1185(양성률 0.0405), 마지막 날짜
  Test PR-AUC는 0.0495(양성률 0.0198)였다.
- OOF의 모든 진입 비율에서 순손익이 음수였다. 후보 Test 운용도 293건, 승률 30.72%,
  평균 순수익률 -1.43%, 총손익 -$419.68이므로 checkpoint의
  `deployment_eligible`은 `false`다.
- 모델 score는 +3% 발생 가능성과 함께 timeout downside도 높이는 경향이 있었다.
  모델 크기 확대가 아니라 보수적 +3% label을 유지하면서 false positive의 손실 크기를
  반영해야 한다.
- `notebooks/03_multitask_tp_timeout_downside.ipynb`에서 TP, timeout return, 미래 5분
  downside를 함께 학습했다. Test downside Spearman은 0.697이었지만 Test 진입 6건,
  TP 0건, 총손익 -$11.67로 `deployment_eligible=false`다.
- OOF 후보는 36건, 승률 58.33%, 평균 +0.41%, 총 +$14.58이었지만 위험조정수익률은
  -0.06%였고 Test에 재현되지 않았다.
- 실제 OOF TP 표본의 평균 downside는 -1.37%였지만 모델은 -5.11%로 예측했다.
  회귀 head가 96%를 차지하는 음성 표본에 지배되어 TP와 downside 관계를 반대로 배웠다.
- `notebooks/04_conditional_buy_model_oof.ipynb`에서 `P(TP)`,
  `E(timeout | no TP)`, `Q20(downside | no TP)`를 분리해 학습했다. Test TP PR-AUC는
  0.0481, Test 정책 180건의 평균 수익률은 -1.62%, 총손익은 -$291.09로
  `deployment_eligible=false`다.
- `notebooks/05_sell_state_dataset_from_buy_oof.ipynb`는 날짜 walk-forward 매수 OOF
  신호에만 고정 5분 cooldown을 적용해 매도 학습 표본을 만든다. 진입 후 완성되는 각
  1분봉에서 `HOLD/EXIT` 상태를 만들며 OOF 6,395개, Test 2,670개 state가 생성됐다.
- `notebooks/06_sell_optimal_stopping_and_sequential_backtest.ipynb`는 HOLD 분류,
  미래 청산 advantage, downside를 함께 학습한다. Test HOLD PR-AUC는 0.5914,
  advantage/downside Spearman은 각각 0.2251/0.3278이다.
- 매수→보유→매도 Test sequential 결과는 318건, 승률 28.30%, 평균 -0.98%,
  총손익 -$312.20, profit factor 0.364로 `deployment_eligible=false`다.
- 동일 318개 진입의 미래 최적 청산은 평균 +0.67%, 총 +$212.33이므로 청산 기회 자체는
  존재하지만 현재 매도 모델이 이를 식별하지 못했다. 현 checkpoint는 실거래에 사용하지 않는다.
