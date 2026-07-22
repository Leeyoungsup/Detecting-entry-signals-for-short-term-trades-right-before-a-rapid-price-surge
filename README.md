# US 1분봉 초단타 스캘핑 AI

미국 주식 1분봉으로 완료된 `t`봉 직후 진입했을 때, 앞으로 1~5분 안에 비용 차감 후
3% 이상 수익을 낼 수 있는 시점을 예측한다.

## 새 기준

- 입력: 연속된 완성 1분봉 30개
- 원본 feature: OHLC, 거래량, 거래 건수, VWAP
- 파생 feature: EMA, 당일 VWAP, 당일 전고점·전저점, 상대 거래량 등
- 판단: `t`봉 종료 직후
- 진입 proxy: `first_ask[t+1]`
- positive: 미래 1~5분 중 완성봉의 `min_bid`로도 비용 차감 후 +3% 이상 유지
- 다음 하락봉이나 TP/SL 4-class 조건은 사용하지 않음

시간 정렬과 label의 상세 기준은 `AGENT.md`를 따른다.

## 경로

```text
원본 데이터: ../../data/stock_data/raw
모델:        ../../model
결과:        ../../results
가상환경:    conda urban
```

## 현재 산출물

현재 다시 시작한 확률 pipeline의 기준은 다음과 같다.

- notebook: `notebooks/01_multi_horizon_tp_probability_preprocessing.ipynb`
- artifact: `../../results/preprocessing/scalp_30m_ohlcv_net3_multihorizon_5m_v2_*`
- sequence: 72,181개 × 30봉 × 39 feature; 기본 입력은 38 feature
- 전체 5분 positive: 2,847개(3.94%), 전부 유지
- positive episode: 1,128개
- 누적 TP: 2분 413개(0.57%), 3분 1,179개(1.63%),
  4분 2,023개(2.80%), 5분 2,847개(3.94%)
- probability model target: 2·3·4·5분 discrete-time hazard
- 1분 target은 같은 진입 분 전체 `min_bid`를 사용하는 보수적 정의상 0건이므로 감사용으로만 저장

기존 `01_scalping_ohlcv_preprocessing.ipynb`와 모델 02~06은 실패 원인 비교용 기록이다.
새 확률 pipeline에는 사용하지 않는다.

## Multi-horizon 확률 모델

- notebook: `notebooks/02_multihorizon_hazard_probability_tcn.ipynb`
- model: `../../model/buy_multihorizon_hazard_tcn_v1.pt`
- 구조: 50,648 parameter probability TCN
- loss: class weight 없는 2·3·4·5분 hazard BCE + 분별 TP 유지 보조 BCE
- OOF 5분: 양성률 4.32%, PR-AUC 0.1259
- Test 5분: 양성률 1.98%, PR-AUC 0.0591, ROC-AUC 0.7886
- 선택 확률: OOF Brier가 더 좋은 raw hazard probability
- Test 평균 확률 2.70% / 실제 TP 1.98%
- 연구용 1% threshold Test: 686건, TP precision 3.94%, 평균 -1.01%, 총 -$696.26
- 결론: TP 확률 순위는 학습했지만 고확률 구간의 downside가 더 커 모든 threshold가 손실
- `authorized_probability_threshold=None`, `deployment_eligible=false`

- 전처리 notebook: `notebooks/01_scalping_ohlcv_preprocessing.ipynb`
- 전처리 결과: `../../results/preprocessing/scalp_30m_ohlcv_net3_minbid_5m_v1_*`
- 표본: 9거래일, 111종목, 72,181개
- positive: 2,847개(3.94%)
- sequence: 30봉 × 39 feature; 기본 모델 입력은 절대 거래량을 제외한 38 feature

## 첫 모델 평가

- notebook: `notebooks/02_compact_mptsnet_weighted_oof.ipynb`
- 모델: AAAI 2025 MPTSNet 기반 430,913 parameter compact classifier
- loss: Train fold에서 계산한 positive weight를 적용한 BCE
- OOF PR-AUC: 0.1185 / OOF 양성률: 0.0405
- Test PR-AUC: 0.0495 / Test 양성률: 0.0198
- 후보 Test 거래: 293건, 승률 30.72%, 평균 -1.43%, 총 -$419.68
- 결론: 모든 OOF threshold가 손실이므로 `deployment_eligible=false`

## Multi-task 평가

- notebook: `notebooks/03_multitask_tp_timeout_downside.ipynb`
- heads: TP 확률, 5분 timeout 수익률, 미래 5분 downside 20% quantile
- Test downside Spearman: 0.697
- OOF 후보: 36건, 승률 58.33%, 평균 +0.41%, 총 +$14.58
- 최종 Test: 6건, TP 0건, 승률 50%, 평균 -1.95%, 총 -$11.67
- 결론: 거래 수와 총손실은 줄었지만 OOF edge가 Test에 재현되지 않아
  `deployment_eligible=false`

## 조건부 매수 + AI 청산 평가

- 조건부 매수 notebook: `notebooks/04_conditional_buy_model_oof.ipynb`
- 매수 OOF 기반 매도 표본: `notebooks/05_sell_state_dataset_from_buy_oof.ipynb`
- optimal-stopping/결합 백테스트: `notebooks/06_sell_optimal_stopping_and_sequential_backtest.ipynb`
- 매수 head: `P(TP)`, `E(timeout | no TP)`, `Q20(downside | no TP)`
- 매도 head: `HOLD`, 미래 청산 advantage, 미래 downside
- Test 매도 state: HOLD PR-AUC 0.5914, advantage Spearman 0.2251,
  downside Spearman 0.3278
- 결합 Test: 318건, 승률 28.30%, 평균 -0.98%, 총 -$312.20,
  profit factor 0.364
- 동일 진입 318건의 미래 최적 청산 진단: 승률 50.31%, 평균 +0.67%, 총 +$212.33

결론은 매도 가능한 가격 경로는 일부 존재하지만 현재 모델이 적절한 청산 분을 일반화하지
못했다는 것이다. 두 checkpoint 모두 `deployment_eligible=false`이며 실거래에 사용하지 않는다.
