# US 1분봉 초단타 진입·청산 가치 모델

현재 primary V8은 희소한 “3분 +3% 사건” 분류 대신, 매 시각 종목별
**1·2·3·5분 실행가능 수익 표면을 예측하고 상대 순위를 매기는 진입 모델**과
진입 후 다음 1분을 더 보유할 가치를 예측하는 청산 모델을 결합한다.

모델 입력은 최근 15개 1분 OHLC에서 만든 12개 순서형 특징과 현재 시점의 24개
OHLC context로 제한한다. 거래량, 거래대금, VWAP, 호가·체결 흐름은 입력하지 않는다.
`last_ask_t`와 미래 `last_bid`는 라벨 및 시장가 즉시체결 백테스트에만 사용한다.
V5~V7은 비교·실패 진단 baseline으로 보존한다.

## 데이터와 현재 split

- 원본: `../../data/stock_data/raw/session_*/*_enriched.csv`
- Train: 2026-07-07 ~ 2026-07-10, 4세션
- Validation: 2026-07-13 ~ 2026-07-14, 2세션
- V1 고정 Test: 2026-07-15 ~ 2026-07-17, 3세션
- Reduced V2 Reference: 2026-07-15 ~ 2026-07-17, 3세션
- Reduced V2 최종 Test: 아직 없음. 새로운 미래 날짜 필요
- Fixed V3 실험: Train 2026-07-07~15, Test 2026-07-16~17
- Quote Surge V5/V6: Train 2026-07-07~15, Test 2026-07-16~17, Validation 없음
- Value Ranking V8: Train 2026-07-07~15, OOF 2026-07-10~15, 진단 Test 2026-07-16~17
- 환경: conda `urban`, PyTorch

원본은 읽기 전용이며 파생 artifact는 `processed`, `models`, `backtests`에 분리해
저장한다.

## Notebook 순서

1. `01_data_universe_audit.ipynb`: 원본·universe·gap·selection bias 감사
2. `02_feature_pipeline.ipynb`: 공급자 독립 OHLC feature와 60봉 sequence 생성
3. `03_fee_barrier_dual_path.ipynb`: Legacy 지정가·NO_FILL 비교용 라벨
4. `04_gradient_boosting_baseline.ipynb`: Torch 분류 및 순수익률 Huber 회귀
5. `05_sequential_backtest.ipynb`: 시간순 포지션·자금 제약과 세션별 threshold 검증
6. `06_walk_forward_oof_expected_value.ipynb`: 날짜 walk-forward OOF와 4-outcome 기대수익 최적화
7. `07_reduced_walk_forward_model.ipynb`: feature·모델 용량 축소와 OOF 순차 복구
8. `08_fixed_train_test_80_20.ipynb`: Legacy 지정가의 validation 없는 7:2 실험
9. `09_immediate_fill_rebaseline.ipynb`: 10분 즉시체결 3-outcome PnL baseline
10. `10_quote_surge_3m_binary.ipynb`: 하락봉 급등 탐지와 weighted 40-epoch V6 비교
11. `11_event_centric_multihorizon_entry.ipynb`: 모든 봉·최근 10봉 sequence·다중 horizon V7
12. `12_forward_return_surface.ipynb`: current ask→future bid 1·2·3·5분 수익 표면
13. `13_cross_sectional_entry_ranker.ipynb`: 날짜 OOF 진입 효용·종목 순위 모델
14. `14_exit_value_model.ipynb`: 보유 중 다음 1분 continuation value 모델
15. `15_end_to_end_walk_forward_backtest.ipynb`: OOF 정책 선택과 순차 수익률 백테스트

현재 V2 feature는 학습 날짜가 적을 때 생기는 요일 암기를 막기 위해
`day_of_week_sin/cos`를 제외한다. 1분 간격 window는 서로 겹치므로 행 수를 독립
표본 수로 간주하지 않으며, 10분 신호 cluster와 날짜별 성과를 함께 확인한다.
V5는 같은 이유로 3분 신호 cluster를 별도로 확인한다.

06 실험은 개발 구간 6일에서 다음 expanding OOF를 사용한다.

```text
fit 7/7~8,  inner validation 7/9,  OOF 7/10
fit 7/7~9,  inner validation 7/10, OOF 7/13
fit 7/7~10, inner validation 7/13, OOF 7/14
final fit 7/7~14, fixed test 7/15~17
```

각 fold는 `NO_FILL / TP / SL / TIMEOUT` 확률을 temperature calibration한 뒤,
과거 데이터에서만 계산한 가격대별 outcome payoff를 곱해 예상 순수익률을 만든다.
이는 지정가 V1/V2 및 09의 10분 PnL 비교 설명이며 현재 탐지 목표는 10의 V5/V6다.

## 현재 상태

V8 데이터셋은 9개 세션 48,991개 판단 시점이다. 무조건 진입의 3분 순수익률은 평균
`-0.777%`, 중앙값 `-0.701%`로 스프레드·수수료를 넘기 어렵다. 진입 랭커는 4개
날짜 walk-forward OOF에서 epoch 20을 선택했고, OOF 효용 Spearman `0.204`, 시각별
cross-sectional rank IC `0.094`를 기록했다. 매 시각 상위 후보의 효용 lift는
`+0.122%p`였지만 평균 효용 자체는 `-0.633%`로 음수였다.

청산 continuation 모델은 epoch 16을 선택했다. OOF 다음 1분 수익률 Spearman은
`0.083`; HOLD 구간의 다음 1분 평균은 `+0.010%`, SELL 구간은 `-0.024%`였다.
방향 분리는 약하게 존재하지만 날짜별 안정성이 충분하지 않다.

OOF 4일 모두 거래가 존재하도록 제한해 선택한 정책은 `top_k=1`, 진입 효용
`q50>=-0.5%`, `q10>=-2%`다. OOF 765건 평균 순수익률은 `-0.499%`, 양수 날짜는
0/4였고, 진단 Test 943건도 평균 `-0.393%`, 양수 날짜 0/2였다. 따라서 V8 배포
상태는 `FAIL_OOF`다. 7/16~17은 이미 반복 사용한 non-pristine 날짜이므로 Test는
선택에 사용하지 않았고 최종 승인 근거도 아니다.

현재 Quote Surge V5는 하락봉 후보 22,700개를 사용한다. 시간순 Train 7일은
18,874개(`positive=883`, 4.68%), Test 2일은 3,826개(`positive=124`, 3.24%)다.
학습에는 종목별 3분 event bucket에서 가장 이른 행만 뽑은 11,630개를 사용하고,
추론과 평가는 모든 후보 행에서 수행했다.

Train/Test PR-AUC는 `0.209/0.121`, Test ROC-AUC는 `0.788`이다. Test PR-AUC는
무작위 기준 3.24%의 약 `3.73배`다. Test 자체 점수 상위 1%는 39건 중 10건
(`precision 25.64%`), 상위 5%는 192건 중 27건(`precision 14.06%`)이 양성이다.
Train 점수의 95% 분위수 `0.8014`를 고정 threshold로 적용하면 Test 신호는 58건,
양성 적중은 14건, precision은 `24.14%`, recall은 `11.29%`다. 이 수치는 ‘실현수익 TP’가
아니라 정의한 3분 +3% 사건 적중을 뜻한다.

양성 class weight를 사용했으므로 `0.8014`는 보정된 80.14% 확률이 아닌 **모델
점수**다. 날짜별 고정-threshold 신호도 7/16은 50건, 7/17은 8건으로 차이가 크다.
게다가 7/16~17은 이전 실험에서 이미 본 날짜이므로 `test_is_pristine=false`다.
따라서 현재 결론은 “급등 후보 농축 신호를 학습함”이며 배포나 수익성 검증 통과가
아니다. 구조와 threshold를 고정한 뒤 새로운 미래 날짜에서 재평가해야 한다.

요청에 따른 Weighted 40-epoch V6는 balanced weight `22.79`에 `1.5배`를 적용한
`34.18`과 40 epoch를 사용했다. Train PR-AUC는 `0.209→0.219`로 증가했지만 Test
PR-AUC는 `0.121→0.115`로 감소했다. 같은 Train 95% score 정책의 Test는 58건 중
11건만 적중해 precision/recall이 `18.97%/8.87%`로 V5의 `24.14%/11.29%`보다
나빠졌다. 이는 Train만 개선된 과적합 방향이므로 V6를 성능 개선 모델로 채택하지
않고 weight·epoch ablation으로 보존한다.

Event-centric V7은 하락봉 필터를 제거해 모든 봉 49,664개를 후보로 보존했다.
Train/Test는 각각 `41,026/8,638`개이고 +3%/3분 양성은 `1,895/267`개다. 입력은
최근 10봉 × 12개 순서형 OHLC 특징과 24개 장기 context이며, 모델은 5,285
parameters다. `+1%/1분`, `+2%/2분`, `+3%/3분`과 미래 최대·최소 bid 수익률을
동시에 학습했다. 양성·hard-negative event는 cluster inverse weight로 중복을 줄이고,
나머지 음성은 종목별 최소 3분 간격으로 sampling했다. 최종 Train 학습 표본은
17,401개이고 class weight는 target별 최대 5로 제한했다.

Train 날짜 walk-forward OOF만 사용해 epoch 12와 `+1%/1분` head를 +3%/3분의
선행 decision score로 선택했다. 전체 봉 OOF/Test PR-AUC는 `0.166/0.095`, Test
PR lift는 `3.08배`다. OOF 상위 5% score threshold를 고정한 Test는 신호 141개 중
11개 적중으로 precision/recall이 `7.80%/4.12%`다. 기존 전략과 같은 하락봉
subgroup도 Test PR-AUC `0.088`, precision/recall `7.83%/7.26%`로 V5보다 낮다.

따라서 최근 10봉 순서 보존, 모든 봉 후보, 다중 horizon 및 event weight도 현재의
날짜 regime 차이를 해결하지 못했다. V7은 실패한 진단 baseline으로 보존한다. 유효한
진입 신호가 없으므로 그 신호를 전제로 한 Exit AI 학습은 진행하지 않았으며, 다음
필수 입력은 모델 변경이 아니라 더 많은 독립 거래일이다.

### 이전 10분 PnL baseline

9개 세션으로 재학습한 V2 two-stage 모델은 test ROC-AUC 약 `0.808`, PR-AUC 약
`0.144`로 TP 신호 순위는 학습했다. 그러나 validation 두 날짜가 각각 양수여야 하는
threshold gate를 direct, two-stage, expected-return 전략 모두 통과하지 못했다.
따라서 현재 baseline의 배포 상태는 `FAIL`이며, test에 맞춘 threshold 재조정은 하지
않는다.

Walk-forward 4-outcome V1 결과는 OOF expected-return Spearman `0.150`, test
`0.160`으로 기존 two-stage 수익률 상관의 음수 문제를 개선했다. 하지만 OOF에서
선택한 예상수익 threshold `0.1508%`는 OOF 3일 중 2일만 양수였고 OOF 평균 체결
순수익률 `-0.322%`, test `-1.650%`였다. 따라서 threshold 상태는
`NO_VALID_THRESHOLD`, 배포 상태는 계속 `FAIL`이다.

과적합 완화 Reduced V2는 모든 base feature에 같은 5개 집계를 적용하지 않는다.
캔들·변동성·가격 위치별로 필요한 집계만 골라 core 82개를 구성하고, 동일한 날짜
walk-forward OOF에서 trend, momentum, time을 순서대로 하나씩 추가했다. 결과적으로
trend 8개만 채택되어 최종 입력은 90개이며, momentum과 time은 OOF 기대수익 순위와
log-loss를 악화시켜 제외됐다. MLP는 `450→128→64→4` 66,244 parameters에서
`90→32→4` 3,044 parameters로 축소됐다.

Reduced V2 학습은 세션·종목·10분 bucket마다 가장 이른 한 행만 사용하고 각 날짜의
총 loss weight를 같게 둔다. Fold별 학습 표본은 `9,668→1,017`, `16,367→1,712`,
`23,226→2,430`으로 줄었다. 최종 OOF expected-return Spearman은 `0.126`, OOF
TP PR-AUC는 `0.164`다. 선택된 진단용 threshold `0.4426%`의 OOF 평균 체결
순수익률은 `-0.118%`이고 3일 중 1일만 양수여서 `NO_VALID_THRESHOLD`다.
따라서 Reduced V2도 배포 상태는 `FAIL`이다.

7/15~17은 과적합 원인 진단과 Reduced V2 설계에 이미 참고했으므로 V2에서는
`reference_test`로만 기록한다. 이 구간 성능으로 모델을 승인하지 않으며, 이후 구조를
고정하고 새로운 미래 날짜를 확보해야 최종 test가 성립한다.

사용자 요청에 따른 Fixed V3는 9개 날짜를 쪼개지 않고 앞 7일을 Train, 뒤 2일을
Test로 사용했다. 이는 행 기준 `82.58:17.42`이며 validation, early stopping,
temperature calibration을 사용하지 않고 15 epoch를 고정했다. Train/Test
expected-return Spearman은 각각 `0.156/0.130`, TP PR-AUC는 `0.172/0.122`다.
Train 점수 99% 분위수가 음수여서 예상수익 0%를 threshold로 적용한 결과 Train은
68건 체결 평균 `+0.432%`, Test는 4건 체결 평균 `-1.891%`였다. Test 체결 수가 너무
적고 7/16~17은 이미 본 날짜이므로 이 결과는 실험용이며 배포 근거가 아니다.

즉시체결 Immediate V4에서는 진입가를 `close_t`로 바꾸고 `NO_FILL`을 완전히
제거했다. 전체 49,084개 라벨 중 dual-path 확정 표본은 49,014개이며 분포는
`TP 5,455 / SL 11,247 / TIMEOUT 32,312`다. 동일한 7 Train / 2 Test에서
expected-return Spearman은 Train `0.143`, Test `0.148`, TP PR-AUC는
`0.255/0.201`이다. Train 점수 99% 분위수 threshold `0.0546%`를 고정한 Test는
신호 25개, 실제 실행 21건, `TP 4 / SL 12 / TIMEOUT 5`, 평균 순수익률
`-0.885%`였다. 체결 가정 오류는 수정됐지만 매수 신호 수익성은 여전히 검증 실패다.

## 전체 재실행

```bash
for notebook in notebooks/01_data_universe_audit.ipynb \
  notebooks/02_feature_pipeline.ipynb \
  notebooks/03_fee_barrier_dual_path.ipynb \
  notebooks/04_gradient_boosting_baseline.ipynb \
  notebooks/05_sequential_backtest.ipynb \
  notebooks/06_walk_forward_oof_expected_value.ipynb \
  notebooks/07_reduced_walk_forward_model.ipynb \
  notebooks/08_fixed_train_test_80_20.ipynb \
  notebooks/09_immediate_fill_rebaseline.ipynb \
  notebooks/10_quote_surge_3m_binary.ipynb \
  notebooks/11_event_centric_multihorizon_entry.ipynb \
  notebooks/12_forward_return_surface.ipynb \
  notebooks/13_cross_sectional_entry_ranker.ipynb \
  notebooks/14_exit_value_model.ipynb \
  notebooks/15_end_to_end_walk_forward_backtest.ipynb; do
  /home/user/anaconda3/envs/urban/bin/jupyter nbconvert \
    --to notebook --execute --inplace "$notebook" \
    --ExecutePreprocessor.kernel_name=urban \
    --ExecutePreprocessor.timeout=1800
done
```

V1 walk-forward 설정과 구현은 `configs/walk_forward_oof.yaml`,
`src/walk_forward_oof.py`에 있다. Reduced V2는
`configs/reduced_walk_forward_oof.yaml`, `src/reduced_walk_forward.py`를 사용한다.
Validation 없는 Fixed V3는 `configs/fixed_train_test_80_20.yaml`,
`src/fixed_train_test.py`에 있다.
10분 즉시체결 라벨은 `src/immediate_fill_labeling.py`, 재학습 설정은
`configs/fixed_train_test_immediate_fill.yaml`에 있다.
현재 3분 급등 탐지는 `src/quote_surge_binary.py`, 설정은
`configs/quote_surge_3m_binary.yaml`에 있다.
Event-centric V7은 `src/event_centric_entry.py`, 설정은
`configs/event_centric_entry.yaml`에 있다.
현재 Value Ranking V8은 `src/value_ranking_strategy.py`, 설정은
`configs/value_ranking_strategy.yaml`에 있다.

세부 모델·라벨·수수료·백테스트 원칙은 `AGENT.md`를 따른다.
