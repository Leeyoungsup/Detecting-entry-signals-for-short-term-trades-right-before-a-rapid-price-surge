# US 1분봉 초단타 매수·매도 타이밍 AI

미국 주식 1분봉 데이터로 초단타 매수·매도 타이밍을 판단하는 AI 모델을 개발한다.
세부 목표와 경로는 `AGENT.md`에 기록한다.

## 현재 단계

현재는 **연속 60봉 OHLC 전처리와 +3% 라벨 생성 단계**다.

구현된 확인 항목:

- `*_enriched.csv` 파일 및 세션 탐색
- 파일별 컬럼 스키마 확인
- `symbol`, `timestamp_utc`, OHLC dtype 정리
- 필수값 결측 확인
- OHLC 가격 관계 확인
- `symbol + timestamp_utc` 중복 확인
- 1분 초과 데이터 간격 집계

데이터 간격이 발견되어도 현재 단계에서는 행을 제거하거나 보간하지 않는다.

두 번째 노트북에서는 gap을 보간하지 않고 연속된 구간만 사용해 60봉 sequence를 만든다.
`t+1 open` 진입 기준으로 미래 1·3·5분 +3% 도달 여부와 MFE·MAE를 생성한다.

## 경로

```text
데이터:  ../../data/stock_data
모델:    ../../model
결과:    ../../results
```

## 실행

conda `urban` 환경에서 다음 명령을 실행한다.

노트북에서 conda `urban` 커널을 선택하고 번호 순서대로 실행한다.

## 구조

```text
notebooks/01_data_read_check.ipynb   데이터 경로, 적재, 기본 확인
notebooks/02_ohlc_60m_preprocessing.ipynb   60봉 특징과 +3%·MFE·MAE 라벨 생성
notebooks/03_random_tick_entry_sensitivity.ipynb   t+1 low~high 랜덤 진입 라벨 비교
notebooks/04_date_split_and_model_research.ipynb   날짜 split 확정과 모델 조사
```

02의 artifact는 `../../results/preprocessing/ohlc_60m_tp3pct_v1_*`에 저장된다.
