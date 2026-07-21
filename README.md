# US 1분봉 초단타 매수·매도 타이밍 AI

미국 주식 1분봉 데이터로 초단타 매수·매도 타이밍을 판단하는 AI 모델을 개발한다.
세부 목표와 경로는 `AGENT.md`에 기록한다.

## 현재 단계

현재는 모델이나 라벨을 만들기 전의 **데이터 읽기 및 기본 확인 단계**다.

구현된 확인 항목:

- `*_enriched.csv` 파일 및 세션 탐색
- 파일별 컬럼 스키마 확인
- `symbol`, `timestamp_utc`, OHLC dtype 정리
- 필수값 결측 확인
- OHLC 가격 관계 확인
- `symbol + timestamp_utc` 중복 확인
- 1분 초과 데이터 간격 집계

데이터 간격이 발견되어도 현재 단계에서는 행을 제거하거나 보간하지 않는다.

## 경로

```text
데이터:  ../../data/stock_data
모델:    ../../model
결과:    ../../results
```

## 실행

conda `urban` 환경에서 다음 명령을 실행한다.

첫 단계는 `notebooks/01_data_read_check.ipynb`에서 진행한다. conda `urban` 커널을
선택하고 셀을 위에서부터 실행한다. 이 노트북은 원본을 읽고 화면에 요약만 출력하며
전처리 결과 파일은 아직 저장하지 않는다.

## 구조

```text
notebooks/01_data_read_check.ipynb   데이터 경로, 적재, 기본 확인
```
