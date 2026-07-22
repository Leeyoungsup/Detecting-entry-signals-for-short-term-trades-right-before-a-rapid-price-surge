# US 1분봉 초단타 스캘핑 AI

미국 주식 1분봉으로 완료된 `t`봉 직후 진입했을 때, 앞으로 1~5분 안에 비용 차감 후
3% 이상 수익을 낼 수 있는 시점을 예측한다.

## 새 기준

- 입력: 연속된 완성 1분봉 30개
- 원본 feature: OHLC, 거래량, 거래 건수, VWAP
- 파생 feature: EMA, 당일 VWAP, 당일 전고점·전저점, 상대 거래량 등
- 판단: `t`봉 종료 직후
- 진입 proxy: `first_ask[t+1]`
- positive: 미래 1~5분 executable bid에서 비용 차감 후 +3% 이상 도달
- 다음 하락봉이나 TP/SL 4-class 조건은 사용하지 않음

시간 정렬과 label의 상세 기준은 `AGENT.md`를 따른다.

## 경로

```text
원본 데이터: ../../data/stock_data/raw
모델:        ../../model
결과:        ../../results
가상환경:    conda urban
```

현재 기존 실험은 모두 정리된 상태다. 다음 작업은 새
`01_scalping_ohlcv_data_audit.ipynb`부터 시작한다.
