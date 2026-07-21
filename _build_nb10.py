from pathlib import Path

import nbformat as nbf


ROOT = Path.cwd()
OUT = ROOT / "notebooks" / "10_final_model_selection_100usd_backtest.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "urban", "language": "python", "name": "urban"},
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.12.12",
    },
}

cells = []

cells.append(
    nbf.v4.new_markdown_cell(
        """# 10. 최종 모델 선정과 종목별 $100 순차 백테스트

지금까지 만든 모델 중 **최종 모델과 진입 threshold를 OOF 결과만으로 먼저 고정**한 뒤, OOF와 Test에서 종목별 신호당 $100를 투자했을 때의 수익률을 계산한다. Test 수익은 모델 선택에 사용하지 않는다.

고정 거래 규칙:

- 판단 시점: 완성된 `t`봉 직후
- 투자금: 체결 신호마다 $100, fractional share 허용, 복리 재투자 없음
- 동일 종목: 한 포지션이 열려 있는 동안 새 신호 무시
- 목표: 진입가 대비 +3%, 관찰 구간 `t+1~t+3`
- 목표 미도달: `t+3`에서 전량 청산
- 손절 규칙 없음
- Test는 여러 번 확인한 development holdout이며 pristine 최종 검증이 아님

체결 가정에 따른 왜곡을 분리하기 위해 세 시나리오를 함께 계산한다.

1. `ohlc_proxy`: `t+1 open` 진입, 미래 `high`가 +3%에 닿으면 목표가 청산, 아니면 `t+3 close`
2. `quote_max_bid`: `t last_ask` 즉시 매수, 미래 분봉의 `max_bid`가 +3%에 닿으면 목표가 청산, 아니면 `t+3 last_bid`
3. `quote_min_bid`: 같은 진입이지만 미래 분봉의 `min_bid`가 +3% 이상일 때만 익절로 보는 보수적 민감도

`quote_*`에는 bid-ask spread가 반영된다. 호가 잔량, 시장 충격, 브로커 수수료는 데이터가 없어 포함하지 않고 별도 비용 stress로 확인한다."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == 'notebooks':
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_ROOT = (PROJECT_ROOT / '../../data/stock_data').resolve()
MODEL_ROOT = (PROJECT_ROOT / '../../model').resolve()
RESULTS_ROOT = (PROJECT_ROOT / '../../results').resolve()
PREPROCESS_ROOT = RESULTS_ROOT / 'preprocessing'
EPISODE_ROOT = RESULTS_ROOT / 'evaluation' / 'episode_first_entry_v1'
TRAINING_ROOT = RESULTS_ROOT / 'training' / 'moderntcn_ohlc_60m_v1'
FINAL_ROOT = RESULTS_ROOT / 'backtest' / 'final_modern_tcn_100usd_v1'
FINAL_ROOT.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_ROOT / 'moderntcn_ohlc_60m_v1' / 'open_hard.pt'
STAKE_USD = 100.0
TP_RETURN = 0.03
HORIZON_MINUTES = 3
TOP_FRACTION = 0.05

# 2026-04-04 이후 SEC Section 31 rate와 2026 FINRA TAF. 매도 시에만 근사 적용한다.
SEC_RATE_PER_DOLLAR = 20.60 / 1_000_000
FINRA_TAF_PER_SHARE = 0.000195
FINRA_TAF_CAP = 9.79

pd.set_option('display.max_columns', 60)
pd.set_option('display.float_format', lambda value: f'{value:,.6f}')

print('project :', PROJECT_ROOT)
print('raw data:', DATA_ROOT / 'raw')
print('model   :', MODEL_PATH)
print('results :', FINAL_ROOT)"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 1. OOF만으로 최종 모델과 threshold 고정

우선순위는 급등 직전 **첫 진입 precision**이다. 동일 상위 5% OOF 기준에서 기존 hard-label 확률이 downside-aware 및 utility score보다 첫 진입 precision이 높았으므로 `ModernTCN open_hard`를 최종 모델로 고정한다. 이 셀의 선택 로직에는 Test 지표나 수익률을 넣지 않는다."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """episode_metrics = pd.read_parquet(EPISODE_ROOT / 'episode_metrics.parquet')
thresholds = pd.read_parquet(EPISODE_ROOT / 'oof_score_thresholds.parquet')

selection_evidence = (
    episode_metrics[
        episode_metrics['phase'].eq('OOF')
        & episode_metrics['target_name'].eq('original_tp3')
        & episode_metrics['top_fraction_oof'].eq(TOP_FRACTION)
    ][[
        'score_name', 'threshold', 'selected_rows', 'signal_clusters',
        'row_precision', 'first_entry_precision', 'episode_recall'
    ]]
    .sort_values(['first_entry_precision', 'episode_recall'], ascending=False)
    .reset_index(drop=True)
)

SELECTED_SCORE = 'original_hard_probability'
SELECTED_MODEL = 'ModernTCN open_hard'
selected_threshold_row = thresholds[
    thresholds['score_name'].eq(SELECTED_SCORE)
    & thresholds['top_fraction_oof'].eq(TOP_FRACTION)
]
assert len(selected_threshold_row) == 1
SELECTED_THRESHOLD = float(selected_threshold_row.iloc[0]['threshold'])
assert selection_evidence.iloc[0]['score_name'] == SELECTED_SCORE
assert MODEL_PATH.exists()

checkpoint_sha256 = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()

print(f'최종 모델: {SELECTED_MODEL}')
print(f'OOF 상위 {TOP_FRACTION:.0%} 고정 threshold: {SELECTED_THRESHOLD:.9f}')
print(f'checkpoint SHA256: {checkpoint_sha256}')
display(selection_evidence.style.format({
    'threshold': '{:.6f}', 'row_precision': '{:.2%}',
    'first_entry_precision': '{:.2%}', 'episode_recall': '{:.2%}'
}))"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 2. 고정 threshold의 첫 진입 신호와 원본 체결 가격 연결

09번 노트북에서 연속된 threshold 초과 신호를 하나의 signal cluster로 묶고 첫 행만 남겼다. 여기서는 해당 `sample_id`를 원본 CSV에 연결해 판단 시점의 마지막 ask와 이후 3분의 OHLC/bid를 복원한다."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """first_entries_all = pd.read_parquet(EPISODE_ROOT / 'first_entries.parquet')
signals = first_entries_all[
    first_entries_all['score_name'].eq(SELECTED_SCORE)
    & first_entries_all['target_name'].eq('original_tp3')
    & first_entries_all['top_fraction_oof'].eq(TOP_FRACTION)
].copy()

metadata = pd.read_parquet(
    PREPROCESS_ROOT / 'ohlc_60m_tp3pct_v1_metadata.parquet',
    columns=[
        'sample_id', 'session', 'symbol', 'source_file', 'run_id',
        'decision_timestamp', 'entry_timestamp', 'decision_close', 'entry_open',
        'mfe_3m', 'mae_3m', 'target_tp3_3m'
    ]
)

signals = signals.merge(
    metadata,
    on=['sample_id', 'session', 'symbol', 'run_id', 'decision_timestamp'],
    how='left', validate='one_to_one'
)
assert signals['source_file'].notna().all()
assert signals.groupby('phase').size().to_dict() == {'OOF': 602, 'Test': 86}

display(signals.groupby('phase').agg(
    signal_clusters=('sample_id', 'size'),
    symbols=('symbol', 'nunique'),
    sessions=('session', 'nunique'),
    score_min=('score', 'min'),
    score_median=('score', 'median'),
    score_max=('score', 'max'),
))"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """RAW_COLUMNS = [
    'timestamp_utc', 'open', 'high', 'low', 'close',
    'last_bid', 'last_ask', 'min_bid', 'max_bid'
]

def reconstruct_execution_prices(signal_frame):
    records = []
    for source_file, group in signal_frame.groupby('source_file', sort=False):
        raw_path = Path(source_file)
        if not raw_path.exists():
            candidate = DATA_ROOT / 'raw' / group.iloc[0]['session'] / raw_path.name
            raw_path = candidate
        raw = pd.read_csv(raw_path, usecols=RAW_COLUMNS)
        raw['timestamp_utc'] = pd.to_datetime(raw['timestamp_utc'], utc=True)
        positions = pd.Series(raw.index.to_numpy(), index=raw['timestamp_utc']).to_dict()

        for signal in group.itertuples(index=False):
            decision_ts = pd.Timestamp(signal.decision_timestamp)
            pos = positions.get(decision_ts)
            if pos is None or pos + HORIZON_MINUTES >= len(raw):
                raise ValueError(f'원본 미래 3분을 찾을 수 없음: {raw_path}, {decision_ts}')
            decision = raw.iloc[pos]
            future = raw.iloc[pos + 1: pos + HORIZON_MINUTES + 1].reset_index(drop=True)
            record = {
                'sample_id': signal.sample_id,
                'phase': signal.phase,
                'session': signal.session,
                'symbol': signal.symbol,
                'run_id': signal.run_id,
                'source_file': str(raw_path),
                'decision_timestamp': decision_ts,
                'score': float(signal.score),
                'label_target_tp3_3m': int(signal.target_tp3_3m),
                'label_mfe_3m': float(signal.mfe_3m),
                'label_mae_3m': float(signal.mae_3m),
                'metadata_entry_open': float(signal.entry_open),
                'decision_close': float(decision['close']),
                'current_last_bid': float(decision['last_bid']),
                'current_last_ask': float(decision['last_ask']),
            }
            for minute in range(1, HORIZON_MINUTES + 1):
                row = future.iloc[minute - 1]
                record[f'ts_{minute}'] = row['timestamp_utc']
                for column in ['open', 'high', 'low', 'close', 'last_bid', 'min_bid', 'max_bid']:
                    record[f'{column}_{minute}'] = float(row[column])
            records.append(record)
    return pd.DataFrame(records)

execution_prices = reconstruct_execution_prices(signals)
execution_prices = execution_prices.sort_values(
    ['phase', 'decision_timestamp', 'symbol', 'sample_id']
).reset_index(drop=True)

quote_columns = ['current_last_ask', 'last_bid_3'] + [
    f'{side}_{minute}'
    for side in ['min_bid', 'max_bid']
    for minute in range(1, HORIZON_MINUTES + 1)
]
price_quality = execution_prices.groupby('phase').agg(
    signals=('sample_id', 'size'),
    entry_open_match=('sample_id', lambda idx: True),
)
price_quality['entry_open_match_rate'] = execution_prices.groupby('phase').apply(
    lambda frame: np.isclose(frame['metadata_entry_open'], frame['open_1'], rtol=1e-7, atol=1e-9).mean(),
    include_groups=False,
)
price_quality['quote_complete_rate'] = execution_prices.groupby('phase').apply(
    lambda frame: np.isfinite(frame[quote_columns]).all(axis=1).mean(),
    include_groups=False,
)
price_quality['ask_vs_close_mean'] = execution_prices.groupby('phase').apply(
    lambda frame: (frame['current_last_ask'] / frame['decision_close']).mean(),
    include_groups=False,
)
price_quality['ask_vs_close_p95'] = execution_prices.groupby('phase').apply(
    lambda frame: (frame['current_last_ask'] / frame['decision_close']).quantile(0.95),
    include_groups=False,
)

assert price_quality['entry_open_match_rate'].eq(1.0).all()
assert price_quality['quote_complete_rate'].eq(1.0).all()
display(price_quality)"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 3. 시나리오별 주문 실행과 동일 종목 중복 제거

익절에 닿으면 목표가(+3%)에 체결된 것으로 제한해 favorable slippage를 주지 않는다. 규제성 매도 비용은 2026년 7월에 적용되는 SEC Section 31 rate와 FINRA TAF를 근사 반영한다. 실제 고객 전가 및 반올림 방식은 브로커마다 다를 수 있다."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """SCENARIOS = {
    'ohlc_proxy': {
        'entry_column': 'open_1',
        'hit_prefix': 'high',
        'timeout_column': 'close_3',
        'entry_timestamp_column': 'ts_1',
    },
    'quote_max_bid': {
        'entry_column': 'current_last_ask',
        'hit_prefix': 'max_bid',
        'timeout_column': 'last_bid_3',
        'entry_timestamp_column': 'decision_timestamp',
    },
    'quote_min_bid': {
        'entry_column': 'current_last_ask',
        'hit_prefix': 'min_bid',
        'timeout_column': 'last_bid_3',
        'entry_timestamp_column': 'decision_timestamp',
    },
}

def create_candidate_trades(price_frame, scenario_name, config):
    trades = price_frame.copy()
    trades['scenario'] = scenario_name
    trades['entry_timestamp'] = pd.to_datetime(trades[config['entry_timestamp_column']], utc=True)
    trades['entry_price'] = trades[config['entry_column']].astype(float)
    trades['target_price'] = trades['entry_price'] * (1.0 + TP_RETURN)

    hit_matrix = np.column_stack([
        trades[f\"{config['hit_prefix']}_{minute}\"].to_numpy(dtype=float)
        >= trades['target_price'].to_numpy(dtype=float)
        for minute in range(1, HORIZON_MINUTES + 1)
    ])
    any_hit = hit_matrix.any(axis=1)
    first_hit_minute = np.where(any_hit, hit_matrix.argmax(axis=1) + 1, HORIZON_MINUTES)
    trades['exit_reason'] = np.where(any_hit, 'TP', 'TIMEOUT')
    trades['holding_minutes'] = first_hit_minute
    trades['exit_price'] = np.where(
        any_hit, trades['target_price'], trades[config['timeout_column']]
    )
    exit_timestamps = []
    for row_index, minute in enumerate(first_hit_minute):
        exit_timestamps.append(trades.iloc[row_index][f'ts_{int(minute)}'])
    trades['exit_timestamp'] = pd.to_datetime(exit_timestamps, utc=True)

    valid = (
        np.isfinite(trades['entry_price']) & (trades['entry_price'] > 0)
        & np.isfinite(trades['exit_price']) & (trades['exit_price'] > 0)
    )
    if not valid.all():
        raise ValueError(f'{scenario_name}: invalid execution price {int((~valid).sum())} rows')
    return trades

def remove_overlapping_same_symbol(candidate_trades):
    accepted_parts = []
    skipped = 0
    group_columns = ['phase', 'session', 'symbol']
    for _, group in candidate_trades.groupby(group_columns, sort=False):
        group = group.sort_values(['entry_timestamp', 'score'], ascending=[True, False])
        last_exit = None
        accepted_indices = []
        for row in group.itertuples():
            if last_exit is not None and row.entry_timestamp < last_exit:
                skipped += 1
                continue
            accepted_indices.append(row.Index)
            last_exit = row.exit_timestamp
        accepted_parts.append(candidate_trades.loc[accepted_indices])
    accepted = pd.concat(accepted_parts, ignore_index=True)
    return accepted, skipped

def add_cash_returns(trades):
    result = trades.copy()
    result['stake_usd'] = STAKE_USD
    result['shares'] = STAKE_USD / result['entry_price']
    result['exit_notional_usd'] = result['shares'] * result['exit_price']
    result['gross_pnl_usd'] = result['exit_notional_usd'] - STAKE_USD
    result['gross_return'] = result['gross_pnl_usd'] / STAKE_USD
    result['sec_fee_usd'] = result['exit_notional_usd'] * SEC_RATE_PER_DOLLAR
    result['finra_taf_usd'] = np.minimum(
        result['shares'] * FINRA_TAF_PER_SHARE, FINRA_TAF_CAP
    )
    result['regulatory_fees_usd'] = result['sec_fee_usd'] + result['finra_taf_usd']
    result['net_pnl_usd'] = result['gross_pnl_usd'] - result['regulatory_fees_usd']
    result['net_return'] = result['net_pnl_usd'] / STAKE_USD
    return result

trade_parts = []
overlap_rows = []
for scenario_name, config in SCENARIOS.items():
    candidates = create_candidate_trades(execution_prices, scenario_name, config)
    accepted, skipped = remove_overlapping_same_symbol(candidates)
    trade_parts.append(add_cash_returns(accepted))
    for phase, phase_candidates in candidates.groupby('phase'):
        phase_accepted = accepted[accepted['phase'].eq(phase)]
        overlap_rows.append({
            'phase': phase,
            'scenario': scenario_name,
            'candidate_signals': len(phase_candidates),
            'accepted_trades': len(phase_accepted),
            'overlap_skipped': len(phase_candidates) - len(phase_accepted),
        })

trades = pd.concat(trade_parts, ignore_index=True).sort_values(
    ['phase', 'scenario', 'entry_timestamp', 'symbol']
).reset_index(drop=True)
overlap_summary = pd.DataFrame(overlap_rows)
display(overlap_summary)"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 4. $100 기준 전체 수익률

`deployed_return`은 총 순손익을 모든 거래의 투입금 합계(`거래 수 × $100`)로 나눈 값이다. 고정 $100이므로 거래당 평균 순수익률과 같다. `max_drawdown_usd`는 시간순 누적 순손익의 고점 대비 최대 하락이다."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """def peak_concurrent_positions(frame):
    events = []
    for row in frame.itertuples(index=False):
        events.append((row.entry_timestamp, 0, 1))   # 같은 timestamp면 진입을 먼저 세어 보수적 peak
        events.append((row.exit_timestamp, 1, -1))
    active = peak = 0
    for _, _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    return peak

def summarize_group(frame):
    ordered = frame.sort_values(['exit_timestamp', 'symbol', 'sample_id'])
    cumulative = ordered['net_pnl_usd'].cumsum()
    drawdown = cumulative - cumulative.cummax().clip(lower=0)
    deployed = float(frame['stake_usd'].sum())
    return pd.Series({
        'trades': len(frame),
        'symbols': frame['symbol'].nunique(),
        'sessions': frame['session'].nunique(),
        'tp_trades': frame['exit_reason'].eq('TP').sum(),
        'tp_rate': frame['exit_reason'].eq('TP').mean(),
        'win_rate': frame['net_pnl_usd'].gt(0).mean(),
        'total_deployed_usd': deployed,
        'gross_pnl_usd': frame['gross_pnl_usd'].sum(),
        'regulatory_fees_usd': frame['regulatory_fees_usd'].sum(),
        'net_pnl_usd': frame['net_pnl_usd'].sum(),
        'deployed_return': frame['net_pnl_usd'].sum() / deployed,
        'mean_trade_return': frame['net_return'].mean(),
        'median_trade_return': frame['net_return'].median(),
        'worst_trade_return': frame['net_return'].min(),
        'best_trade_return': frame['net_return'].max(),
        'max_drawdown_usd': drawdown.min(),
        'peak_concurrent_positions': peak_concurrent_positions(frame),
        'peak_capital_usd': peak_concurrent_positions(frame) * STAKE_USD,
    })

summary = (
    trades.groupby(['phase', 'scenario'], sort=False)
    .apply(summarize_group, include_groups=False)
    .reset_index()
)

display(summary.style.format({
    'tp_rate': '{:.2%}', 'win_rate': '{:.2%}', 'deployed_return': '{:.2%}',
    'mean_trade_return': '{:.2%}', 'median_trade_return': '{:.2%}',
    'worst_trade_return': '{:.2%}', 'best_trade_return': '{:.2%}',
    'total_deployed_usd': '${:,.2f}', 'gross_pnl_usd': '${:,.2f}',
    'regulatory_fees_usd': '${:,.4f}', 'net_pnl_usd': '${:,.2f}',
    'max_drawdown_usd': '${:,.2f}', 'peak_capital_usd': '${:,.2f}',
}))"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 5. 날짜별·종목별 결과

종목별 표의 `deployed_return` 역시 해당 종목에 실행된 모든 $100 거래의 총 투입금 대비 수익률이다. 한 번도 신호가 없던 종목은 표에 나타나지 않는다."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """def grouped_cash_summary(frame, group_columns):
    return (
        frame.groupby(group_columns)
        .agg(
            trades=('sample_id', 'size'),
            tp_rate=('exit_reason', lambda values: values.eq('TP').mean()),
            win_rate=('net_pnl_usd', lambda values: values.gt(0).mean()),
            total_deployed_usd=('stake_usd', 'sum'),
            net_pnl_usd=('net_pnl_usd', 'sum'),
            mean_trade_return=('net_return', 'mean'),
            median_trade_return=('net_return', 'median'),
        )
        .reset_index()
        .assign(deployed_return=lambda frame: frame['net_pnl_usd'] / frame['total_deployed_usd'])
    )

daily_summary = grouped_cash_summary(trades, ['phase', 'scenario', 'session'])
symbol_summary = grouped_cash_summary(trades, ['phase', 'scenario', 'symbol'])

print('Test 날짜별 결과')
display(
    daily_summary[daily_summary['phase'].eq('Test')]
    .style.format({
        'tp_rate': '{:.2%}', 'win_rate': '{:.2%}', 'deployed_return': '{:.2%}',
        'mean_trade_return': '{:.2%}', 'median_trade_return': '{:.2%}',
        'total_deployed_usd': '${:,.2f}', 'net_pnl_usd': '${:,.2f}',
    })
)

print('Test quote_max_bid 종목별 결과 (순손익 내림차순)')
test_symbol_view = symbol_summary[
    symbol_summary['phase'].eq('Test') & symbol_summary['scenario'].eq('quote_max_bid')
].sort_values('net_pnl_usd', ascending=False)
display(test_symbol_view.style.format({
    'tp_rate': '{:.2%}', 'win_rate': '{:.2%}', 'deployed_return': '{:.2%}',
    'mean_trade_return': '{:.2%}', 'median_trade_return': '{:.2%}',
    'total_deployed_usd': '${:,.2f}', 'net_pnl_usd': '${:,.2f}',
}))"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 6. 추가 왕복 거래비용 stress

호가 spread 이외에 슬리피지·브로커 비용·시장 충격이 진입 원금의 0.2%, 0.5%, 1.0%만큼 추가된다고 가정한다. 이는 실제 비용 추정치가 아니라 수익성의 민감도다."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """stress_rows = []
for (phase, scenario), group in trades.groupby(['phase', 'scenario'], sort=False):
    for extra_cost in [0.0, 0.002, 0.005, 0.010]:
        stress_pnl = group['net_pnl_usd'] - group['stake_usd'] * extra_cost
        stress_rows.append({
            'phase': phase,
            'scenario': scenario,
            'extra_roundtrip_cost': extra_cost,
            'trades': len(group),
            'net_pnl_usd': stress_pnl.sum(),
            'deployed_return': stress_pnl.sum() / group['stake_usd'].sum(),
            'win_rate': stress_pnl.gt(0).mean(),
        })
cost_sensitivity = pd.DataFrame(stress_rows)

display(cost_sensitivity.style.format({
    'extra_roundtrip_cost': '{:.2%}', 'deployed_return': '{:.2%}',
    'win_rate': '{:.2%}', 'net_pnl_usd': '${:,.2f}',
}))"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=False)
for ax, phase in zip(axes, ['OOF', 'Test']):
    for scenario, group in trades[trades['phase'].eq(phase)].groupby('scenario'):
        curve = group.sort_values(['exit_timestamp', 'symbol', 'sample_id'])
        ax.plot(np.arange(1, len(curve) + 1), curve['net_pnl_usd'].cumsum(), label=scenario)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(f'{phase} cumulative net P&L ($100/trade)')
    ax.set_xlabel('closed trades')
    ax.set_ylabel('cumulative net P&L (USD)')
    ax.grid(alpha=0.25)
    ax.legend()
plt.tight_layout()
plt.show()"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 7. Artifact 저장과 최종 판정

거래 ledger에는 각 진입·청산가, 보유 시간, TP/TIMEOUT, 매도 규제비용과 순손익이 들어간다. 최종 판정은 모델 성능과 체결 수익성을 구분한다. OOF로 모델은 고정했지만, 이미 여러 차례 본 Test는 참고치일 뿐이며 새 미래 날짜에서 재현되어야 배포 후보가 된다."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """artifact_paths = {
    'execution_prices': FINAL_ROOT / 'execution_prices.parquet',
    'trade_ledger': FINAL_ROOT / 'trade_ledger.parquet',
    'summary': FINAL_ROOT / 'summary.parquet',
    'overlap_summary': FINAL_ROOT / 'overlap_summary.parquet',
    'daily_summary': FINAL_ROOT / 'daily_summary.parquet',
    'symbol_summary': FINAL_ROOT / 'symbol_summary.parquet',
    'cost_sensitivity': FINAL_ROOT / 'cost_sensitivity.parquet',
    'model_selection_evidence': FINAL_ROOT / 'model_selection_evidence.parquet',
    'manifest': FINAL_ROOT / 'manifest.json',
}

execution_prices.to_parquet(artifact_paths['execution_prices'], index=False)
trades.to_parquet(artifact_paths['trade_ledger'], index=False)
summary.to_parquet(artifact_paths['summary'], index=False)
overlap_summary.to_parquet(artifact_paths['overlap_summary'], index=False)
daily_summary.to_parquet(artifact_paths['daily_summary'], index=False)
symbol_summary.to_parquet(artifact_paths['symbol_summary'], index=False)
cost_sensitivity.to_parquet(artifact_paths['cost_sensitivity'], index=False)
selection_evidence.to_parquet(artifact_paths['model_selection_evidence'], index=False)

manifest = {
    'experiment': 'final_modern_tcn_100usd_v1',
    'created_by_notebook': 'notebooks/10_final_model_selection_100usd_backtest.ipynb',
    'selection_data': 'OOF only',
    'test_role': 'non-pristine development holdout; reporting only',
    'selected_model': SELECTED_MODEL,
    'selected_score': SELECTED_SCORE,
    'checkpoint': str(MODEL_PATH),
    'checkpoint_sha256': checkpoint_sha256,
    'oof_top_fraction': TOP_FRACTION,
    'absolute_threshold': SELECTED_THRESHOLD,
    'stake_usd_per_trade': STAKE_USD,
    'fractional_shares': True,
    'compound_reinvestment': False,
    'same_symbol_overlap': 'skip while prior position is open',
    'tp_return': TP_RETURN,
    'horizon_minutes': HORIZON_MINUTES,
    'stop_loss': None,
    'scenarios': SCENARIOS,
    'fees': {
        'broker_commission': 0.0,
        'sec_section_31_rate_per_dollar_sold': SEC_RATE_PER_DOLLAR,
        'finra_taf_per_share_sold': FINRA_TAF_PER_SHARE,
        'finra_taf_cap_per_trade': FINRA_TAF_CAP,
        'note': 'continuous approximation; actual customer pass-through and rounding depend on broker',
        'sec_source': 'https://www.sec.gov/rules-regulations/fee-rate-advisories/2026-2',
        'finra_source': 'https://www.finra.org/rules-guidance/rule-filings/sr-finra-2024-019/fee-adjustment-schedule',
    },
    'limitations': [
        'quote size, queue priority, partial fills and market impact are unavailable',
        'TP is filled exactly at target when the selected minute-level bid condition is met',
        'Test dates have already influenced the broader research discussion and are not pristine',
    ],
    'artifacts': {key: str(value) for key, value in artifact_paths.items()},
}
artifact_paths['manifest'].write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')

print('saved artifacts')
for key, path in artifact_paths.items():
    print(f'- {key}: {path}')

final_view = summary[['phase', 'scenario', 'trades', 'tp_rate', 'win_rate', 'net_pnl_usd', 'deployed_return']].copy()
display(final_view.style.format({
    'tp_rate': '{:.2%}', 'win_rate': '{:.2%}',
    'net_pnl_usd': '${:,.2f}', 'deployed_return': '{:.2%}',
}))"""
    )
)

nb.cells = cells
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(OUT)
