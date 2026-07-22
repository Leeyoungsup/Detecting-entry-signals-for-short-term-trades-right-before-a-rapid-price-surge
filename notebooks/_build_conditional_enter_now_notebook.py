from pathlib import Path

import nbformat as nbf


root = Path(__file__).resolve().parents[1]
path = root / "notebooks/05_conditional_enter_now_tcn.ipynb"
nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python (urban)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

cells = []
cells.append(
    nbf.v4.new_markdown_cell(
        """# 05. ZONE 조건부 ENTER_NOW TCN

급등 가능 구간과 실제 진입 시점을 서로 다른 학습 문제로 분리한다.

- `ZONE`: 기존 multi-horizon hazard 모델이 `5분 내 보수적 net +3%` 가능성을 예측한다.
- `ENTER_NOW | ZONE`: **실제 ZONE 행만** 학습하여 같은 positive episode 안에서 가장 좋은 한 진입 anchor를 찾는다.
- 조건부 BCE와 episode listwise ranking loss를 함께 사용한다.
- 날짜 expanding OOF에서 두 확률의 threshold를 선택하고 마지막 날짜는 Test로만 사용한다.

이 노트북은 이전처럼 전체 음성 표본을 ENTER_NOW 음성으로 학습하지 않는다. 따라서 조건부 모델이
ZONE 탐지를 다시 복제하는 지름길을 차단한다."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """from pathlib import Path
import gc
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def find_project_root(start=None):
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / 'AGENT.md').is_file() and (candidate / 'README.md').is_file():
            return candidate
    raise FileNotFoundError('프로젝트 루트를 찾지 못했습니다.')


SEED = 91
BASE_VERSION = 'scalp_30m_ohlcv_net3_multihorizon_5m_v2'
TARGET_VERSION = 'scalp_30m_ohlcv_zone_entry_5m_v3'
ZONE_MODEL_VERSION = 'buy_multihorizon_hazard_tcn_v1'
MODEL_VERSION = 'conditional_enter_now_tcn_v2'
MAX_EPOCHS = 24
PATIENCE = 5
PREDICT_BATCH_SIZE = 1024
MAX_EPISODE_BATCH_ROWS = 768
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 1e-3
LISTWISE_WEIGHT = 0.35
STAKE_USD = 100.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.set_float32_matmul_precision('high')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PROJECT_ROOT = find_project_root()
PREPROCESS_ROOT = (PROJECT_ROOT / '../../results/preprocessing').resolve()
RESULT_ROOT = (PROJECT_ROOT / '../../results/modeling').resolve()
MODEL_ROOT = (PROJECT_ROOT / '../../model').resolve()
RESULT_ROOT.mkdir(parents=True, exist_ok=True)
MODEL_ROOT.mkdir(parents=True, exist_ok=True)

base_schema = json.loads((PREPROCESS_ROOT / f'{BASE_VERSION}_schema.json').read_text())
metadata = pd.read_parquet(PREPROCESS_ROOT / f'{TARGET_VERSION}_metadata.parquet')
with np.load(PREPROCESS_ROOT / f'{BASE_VERSION}_features.npz') as loaded:
    all_sequence = loaded['sequence'].astype(np.float32)
with np.load(PREPROCESS_ROOT / f'{TARGET_VERSION}_targets.npz') as loaded:
    enter_target = loaded['enter_now_target'].astype(np.float32)
zone_target = metadata['target_zone_5m'].to_numpy(dtype=np.int8)

default_indices = [base_schema['feature_names'].index(name) for name in base_schema['default_feature_names']]
sequence = np.ascontiguousarray(all_sequence[:, :, default_indices], dtype=np.float32)
del all_sequence
sessions = sorted(metadata['session'].unique())
TEST_SESSION = sessions[-1]
OOF_SESSIONS = sessions[-6:-1]

assert len(sequence) == len(metadata) == len(enter_target) == len(zone_target)
assert np.array_equal(metadata['feature_index'].to_numpy(), np.arange(len(metadata)))
assert np.array_equal(enter_target, metadata['target_enter_now'].to_numpy())
assert np.array_equal(zone_target, metadata['target_zone_5m'].to_numpy())
assert np.all(enter_target <= zone_target)

zone_rows = zone_target.astype(bool)
print('device:', DEVICE, '| sequence:', sequence.shape)
print('all rows:', len(metadata), '| ZONE:', int(zone_rows.sum()), '| ENTER_NOW:', int(enter_target.sum()))
print('P(ENTER_NOW | ZONE):', enter_target[zone_rows].mean())
print('OOF:', OOF_SESSIONS, '| Test:', TEST_SESSION)"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit_scaler(indices, max_windows=20_000, seed=SEED):
    indices = np.asarray(indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if len(indices) > max_windows:
        indices = rng.choice(indices, max_windows, replace=False)
    sample = sequence[indices].reshape(-1, sequence.shape[-1])
    center = np.median(sample, axis=0).astype(np.float32)
    q25, q75 = np.percentile(sample, [25, 75], axis=0)
    scale = np.maximum((q75 - q25).astype(np.float32), 1e-4)
    return center, scale


def scale_rows(indices, center, scale):
    values = (sequence[indices] - center[None, None, :]) / scale[None, None, :]
    return np.clip(values, -10, 10).astype(np.float32)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.depthwise = nn.Conv1d(
            channels, channels, 3, padding=dilation, dilation=dilation, groups=channels
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(channels * 2, channels, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.pointwise(self.depthwise(self.norm(x))))


class ConditionalEnterNowTCN(nn.Module):
    def __init__(self, input_dim, initial_probability, channels=48, dropout=0.15):
        super().__init__()
        self.input_projection = nn.Conv1d(input_dim, channels, 1)
        self.blocks = nn.ModuleList(
            [ResidualTemporalBlock(channels, dilation, dropout) for dilation in [1, 2, 4, 8]]
        )
        self.final_norm = nn.GroupNorm(1, channels)
        self.shared = nn.Sequential(
            nn.LayerNorm(channels * 3),
            nn.Linear(channels * 3, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.entry_head = nn.Linear(64, 1)
        probability = float(np.clip(initial_probability, 1e-4, 1 - 1e-4))
        with torch.no_grad():
            self.entry_head.bias.fill_(math.log(probability / (1 - probability)))

    def forward(self, x):
        x = self.input_projection(x.transpose(1, 2))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        pooled = torch.cat([x[:, :, -1], x.mean(2), x.amax(2)], dim=1)
        return self.entry_head(self.shared(pooled)).squeeze(1)


def ece(actual, probability, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probability >= left) & (probability < right if right < 1 else probability <= right)
        if mask.any():
            value += mask.mean() * abs(actual[mask].mean() - probability[mask].mean())
    return float(value)


def binary_metrics(actual, probability):
    probability = np.clip(np.asarray(probability), 1e-7, 1 - 1e-7)
    actual = np.asarray(actual)
    return {
        'samples': len(actual),
        'positives': int(actual.sum()),
        'prevalence': float(actual.mean()),
        'mean_probability': float(probability.mean()),
        'pr_auc': float(average_precision_score(actual, probability)),
        'roc_auc': float(roc_auc_score(actual, probability)),
        'brier': float(brier_score_loss(actual, probability)),
        'log_loss': float(log_loss(actual, probability, labels=[0, 1])),
        'ece': ece(actual, probability),
    }


def episode_ranking_metrics(frame, score_column):
    frame = frame[frame.target_zone_5m.eq(1)].copy()
    rows = []
    for _, group in frame.groupby('positive_episode_id', sort=False):
        group = group.sort_values('decision_bar_timestamp')
        anchor = group.loc[group.target_enter_now.eq(1)].iloc[0]
        pick = group.loc[group[score_column].idxmax()]
        siblings = group.loc[~group.sample_id.eq(anchor.sample_id), score_column]
        pairwise = np.nan if len(siblings) == 0 else float(
            (anchor[score_column] > siblings).mean()
            + 0.5 * (anchor[score_column] == siblings).mean()
        )
        rows.append({
            'episode_length': len(group),
            'exact_anchor': int(pick.sample_id == anchor.sample_id),
            'absolute_timing_error_minutes': abs(float(pick.minutes_from_entry_anchor)),
            'anchor_pairwise_win_rate': pairwise,
        })
    detail = pd.DataFrame(rows)
    multi = detail[detail.episode_length.gt(1)]
    return {
        'episodes': len(detail),
        'multirow_episodes': len(multi),
        'all_exact_anchor_rate': float(detail.exact_anchor.mean()),
        'all_timing_mae_minutes': float(detail.absolute_timing_error_minutes.mean()),
        'multirow_exact_anchor_rate': float(multi.exact_anchor.mean()) if len(multi) else np.nan,
        'multirow_timing_mae_minutes': float(multi.absolute_timing_error_minutes.mean()) if len(multi) else np.nan,
        'multirow_anchor_pairwise_win_rate': float(multi.anchor_pairwise_win_rate.mean()) if len(multi) else np.nan,
    }


probe = ConditionalEnterNowTCN(sequence.shape[-1], enter_target[zone_rows].mean())
print('parameters:', sum(parameter.numel() for parameter in probe.parameters()))
del probe"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """def episode_groups(indices):
    indices = np.asarray(indices, dtype=np.int64)
    assert np.all(zone_target[indices] == 1)
    frame = metadata.iloc[indices][['feature_index', 'positive_episode_id']]
    groups = [
        group.feature_index.to_numpy(dtype=np.int64)
        for _, group in frame.groupby('positive_episode_id', sort=False)
    ]
    assert sum(map(len, groups)) == len(indices)
    assert all(enter_target[group].sum() == 1 for group in groups)
    return groups


def packed_episode_batches(indices, shuffle, seed):
    groups = episode_groups(indices)
    order = np.arange(len(groups))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    batch = []
    row_count = 0
    for position in order:
        group = groups[position]
        if batch and row_count + len(group) > MAX_EPISODE_BATCH_ROWS:
            yield batch
            batch = []
            row_count = 0
        batch.append(group)
        row_count += len(group)
    if batch:
        yield batch


def listwise_loss(logits, groups):
    losses = []
    offset = 0
    for group in groups:
        count = len(group)
        group_logits = logits[offset:offset + count]
        group_target = torch.as_tensor(
            enter_target[group], dtype=group_logits.dtype, device=group_logits.device
        )
        anchor_logit = group_logits[group_target.eq(1)].squeeze(0)
        if count > 1:
            losses.append(torch.logsumexp(group_logits, dim=0) - anchor_logit)
        offset += count
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def train_epoch(model, indices, center, scale, optimizer, seed):
    model.train()
    total_loss = total_bce = total_rank = 0.0
    count = 0
    for groups in packed_episode_batches(indices, shuffle=True, seed=seed):
        batch_indices = np.concatenate(groups)
        x = torch.from_numpy(scale_rows(batch_indices, center, scale)).to(DEVICE)
        y = torch.from_numpy(enter_target[batch_indices]).to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, y)
        ranking = listwise_loss(logits, groups)
        loss = bce + LISTWISE_WEIGHT * ranking
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        n = len(batch_indices)
        total_loss += loss.item() * n
        total_bce += bce.item() * n
        total_rank += ranking.item() * n
        count += n
    return total_loss / count, total_bce / count, total_rank / count


@torch.no_grad()
def predict(model, indices, center, scale):
    indices = np.asarray(indices, dtype=np.int64)
    dataset = TensorDataset(
        torch.from_numpy(scale_rows(indices, center, scale)),
        torch.as_tensor(indices, dtype=torch.long),
    )
    loader = DataLoader(
        dataset,
        batch_size=PREDICT_BATCH_SIZE,
        shuffle=False,
        pin_memory=DEVICE.type == 'cuda',
        num_workers=0,
    )
    model.eval()
    logits = []
    rows = []
    for x, index in loader:
        logits.append(model(x.to(DEVICE, non_blocking=True)).cpu().numpy())
        rows.append(index.numpy())
    return np.concatenate(logits), np.concatenate(rows)


def numpy_listwise_loss(indices, logits):
    frame = metadata.iloc[indices][['feature_index', 'positive_episode_id']].copy()
    frame['logit'] = logits
    losses = []
    for _, group in frame.groupby('positive_episode_id', sort=False):
        if len(group) == 1:
            continue
        group_indices = group.feature_index.to_numpy(dtype=np.int64)
        values = group.logit.to_numpy()
        anchor_position = np.flatnonzero(enter_target[group_indices] == 1)[0]
        losses.append(logsumexp(values) - values[anchor_position])
    return float(np.mean(losses)) if losses else 0.0


def validation_objective(indices, logits):
    probability = expit(logits)
    bce = log_loss(enter_target[indices], probability, labels=[0, 1])
    ranking = numpy_listwise_loss(indices, logits)
    return bce + LISTWISE_WEIGHT * ranking, bce, ranking


def choose_epoch(train_indices, valid_indices, seed):
    set_seed(seed)
    center, scale = fit_scaler(train_indices, seed=seed)
    initial_probability = enter_target[train_indices].mean()
    model = ConditionalEnterNowTCN(sequence.shape[-1], initial_probability).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=5e-5
    )
    best = np.inf
    best_epoch = 1
    stale = 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_bce, train_rank = train_epoch(
            model, train_indices, center, scale, optimizer, seed + epoch
        )
        valid_logits, valid_rows = predict(model, valid_indices, center, scale)
        assert np.array_equal(valid_rows, valid_indices)
        valid_loss, valid_bce, valid_rank = validation_objective(valid_indices, valid_logits)
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_bce': train_bce,
            'train_listwise': train_rank,
            'valid_loss': valid_loss,
            'valid_bce': valid_bce,
            'valid_listwise': valid_rank,
        })
        if valid_loss < best - 1e-5:
            best = valid_loss
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        scheduler.step()
        if stale >= PATIENCE:
            break
    del model
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return best_epoch, pd.DataFrame(history)


def fit_fixed(train_indices, epochs, seed):
    set_seed(seed)
    center, scale = fit_scaler(train_indices, seed=seed)
    initial_probability = enter_target[train_indices].mean()
    model = ConditionalEnterNowTCN(sequence.shape[-1], initial_probability).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=5e-5
    )
    history = []
    for epoch in range(1, epochs + 1):
        loss, bce, ranking = train_epoch(
            model, train_indices, center, scale, optimizer, seed + epoch
        )
        history.append({'epoch': epoch, 'train_loss': loss, 'train_bce': bce, 'train_listwise': ranking})
        scheduler.step()
    gc.collect()
    return model, center, scale, pd.DataFrame(history)"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """oof_frames = []
fold_rows = []
histories = []
best_epochs = []
started = time.time()
columns = [
    'sample_id', 'feature_index', 'session', 'symbol', 'decision_bar_timestamp',
    'entry_timestamp', 'entry_ask', 'target_zone_5m', 'target_enter_first',
    'target_enter_now', 'first_hit_minute', 'timeout_net_return_5m',
    'positive_episode_id', 'positive_episode_position', 'minutes_from_entry_anchor',
    'entry_quality',
]

for fold, valid_session in enumerate(OOF_SESSIONS, start=1):
    prior_sessions = [session for session in sessions if session < valid_session]
    inner_valid_session = prior_sessions[-1]
    inner_train_sessions = prior_sessions[:-1]

    inner_train_indices = np.flatnonzero(
        metadata.session.isin(inner_train_sessions).to_numpy() & zone_rows
    )
    inner_valid_indices = np.flatnonzero(
        metadata.session.eq(inner_valid_session).to_numpy() & zone_rows
    )
    refit_indices = np.flatnonzero(metadata.session.isin(prior_sessions).to_numpy() & zone_rows)
    valid_all_indices = np.flatnonzero(metadata.session.eq(valid_session).to_numpy())
    valid_zone_indices = np.flatnonzero(metadata.session.eq(valid_session).to_numpy() & zone_rows)

    epoch, history = choose_epoch(
        inner_train_indices, inner_valid_indices, SEED + fold
    )
    best_epochs.append(epoch)
    history['fold'] = fold
    history['stage'] = 'inner'
    histories.append(history)

    model, center, scale, refit_history = fit_fixed(
        refit_indices, epoch, SEED + 100 + fold
    )
    refit_history['fold'] = fold
    refit_history['stage'] = 'refit'
    histories.append(refit_history)

    logits, indices = predict(model, valid_all_indices, center, scale)
    assert np.array_equal(indices, valid_all_indices)
    frame = metadata.iloc[valid_all_indices][columns].copy().reset_index(drop=True)
    frame['raw_conditional_entry_logit'] = logits
    frame['raw_conditional_entry_probability'] = expit(logits)
    frame['fold'] = fold
    oof_frames.append(frame)

    conditioned = frame[frame.target_zone_5m.eq(1)]
    probability_metrics = binary_metrics(
        conditioned.target_enter_now, conditioned.raw_conditional_entry_probability
    )
    ranking_metrics = episode_ranking_metrics(frame, 'raw_conditional_entry_probability')
    fold_rows.append({
        'fold': fold,
        'validation_session': valid_session,
        'best_epoch': epoch,
        **{f'conditional_{key}': value for key, value in probability_metrics.items()},
        **ranking_metrics,
    })
    print(
        f"fold {fold} {valid_session} epoch={epoch} "
        f"conditional AP={probability_metrics['pr_auc']:.4f} "
        f"multirow top1={ranking_metrics['multirow_exact_anchor_rate']:.2%}"
    )
    del model
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

oof = pd.concat(oof_frames, ignore_index=True).sort_values('entry_timestamp').reset_index(drop=True)
fold_metrics = pd.DataFrame(fold_rows)
training_history = pd.concat(histories, ignore_index=True)
print(f'elapsed {(time.time() - started) / 60:.2f} min')
display(fold_metrics)"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """def fit_platt(frame):
    if frame.target_enter_now.nunique() < 2:
        return {'coef': 1.0, 'intercept': 0.0, 'identity': True}
    estimator = LogisticRegression(C=10.0, solver='lbfgs', max_iter=1000)
    estimator.fit(frame[['raw_conditional_entry_logit']], frame.target_enter_now)
    return {
        'coef': float(estimator.coef_[0, 0]),
        'intercept': float(estimator.intercept_[0]),
        'identity': False,
    }


rolling = []
for position, session in enumerate(OOF_SESSIONS):
    current = oof[oof.session.eq(session)].copy()
    prior_zone = oof[
        oof.session.isin(OOF_SESSIONS[:position]) & oof.target_zone_5m.eq(1)
    ]
    calibration = fit_platt(prior_zone) if len(prior_zone) else {
        'coef': 1.0, 'intercept': 0.0, 'identity': True
    }
    current['calibrated_conditional_entry_probability'] = expit(
        calibration['coef'] * current.raw_conditional_entry_logit + calibration['intercept']
    )
    current['calibration_history_sessions'] = position
    rolling.append(current)

oof = pd.concat(rolling, ignore_index=True).sort_values('entry_timestamp').reset_index(drop=True)
eligible_zone = oof[oof.calibration_history_sessions.gt(0) & oof.target_zone_5m.eq(1)]
raw_metrics = binary_metrics(
    eligible_zone.target_enter_now, eligible_zone.raw_conditional_entry_probability
)
calibrated_metrics = binary_metrics(
    eligible_zone.target_enter_now, eligible_zone.calibrated_conditional_entry_probability
)
SELECTED_PROBABILITY_TYPE = (
    'raw'
    if (raw_metrics['brier'], raw_metrics['log_loss'])
    <= (calibrated_metrics['brier'], calibrated_metrics['log_loss'])
    else 'calibrated'
)
ENTRY_PROBABILITY_COLUMN = (
    'raw_conditional_entry_probability'
    if SELECTED_PROBABILITY_TYPE == 'raw'
    else 'calibrated_conditional_entry_probability'
)
FINAL_CALIBRATION = fit_platt(oof[oof.target_zone_5m.eq(1)])

zone_oof = pd.read_parquet(
    RESULT_ROOT / f'{ZONE_MODEL_VERSION}_oof_predictions.parquet',
    columns=['feature_index', 'raw_tp_by_probability_5m'],
)
combined_oof = oof.merge(zone_oof, on='feature_index', how='inner', validate='one_to_one')
combined_oof['joint_entry_probability'] = (
    combined_oof.raw_tp_by_probability_5m * combined_oof[ENTRY_PROBABILITY_COLUMN]
)

conditional_ranking = episode_ranking_metrics(combined_oof, ENTRY_PROBABILITY_COLUMN)
zone_ranking = episode_ranking_metrics(combined_oof, 'raw_tp_by_probability_5m')
global_joint_metrics = binary_metrics(
    combined_oof.target_enter_now, combined_oof.joint_entry_probability
)
score_correlation = float(combined_oof[
    ['raw_tp_by_probability_5m', ENTRY_PROBABILITY_COLUMN]
].corr(method='spearman').iloc[0, 1])

print('conditional raw:', raw_metrics)
print('conditional calibrated:', calibrated_metrics)
print('selected:', SELECTED_PROBABILITY_TYPE)
print('OOF conditional ranking:', conditional_ranking)
print('OOF zone-only ranking:', zone_ranking)
print('OOF joint target metrics:', global_joint_metrics)
print('OOF zone/conditional score Spearman:', score_correlation)"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """def sequential_trades(frame, signal_column):
    candidates = frame[frame[signal_column]].sort_values('entry_timestamp').copy()
    selected_indices = []
    for _, group in candidates.groupby(['session', 'symbol'], sort=False):
        available = None
        for index, row in group.sort_values('entry_timestamp').iterrows():
            if available is not None and row.entry_timestamp < available:
                continue
            selected_indices.append(index)
            hold_minutes = int(row.first_hit_minute) if row.target_zone_5m else 5
            available = row.entry_timestamp + pd.Timedelta(minutes=hold_minutes)
    trades = candidates.loc[selected_indices].copy()
    trades['policy_return'] = np.where(
        trades.target_zone_5m.eq(1), 0.03, trades.timeout_net_return_5m
    )
    return trades


def policy_metrics(frame, signal_column):
    trades = sequential_trades(frame, signal_column)
    if len(trades) == 0:
        return {
            'trades': 0, 'sessions_traded': 0, 'zone_precision': np.nan,
            'entry_precision': np.nan, 'entry_recall': 0.0,
            'mean_abs_timing_error': np.nan, 'mean_return': np.nan,
            'risk_adjusted_return': np.nan, 'total_pnl_usd': 0.0,
            'profit_factor': np.nan,
        }
    returns = trades.policy_return.to_numpy()
    pnl = returns * STAKE_USD
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    timed = trades[trades.target_zone_5m.eq(1) & trades.minutes_from_entry_anchor.notna()]
    return {
        'trades': len(trades),
        'sessions_traded': trades.session.nunique(),
        'zone_precision': trades.target_zone_5m.mean(),
        'entry_precision': trades.target_enter_now.mean(),
        'entry_recall': trades.target_enter_now.sum() / max(frame.target_enter_now.sum(), 1),
        'mean_abs_timing_error': (
            timed.minutes_from_entry_anchor.abs().mean() if len(timed) else np.nan
        ),
        'mean_return': returns.mean(),
        'risk_adjusted_return': (
            returns.mean() - returns.std(ddof=1) / math.sqrt(len(returns))
            if len(returns) > 1 else returns.mean()
        ),
        'total_pnl_usd': pnl.sum(),
        'profit_factor': gross_profit / gross_loss if gross_loss > 0 else np.inf,
    }


threshold_frame = combined_oof.copy()
zone_thresholds = np.unique(np.round(np.r_[
    np.arange(0.01, 0.151, 0.02),
    threshold_frame.raw_tp_by_probability_5m.quantile([0.5, 0.7, 0.8, 0.9, 0.95, 0.975]),
], 6))
entry_thresholds = np.unique(np.round(np.r_[
    np.arange(0.10, 0.91, 0.05),
    threshold_frame[ENTRY_PROBABILITY_COLUMN].quantile([0.5, 0.7, 0.8, 0.9, 0.95, 0.975]),
], 6))

rows = []
for zone_threshold in zone_thresholds:
    for entry_threshold in entry_thresholds:
        threshold_frame['candidate_signal'] = (
            threshold_frame.raw_tp_by_probability_5m.ge(zone_threshold)
            & threshold_frame[ENTRY_PROBABILITY_COLUMN].ge(entry_threshold)
        )
        rows.append({
            'zone_threshold': zone_threshold,
            'conditional_entry_threshold': entry_threshold,
            **policy_metrics(threshold_frame, 'candidate_signal'),
        })

threshold_table = pd.DataFrame(rows)
candidates = threshold_table[
    threshold_table.trades.ge(50) & threshold_table.sessions_traded.ge(3)
]
selected = candidates.sort_values(['risk_adjusted_return', 'total_pnl_usd']).iloc[-1]
ZONE_THRESHOLD = float(selected.zone_threshold)
ENTRY_THRESHOLD = float(selected.conditional_entry_threshold)
OOF_ELIGIBLE = bool(selected.risk_adjusted_return > 0 and selected.profit_factor > 1)
threshold_frame['selected_signal'] = (
    threshold_frame.raw_tp_by_probability_5m.ge(ZONE_THRESHOLD)
    & threshold_frame[ENTRY_PROBABILITY_COLUMN].ge(ENTRY_THRESHOLD)
)
oof_trades = sequential_trades(threshold_frame, 'selected_signal')

print('selected thresholds:', ZONE_THRESHOLD, ENTRY_THRESHOLD)
print('OOF economically eligible:', OOF_ELIGIBLE)
display(selected.to_frame('value'))
display(threshold_table.sort_values('risk_adjusted_return', ascending=False).head(20))"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """FINAL_EPOCHS = int(np.rint(np.median(best_epochs)))
train_zone_indices = np.flatnonzero(metadata.session.ne(TEST_SESSION).to_numpy() & zone_rows)
test_indices = np.flatnonzero(metadata.session.eq(TEST_SESSION).to_numpy())
model, center, scale, final_history = fit_fixed(
    train_zone_indices, FINAL_EPOCHS, SEED + 999
)
test_logits, test_rows = predict(model, test_indices, center, scale)
assert np.array_equal(test_rows, test_indices)

test = metadata.iloc[test_indices][columns].copy().reset_index(drop=True)
test['raw_conditional_entry_logit'] = test_logits
test['raw_conditional_entry_probability'] = expit(test_logits)
test['calibrated_conditional_entry_probability'] = expit(
    FINAL_CALIBRATION['coef'] * test_logits + FINAL_CALIBRATION['intercept']
)
TEST_ENTRY_COLUMN = (
    'raw_conditional_entry_probability'
    if SELECTED_PROBABILITY_TYPE == 'raw'
    else 'calibrated_conditional_entry_probability'
)
zone_test = pd.read_parquet(
    RESULT_ROOT / f'{ZONE_MODEL_VERSION}_test_predictions.parquet',
    columns=['feature_index', 'raw_tp_by_probability_5m'],
)
test = test.merge(zone_test, on='feature_index', how='inner', validate='one_to_one')
test['joint_entry_probability'] = (
    test.raw_tp_by_probability_5m * test[TEST_ENTRY_COLUMN]
)
test['buy_signal'] = (
    test.raw_tp_by_probability_5m.ge(ZONE_THRESHOLD)
    & test[TEST_ENTRY_COLUMN].ge(ENTRY_THRESHOLD)
)

test_zone = test[test.target_zone_5m.eq(1)]
test_conditional_metrics = binary_metrics(
    test_zone.target_enter_now, test_zone[TEST_ENTRY_COLUMN]
)
test_joint_metrics = binary_metrics(test.target_enter_now, test.joint_entry_probability)
test_conditional_ranking = episode_ranking_metrics(test, TEST_ENTRY_COLUMN)
test_zone_ranking = episode_ranking_metrics(test, 'raw_tp_by_probability_5m')
test_score_correlation = float(test[
    ['raw_tp_by_probability_5m', TEST_ENTRY_COLUMN]
].corr(method='spearman').iloc[0, 1])
test_policy = policy_metrics(test, 'buy_signal')
test_trades = sequential_trades(test, 'buy_signal')
DEPLOYMENT_ELIGIBLE = bool(
    OOF_ELIGIBLE
    and test_policy['trades'] >= 20
    and test_policy['risk_adjusted_return'] > 0
    and test_policy['profit_factor'] > 1
)

model_path = MODEL_ROOT / f'{MODEL_VERSION}.pt'
torch.save({
    'model_version': MODEL_VERSION,
    'architecture': 'ConditionalEnterNowTCN',
    'training_population': 'target_zone_5m == 1 only',
    'loss': {'conditional_bce': 1.0, 'episode_listwise': LISTWISE_WEIGHT},
    'state_dict': {key: value.detach().cpu() for key, value in model.state_dict().items()},
    'feature_names': base_schema['default_feature_names'],
    'scaler_center': torch.from_numpy(center),
    'scaler_scale': torch.from_numpy(scale),
    'selected_probability_type': SELECTED_PROBABILITY_TYPE,
    'calibration': FINAL_CALIBRATION,
    'research_zone_threshold': ZONE_THRESHOLD,
    'research_conditional_entry_threshold': ENTRY_THRESHOLD,
    'authorized_thresholds': (
        {'zone': ZONE_THRESHOLD, 'conditional_entry': ENTRY_THRESHOLD}
        if DEPLOYMENT_ELIGIBLE else None
    ),
    'deployment_eligible': DEPLOYMENT_ELIGIBLE,
}, model_path)

fold_metrics.to_parquet(
    RESULT_ROOT / f'{MODEL_VERSION}_fold_metrics.parquet', index=False, compression='zstd'
)
training_history.to_parquet(
    RESULT_ROOT / f'{MODEL_VERSION}_training_history.parquet', index=False, compression='zstd'
)
oof.to_parquet(
    RESULT_ROOT / f'{MODEL_VERSION}_oof_predictions.parquet', index=False, compression='zstd'
)
test.to_parquet(
    RESULT_ROOT / f'{MODEL_VERSION}_test_predictions.parquet', index=False, compression='zstd'
)
threshold_table.to_parquet(
    RESULT_ROOT / f'{MODEL_VERSION}_threshold_candidates.parquet', index=False, compression='zstd'
)
test_trades.to_parquet(
    RESULT_ROOT / f'{MODEL_VERSION}_test_trades.parquet', index=False, compression='zstd'
)

def clean_mapping(mapping):
    return {
        key: (None if pd.isna(value) else float(value))
        for key, value in mapping.items()
    }


payload = {
    'model_version': MODEL_VERSION,
    'parameters': sum(parameter.numel() for parameter in model.parameters()),
    'training_population': 'actual ZONE rows only',
    'loss': {'conditional_bce': 1.0, 'episode_listwise': LISTWISE_WEIGHT},
    'best_epochs': best_epochs,
    'final_epochs': FINAL_EPOCHS,
    'selected_probability_type': SELECTED_PROBABILITY_TYPE,
    'oof_conditional_probability_metrics': (
        raw_metrics if SELECTED_PROBABILITY_TYPE == 'raw' else calibrated_metrics
    ),
    'oof_conditional_ranking_metrics': conditional_ranking,
    'oof_zone_only_ranking_metrics': zone_ranking,
    'oof_global_joint_metrics': global_joint_metrics,
    'oof_zone_conditional_score_spearman': score_correlation,
    'test_conditional_probability_metrics': test_conditional_metrics,
    'test_conditional_ranking_metrics': test_conditional_ranking,
    'test_zone_only_ranking_metrics': test_zone_ranking,
    'test_global_joint_metrics': test_joint_metrics,
    'test_zone_conditional_score_spearman': test_score_correlation,
    'zone_threshold': ZONE_THRESHOLD,
    'conditional_entry_threshold': ENTRY_THRESHOLD,
    'selected_oof_policy': clean_mapping(selected.to_dict()),
    'test_policy': clean_mapping(test_policy),
    'oof_economic_eligible': OOF_ELIGIBLE,
    'deployment_eligible': DEPLOYMENT_ELIGIBLE,
}
(RESULT_ROOT / f'{MODEL_VERSION}_metrics.json').write_text(
    json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
)

print('Test conditional metrics:', test_conditional_metrics)
print('Test conditional ranking:', test_conditional_ranking)
print('Test zone-only ranking:', test_zone_ranking)
print('Test joint metrics:', test_joint_metrics)
print('Test policy:', test_policy)
print('deployment eligible:', DEPLOYMENT_ELIGIBLE)
display(test_trades.head(30))"""
    )
)

nb["cells"] = cells
nbf.write(nb, path)
print(path)
