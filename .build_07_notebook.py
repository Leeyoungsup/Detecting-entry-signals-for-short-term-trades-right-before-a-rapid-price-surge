from pathlib import Path

import nbformat as nbf


root = Path(__file__).resolve().parent
source_path = root / 'notebooks/06_modern_tcn_baseline.ipynb'
target_path = root / 'notebooks/07_downside_aware_target.ipynb'
nb = nbf.read(source_path, as_version=4)

nb.cells[0].source = r'''# 07. Downside-aware +3% 직접 target

요청한 개선을 한 번에 섞지 않고 첫 번째 항목만 검증한다. `06_modern_tcn_baseline.ipynb`의 데이터, 60봉 입력, context, 날짜 OOF fold, stride=3, scaler, ModernTCN 구조, optimizer, loss와 epoch 후보를 모두 유지하고 **target만 변경**한다.

## Target

```text
original TP:       MFE_3m >= +3%
downside-aware TP: MFE_3m >= +3% AND downside_3m < 3%
```

`downside = -MAE`다. 같은 3분 동안 위·아래가 모두 3% 이상 움직인 샘플은 고변동 양방향 움직임으로 보고 negative 처리한다. 새로운 손절 기준을 튜닝하지 않고 기존 3% 하나만 사용한다.

1분 OHLC만으로 같은 봉 안에서 high와 low 중 어느 것이 먼저 발생했는지는 알 수 없다. 따라서 이 target은 TP-first 체결 라벨이 아니라 **보수적인 clean-upward-excursion 라벨**이다.

비교는 기존 ModernTCN과 새 target 모델을 동일한 downside-aware 정답에서 평가한다. 아직 utility 직접 학습, episode 평가, percentile threshold, 신규 context는 적용하지 않는다.'''

cell1 = nb.cells[1].source.replace(
    '../../model/moderntcn_ohlc_60m_v1',
    '../../model/moderntcn_downside_aware_tp3_v1',
).replace(
    '../../results/training/moderntcn_ohlc_60m_v1',
    '../../results/training/moderntcn_downside_aware_tp3_v1',
)
nb.cells[1].source = cell1

cell3 = nb.cells[3].source
cell3 = cell3.replace(
    "y_hard = metadata['target_tp3_3m'].to_numpy(np.float32)",
    "y_original_hard = metadata['target_tp3_3m'].to_numpy(np.float32)\n"
    "downside_3m = (-metadata['mae_3m']).to_numpy(np.float32)\n"
    "both_sides_3pct = (y_original_hard == 1.0) & (downside_3m >= 0.03)\n"
    "y_hard = ((y_original_hard == 1.0) & (downside_3m < 0.03)).astype(np.float32)",
)
cell3 = cell3.replace(
    "'target': ['open hard TP3/3m', 'random-entry soft TP3/3m'],",
    "'target': ['downside-aware TP3/3m', 'original TP3/3m'],",
).replace(
    "'Train mean': [float(y_hard[train_mask].mean()), float(y_soft[train_mask].mean())],",
    "'Train mean': [float(y_hard[train_mask].mean()), float(y_original_hard[train_mask].mean())],",
).replace(
    "'Test mean': [float(y_hard[test_mask].mean()), float(y_soft[test_mask].mean())],",
    "'Test mean': [float(y_hard[test_mask].mean()), float(y_original_hard[test_mask].mean())],",
)
cell3 += r'''

target_change = pd.DataFrame([
    {
        'partition': name,
        'samples': int(mask.sum()),
        'original_positives': int(y_original_hard[mask].sum()),
        'downside_aware_positives': int(y_hard[mask].sum()),
        'both_sides_relabelled_negative': int(both_sides_3pct[mask].sum()),
        'removed_positive_share': float(
            both_sides_3pct[mask].sum() / max(y_original_hard[mask].sum(), 1)
        ),
    }
    for name, mask in [('Train', train_mask), ('OOF', oof_mask), ('Test', test_mask)]
])
display(target_change.style.format({'removed_positive_share': '{:.2%}'}))'''
nb.cells[3].source = cell3

nb.cells[4].source = r'''## 2. 고정된 Compact ModernTCN 백본

06번과 동일한 117,825 parameter Compact ModernTCN을 그대로 사용한다. 이 실험에서 바뀌는 것은 `target_tp3_3m`을 downside-aware target으로 교체한 것뿐이다.'''

cell7 = nb.cells[7].source
start = cell7.index('STRATEGIES = {')
end = cell7.index('\n\n\ndef robust_center_scale', start)
cell7 = cell7[:start] + "STRATEGIES = {\n    'downside_aware_hard': {'output_dim': 1, 'selection_metric': 'hard_pr_auc', 'mode': 'max'},\n}" + cell7[end:]
cell7 = cell7.replace("if strategy == 'open_hard':", "if strategy == 'downside_aware_hard':")
cell7 = cell7.replace("if strategy in ('open_hard', 'random_soft'):", "if strategy == 'downside_aware_hard':")
nb.cells[7].source = cell7

nb.cells[8].source = r'''## 4. Downside-aware target의 expanding walk-forward OOF 학습

과거 날짜만 fit하고 다음 한 날짜를 예측한다. 모델 구조와 학습 설정은 06번과 같으며 target만 달라진다. Test는 epoch 선택에 사용하지 않는다.'''

nb.cells[10].source = r'''## 5. OOF로 epoch 선택

네 OOF 날짜의 downside-aware 예측을 이어 붙인 pooled PR-AUC로 epoch를 선택한다. 기존 모델과의 비교 역시 새 정답을 기준으로 다시 계산한다.'''

nb.cells[12].source = r'''## 6. 전체 Train 재학습과 고정 Test 진단

OOF에서 선택된 epoch로 7개 Train 날짜 전체를 다시 학습한다. Test는 여기서 처음 평가한다.'''

nb.cells[14].source = r'''## 7. Target·모델·OOF/Test artifact 저장

새 target metadata와 모델 checkpoint를 별도 version으로 저장한다. 기존 preprocessing과 ModernTCN 결과는 덮어쓰지 않는다.'''

nb.cells[15].source = r'''def scaler_to_serializable(scaler):
    if scaler is None:
        return None
    return {key: np.asarray(value).tolist() for key, value in scaler.items()}


def make_prediction_frame(indices, phase):
    frame = metadata.iloc[indices][[
        'sample_id', 'session', 'symbol', 'decision_timestamp', 'entry_timestamp',
        'target_tp3_3m', 'mfe_3m', 'mae_3m',
    ]].reset_index(drop=True).copy()
    frame = frame.rename(columns={'target_tp3_3m': 'target_original_tp3_3m'})
    frame['target_downside_aware_tp3_3m'] = y_hard[indices].astype(np.int8)
    frame['both_sides_3pct'] = both_sides_3pct[indices]
    frame['downside_3m'] = downside_3m[indices]
    if phase == 'oof':
        frame['oof_fold'] = split.iloc[indices]['oof_fold'].to_numpy()
        raw = oof_raw_predictions['downside_aware_hard'][
            selected_epochs['downside_aware_hard']
        ][indices]
    else:
        raw = test_raw_predictions['downside_aware_hard']
    frame['pred_downside_aware_probability'] = clipped_probability(
        raw.reshape(-1)
    ).astype(np.float32)
    return frame


oof_predictions = make_prediction_frame(oof_indices, 'oof')
test_predictions = make_prediction_frame(test_indices, 'test')

target_metadata = metadata[[
    'sample_id', 'session', 'symbol', 'decision_timestamp', 'entry_timestamp'
]].copy()
target_metadata['target_original_tp3_3m'] = y_original_hard.astype(np.int8)
target_metadata['downside_3m'] = downside_3m
target_metadata['both_sides_3pct'] = both_sides_3pct
target_metadata['target_downside_aware_tp3_3m'] = y_hard.astype(np.int8)

target_path = PREPROCESS_ROOT / 'ohlc_60m_downside_aware_tp3_v1_metadata.parquet'
target_schema_path = PREPROCESS_ROOT / 'ohlc_60m_downside_aware_tp3_v1_schema.json'
target_metadata.to_parquet(target_path, index=False, compression='zstd')
target_schema = {
    'version': 'ohlc_60m_downside_aware_tp3_v1',
    'base_version': DATA_VERSION,
    'horizon_minutes': 3,
    'upside_threshold': 0.03,
    'downside_threshold': 0.03,
    'target_rule': 'target_tp3_3m == 1 and -mae_3m < 0.03',
    'both_sides_rule': 'MFE >= 3% and downside >= 3% is negative',
    'intrabar_order_known': False,
    'changed_from_previous_experiment': 'target only',
}
target_schema_path.write_text(
    json.dumps(target_schema, ensure_ascii=False, indent=2), encoding='utf-8'
)

model_paths = {}
for strategy, model in final_models.items():
    path = MODEL_ROOT / f'{strategy}.pt'
    checkpoint = {
        'model_class': 'CompactModernTCN',
        'implementation_scope': 'same CompactModernTCN as notebook 06; target-only ablation',
        'strategy': strategy,
        'selected_epoch': selected_epochs[strategy],
        'model_config': {
            key: CONFIG[key]
            for key in [
                'input_features', 'context_features', 'd_model', 'd_ff', 'num_blocks',
                'patch_size', 'patch_stride', 'large_kernel_size',
                'small_kernel_size', 'dropout',
            ]
        },
        'output_dim': 1,
        'state_dict': {key: value.detach().cpu() for key, value in model.state_dict().items()},
        'input_scaler': scaler_to_serializable(final_input_scalers[strategy]),
        'target_scaler': None,
        'target_definition': target_schema,
        'sequence_features': schema['sequence_features'],
        'context_features': schema['context_features'],
        'data_version': DATA_VERSION,
        'train_sessions': fold_document['train_sessions'],
        'test_sessions': fold_document['test_sessions'],
        'test_is_pristine': False,
        'seed': SEED,
        'parameter_count': MODEL_PARAMETER_COUNT,
    }
    torch.save(checkpoint, path)
    model_paths[strategy] = path

artifact_paths = {
    'fold_metrics': RESULT_ROOT / 'oof_daily_metrics.parquet',
    'epoch_selection': RESULT_ROOT / 'oof_epoch_selection.parquet',
    'test_metrics': RESULT_ROOT / 'test_metrics.parquet',
    'training_history': RESULT_ROOT / 'training_history.parquet',
    'oof_predictions': RESULT_ROOT / 'oof_predictions.parquet',
    'test_predictions': RESULT_ROOT / 'test_predictions.parquet',
    'manifest': RESULT_ROOT / 'manifest.json',
}

fold_metrics.to_parquet(artifact_paths['fold_metrics'], index=False, compression='zstd')
epoch_selection.to_parquet(artifact_paths['epoch_selection'], index=False, compression='zstd')
test_metrics.to_parquet(artifact_paths['test_metrics'], index=False, compression='zstd')
training_history.to_parquet(artifact_paths['training_history'], index=False, compression='zstd')
oof_predictions.to_parquet(artifact_paths['oof_predictions'], index=False, compression='zstd')
test_predictions.to_parquet(artifact_paths['test_predictions'], index=False, compression='zstd')

manifest = {
    'experiment': 'moderntcn_downside_aware_tp3_v1',
    'created_by_notebook': 'notebooks/07_downside_aware_target.ipynb',
    'ablation': 'target only',
    'baseline_experiment': 'moderntcn_ohlc_60m_v1',
    'data_version': DATA_VERSION,
    'target_version': target_schema['version'],
    'device': str(DEVICE),
    'torch_version': torch.__version__,
    'seed': SEED,
    'parameter_count': MODEL_PARAMETER_COUNT,
    'config': CONFIG,
    'selected_epochs': selected_epochs,
    'selection_data': 'Train expanding walk-forward OOF only',
    'test_role': fold_document['test_role'],
    'test_is_pristine': False,
    'model_paths': {key: str(value) for key, value in model_paths.items()},
    'result_paths': {key: str(value) for key, value in artifact_paths.items()},
    'target_paths': {
        'metadata': str(target_path), 'schema': str(target_schema_path)
    },
}
artifact_paths['manifest'].write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
)

artifact_table = []
for name, path in {
    **model_paths, **artifact_paths,
    'target_metadata': target_path, 'target_schema': target_schema_path,
}.items():
    artifact_table.append({
        'artifact': name, 'path': str(path), 'size_mb': path.stat().st_size / 1024**2
    })
display(pd.DataFrame(artifact_table))
print('저장 완료')'''

nb.cells[16].source = r'''## 8. 기존 target 모델과 동일 정답에서 비교

기존 ModernTCN 예측을 downside-aware target으로 다시 채점하고, 새 target으로 재학습한 모델과 비교한다. 이렇게 해야 target 난이도 변화와 재학습 효과를 분리할 수 있다.'''

nb.cells[17].source = r'''from sklearn.metrics import average_precision_score, roc_auc_score, log_loss, brier_score_loss

BASELINE_ROOT = (PROJECT_ROOT / '../../results/training/moderntcn_ohlc_60m_v1').resolve()
baseline_oof = pd.read_parquet(BASELINE_ROOT / 'oof_predictions.parquet')
baseline_test = pd.read_parquet(BASELINE_ROOT / 'test_predictions.parquet')


def comparison_metrics(frame, probability_column):
    y = frame['target_downside_aware_tp3_3m'].to_numpy()
    p = frame[probability_column].to_numpy()
    downside = frame['downside_3m'].to_numpy()
    mfe = frame['mfe_3m'].to_numpy()
    return {
        'samples': len(frame),
        'prevalence': float(y.mean()),
        'PR_AUC': float(average_precision_score(y, p)),
        'PR_lift': float(average_precision_score(y, p) / y.mean()),
        'ROC_AUC': float(roc_auc_score(y, p)),
        'log_loss': float(log_loss(y, p, labels=[0, 1])),
        'brier': float(brier_score_loss(y, p)),
        'p_vs_mfe_spearman': safe_spearman(mfe, p),
        'p_vs_downside_spearman': safe_spearman(downside, p),
        'p_vs_excursion_utility_spearman': safe_spearman(mfe - downside, p),
        'prediction_mean': float(p.mean()),
    }


comparison_rows = []
for phase, new_frame, baseline_frame in [
    ('OOF', oof_predictions, baseline_oof),
    ('Test', test_predictions, baseline_test),
]:
    baseline = new_frame[[
        'sample_id', 'target_downside_aware_tp3_3m', 'mfe_3m',
        'downside_3m',
    ]].merge(
        baseline_frame[['sample_id', 'pred_open_hard_probability']],
        on='sample_id', how='left', validate='one_to_one'
    )
    assert baseline['pred_open_hard_probability'].notna().all()
    comparison_rows.append({
        'model': '기존 original-target ModernTCN', 'phase': phase,
        **comparison_metrics(baseline, 'pred_open_hard_probability'),
    })
    comparison_rows.append({
        'model': '새 downside-aware ModernTCN', 'phase': phase,
        **comparison_metrics(new_frame, 'pred_downside_aware_probability'),
    })

comparison = pd.DataFrame(comparison_rows)
display(comparison)

delta = comparison.pivot(index='phase', columns='model', values=[
    'PR_AUC', 'ROC_AUC', 'p_vs_downside_spearman',
    'p_vs_excursion_utility_spearman',
])
display(delta)
print('판정 기준: PR/ROC가 오르고 downside 상관이 내려가며 excursion utility 상관이 올라야 target 변경이 성공입니다.')'''

nb.cells[18].source = r'''## 9. 상위 점수 구간의 downside 변화

전체 PR-AUC뿐 아니라 새 모델의 상위 점수 구간에서 양방향 고변동 false positive가 실제로 줄었는지 확인한다.'''

nb.cells[19].source = r'''top_rows = []
for phase, new_frame, baseline_frame in [
    ('OOF', oof_predictions, baseline_oof),
    ('Test', test_predictions, baseline_test),
]:
    joined = new_frame.merge(
        baseline_frame[['sample_id', 'pred_open_hard_probability']],
        on='sample_id', how='left', validate='one_to_one'
    )
    for model_name, probability_column in [
        ('기존 original-target ModernTCN', 'pred_open_hard_probability'),
        ('새 downside-aware ModernTCN', 'pred_downside_aware_probability'),
    ]:
        threshold = joined[probability_column].quantile(0.95)
        selected = joined[joined[probability_column].ge(threshold)]
        false_positive = selected[selected['target_downside_aware_tp3_3m'].eq(0)]
        top_rows.append({
            'model': model_name,
            'phase': phase,
            'top_fraction': 0.05,
            'selected': len(selected),
            'precision': float(selected['target_downside_aware_tp3_3m'].mean()),
            'selected_downside_median': float(selected['downside_3m'].median()),
            'false_positive_downside_median': float(false_positive['downside_3m'].median()),
            'false_positive_downside_ge_3pct': float(
                false_positive['downside_3m'].ge(0.03).mean()
            ),
        })

top_score_comparison = pd.DataFrame(top_rows)
display(top_score_comparison)
print('다음 실험은 이 결과를 확정한 후, 같은 split에서 MFE-λ×downside utility를 직접 학습합니다.')'''

for cell in nb.cells:
    if cell.cell_type == 'code':
        cell.execution_count = None
        cell.outputs = []

nbf.write(nb, target_path)
print(target_path)
