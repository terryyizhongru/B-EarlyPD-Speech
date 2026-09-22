import argparse
import json
import os
import random
from pathlib import Path
import importlib

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score

# Keep imports consistent with existing pipeline utilities
from utils import build_model, get_dataloader, override_yaml, task_and_feature_filtering


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _select_threshold_max_pos_f1(y_true: np.ndarray, y_score: np.ndarray, num_thresholds: int = 101) -> float:
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_score = np.asarray(y_score).reshape(-1).astype(float)

    if y_true.size == 0 or y_score.size == 0:
        return 0.5
    if np.unique(y_true).size < 2:
        return 0.5

    thresholds = np.linspace(0.0, 1.0, int(num_thresholds))
    best_t = 0.5
    best_f1 = -1.0

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return float(best_t)


def _extract_binary_labels_and_scores(eval_output: dict):
    y_true = np.asarray(eval_output.get('labels', [])).reshape(-1)
    probs = np.asarray(eval_output.get('probs', []))

    if y_true.size == 0 or probs.size == 0:
        raise ValueError('Missing labels/probs in eval_output')

    uniq = np.unique(y_true)
    if uniq.size == 2 and not np.array_equal(uniq, np.array([0, 1])):
        y_true = (y_true == uniq.max()).astype(int)
    else:
        y_true = y_true.astype(int)

    if probs.ndim == 2 and probs.shape[1] == 2:
        y_score = probs[:, 1]
    elif probs.ndim == 1:
        y_score = probs
    else:
        raise ValueError(f'Unexpected probs shape for binary thresholding: {probs.shape}')

    return y_true, y_score


def _apply_threshold(eval_output: dict, threshold: float) -> dict:
    y_true, y_score = _extract_binary_labels_and_scores(eval_output)
    y_pred = (y_score >= float(threshold)).astype(int)

    eval_output['preds'] = y_pred.tolist()
    eval_output['threshold'] = float(threshold)
    eval_output['acc'] = float(accuracy_score(y_true, y_pred)) * 100.0
    eval_output['macro_f1'] = float(f1_score(y_true, y_pred, average='macro'))
    eval_output['pos_f1'] = float(f1_score(y_true, y_pred, pos_label=1))
    return eval_output


def _evaluate_via_pipeline_new(model, config, eval_loader, *, output_dir: str, is_test: bool) -> dict:
    """Use the repo's canonical evaluator from pipeline_new.py to guarantee identical outputs."""
    pipeline_new = importlib.import_module('pipeline_new')

    # pipeline_new.evaluate relies on module-level globals
    pipeline_new.model = model
    # Only fields accessed by evaluate(): save_embeddings + output_dir
    pipeline_new.args = argparse.Namespace(save_embeddings=False, output_dir=output_dir)
    return pipeline_new.evaluate(config, eval_loader, is_test=is_test)


def _write_allpd_results_tsv(test_dataset_path: str, test_output: dict, output_dir: str) -> str:
    if not str(test_dataset_path).lower().endswith('.tsv'):
        raise ValueError(f'Expected a .tsv test dataset for detailed export, got: {test_dataset_path}')

    tsv_df = pd.read_csv(test_dataset_path, sep='\t')
    if 'AUDIOFILE' not in tsv_df.columns:
        raise ValueError(f"Test TSV missing required column AUDIOFILE: {test_dataset_path}")

    sample_id_from_tsv = (
        tsv_df['AUDIOFILE']
        .astype(str)
        .str.split('/')
        .str[-1]
        .str.replace(r'\.[Ww][Aa][Vv]$', '', regex=True)
    )

    sample_ids = np.asarray(test_output.get('sample_id', [])).astype(str)
    probs = np.asarray(test_output.get('probs', []))
    labels = np.asarray(test_output.get('labels', [])).reshape(-1)
    preds = np.asarray(test_output.get('preds', [])).reshape(-1)

    if sample_ids.size == 0 or probs.size == 0 or labels.size == 0 or preds.size == 0:
        raise ValueError('Missing required keys in test_output for detailed TSV export')

    if probs.ndim == 2 and probs.shape[1] == 2:
        y_score = probs[:, 1]
    elif probs.ndim == 1:
        y_score = probs
    else:
        raise ValueError(f'Unexpected probs shape for binary detailed TSV export: {probs.shape}')

    uniq = np.unique(labels)
    if uniq.size == 2 and not np.array_equal(uniq, np.array([0, 1])):
        labels_01 = (labels == uniq.max()).astype(int)
    else:
        labels_01 = labels.astype(int)

    pred_df = pd.DataFrame({
        '_sample_id': sample_ids,
        'probability': y_score.astype(float),
        'true_label': labels_01.astype(int),
        'predicted_label': preds.astype(int),
    })

    merged = tsv_df.copy()
    merged['_sample_id'] = sample_id_from_tsv
    merged = merged.merge(pred_df, how='left', on='_sample_id')
    merged = merged.drop(columns=['_sample_id'])

    out_path = os.path.join(output_dir, 'AllPD_results.tsv')
    merged.to_csv(out_path, sep='\t', index=False)
    return out_path

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Re-infer: tune threshold on validation, then test inference (no training).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--training-dataset', required=True, type=str)
    parser.add_argument('--validation-dataset', required=True, type=str)
    parser.add_argument('--test-dataset', required=True, type=str)

    parser.add_argument('--load-checkpoint', required=True, type=str, help='Path to .pth state_dict (e.g., model_checkpoints/best_epoch_XXX.pth)')
    parser.add_argument('--output-dir', required=True, type=str)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threshold-grid', type=int, default=101)
    parser.add_argument('--skip-if-exists', action='store_true', help='If set, exit early when reinfer outputs already exist')

    parser.add_argument('--exclude-features', nargs='+', default=['none'])
    parser.add_argument('--filter-tasks', nargs='+', default=['ALL'])
    parser.add_argument('--filter-evaluation-tasks', nargs='+', default=['none'])
    parser.add_argument('--yaml-overrides', metavar='CONF:[KEY]:VALUE', nargs='*')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tuned_json_path = os.path.join(args.output_dir, 'tuned_threshold_reinfer.json')
    allpd_tsv_path = os.path.join(args.output_dir, 'AllPD_results.tsv')
    if args.skip_if_exists and (os.path.exists(tuned_json_path) or os.path.exists(allpd_tsv_path)):
        print(f"[SKIP] Exists: {tuned_json_path} or {allpd_tsv_path}")
        raise SystemExit(0)

    # Determinism basics
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load config
    config_file = Path(args.config)
    with config_file.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg = override_yaml(cfg, args.yaml_overrides)
    config = argparse.Namespace(**cfg)

    # Ensure device is present
    if not hasattr(config, 'device'):
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    diff_eval = 'none' not in args.filter_evaluation_tasks
    same_config = task_and_feature_filtering(config, args.filter_tasks, args.exclude_features)
    evaluation_config = task_and_feature_filtering(config, args.filter_evaluation_tasks, args.exclude_features) if diff_eval else None

    model = build_model(config)

    checkpoint = torch.load(args.load_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.to(device=config.device)

    train_loader = get_dataloader(same_config, args.training_dataset, is_training=True)

    val_loader = get_dataloader(evaluation_config if diff_eval else same_config, args.validation_dataset, is_training=False)
    val_loader.dataset.feature_norm_stats = train_loader.dataset.feature_norm_stats

    test_loader = get_dataloader(evaluation_config if diff_eval else same_config, args.test_dataset, is_training=False)
    test_loader.dataset.feature_norm_stats = train_loader.dataset.feature_norm_stats

    # Use the exact same evaluator as training/evaluation pipeline
    val_output = _evaluate_via_pipeline_new(model, config, val_loader, output_dir=args.output_dir, is_test=False)
    test_output = _evaluate_via_pipeline_new(model, config, test_loader, output_dir=args.output_dir, is_test=True)

    # Threshold tuning and application (binary only)
    num_classes = getattr(config, 'num_classes', None)
    task_type = getattr(config, 'task_type', 'classification')
    if task_type != 'classification' or num_classes != 2:
        raise ValueError(f'Reinfer threshold expects binary classification (task_type=classification, num_classes=2). Got task_type={task_type}, num_classes={num_classes}')

    y_true, y_score = _extract_binary_labels_and_scores(val_output)
    tuned_threshold = _select_threshold_max_pos_f1(y_true, y_score, num_thresholds=args.threshold_grid)

    payload = {
        'threshold': float(tuned_threshold),
        'strategy': 'max_pos_f1',
        'grid': int(args.threshold_grid),
        'seed': int(args.seed),
        'checkpoint': str(args.load_checkpoint),
        'validation_dataset': str(args.validation_dataset),
        'test_dataset': str(args.test_dataset),
    }

    with open(tuned_json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    _apply_threshold(val_output, tuned_threshold)
    _apply_threshold(test_output, tuned_threshold)

    out_path = _write_allpd_results_tsv(args.test_dataset, test_output, args.output_dir)

    print(f"[THRESHOLD] tuned={tuned_threshold:.6f} -> {tuned_json_path}")
    print(f"[DETAILED] {out_path}")


if __name__ == '__main__':
    main()
