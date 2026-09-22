import sys
import yaml
import torch
import random
import argparse
import numpy as np
import os
import re
import json
import pandas as pd
from utils import *
from tqdm import tqdm
from pathlib import Path
from colorama import Fore
from distutils.util import strtobool
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error, f1_score, roc_auc_score


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _select_threshold_max_pos_f1(y_true: np.ndarray, y_score: np.ndarray, num_thresholds: int = 101) -> float:
    """Pick a single threshold that maximizes positive-class F1 on (y_true, y_score)."""
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


def _extract_binary_labels_and_scores_from_eval_output(eval_output: dict):
    """Return (y_true_01, y_score_pos) for binary classification from eval_output."""
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


def _apply_binary_threshold_to_eval_output(eval_output: dict, threshold: float) -> dict:
    """Mutates eval_output: replaces preds using threshold on positive score; updates acc/macro_f1."""
    y_true, y_score = _extract_binary_labels_and_scores_from_eval_output(eval_output)
    y_pred = (np.asarray(y_score) >= float(threshold)).astype(int)

    eval_output['preds'] = y_pred.tolist()
    eval_output['threshold'] = float(threshold)
    # Recompute summary metrics at dataset-level
    eval_output['acc'] = float(accuracy_score(y_true, y_pred)) * 100.0
    eval_output['macro_f1'] = float(f1_score(y_true, y_pred, average='macro'))
    eval_output['pos_f1'] = float(f1_score(y_true, y_pred, pos_label=1))
    return eval_output


def _write_test_detailed_results_tsv(test_dataset_path: str, test_output: dict, output_dir: str) -> None:
    """Create a detailed results TSV by appending columns to the input test TSV.

    Adds columns:
      - probability: positive-class probability score (for binary classification)
      - true_label: 0/1 label
      - predicted_label: 0/1 prediction (after threshold application)
    """
    if not str(test_dataset_path).lower().endswith('.tsv'):
        # Only implement TSV detailed export (matches user's reference file format).
        return

    tsv_df = pd.read_csv(test_dataset_path, sep='\t')
    if 'AUDIOFILE' not in tsv_df.columns:
        raise ValueError(f"Test TSV missing required column AUDIOFILE: {test_dataset_path}")

    # Derive sample_id the same way as ParkinsonDataset TSV conversion (basename without extension)
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

    # Positive-class probability used for AUC and thresholding
    if probs.ndim == 2 and probs.shape[1] == 2:
        y_score = probs[:, 1]
    elif probs.ndim == 1:
        y_score = probs
    else:
        raise ValueError(f'Unexpected probs shape for binary detailed TSV export: {probs.shape}')

    # Normalize labels to {0,1} if needed
    uniq = np.unique(labels)
    if uniq.size == 2 and not np.array_equal(uniq, np.array([0, 1])):
        labels_01 = (labels == uniq.max()).astype(int)
    else:
        labels_01 = labels.astype(int)

    preds_01 = preds.astype(int)

    pred_df = pd.DataFrame({
        '_sample_id': sample_ids,
        'probability': y_score.astype(float),
        'true_label': labels_01.astype(int),
        'predicted_label': preds_01.astype(int),
    })

    merged = tsv_df.copy()
    merged['_sample_id'] = sample_id_from_tsv
    merged = merged.merge(pred_df, how='left', on='_sample_id')
    merged = merged.drop(columns=['_sample_id'])

    out_path = os.path.join(output_dir, 'Neurovoz_and_PC_GITA_detailed_results.tsv')
    merged.to_csv(out_path, sep='\t', index=False)
    print(f"[DETAILED] Saved test detailed results to: {out_path}")


def get_free_gpu():
    """
    Automatically detect and return the GPU with the most free memory.
    Returns the GPU ID as a string, or None if no GPU is available.
    """
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free,memory.total,utilization.gpu',
                               '--format=csv,nounits,noheader'],
                              capture_output=True, text=True, check=True)

        lines = result.stdout.strip().split('\n')
        max_free_memory = -1
        best_gpu = None

        print("GPU Status:")
        for line in lines:
            parts = line.split(', ')
            gpu_id = int(parts[0])
            free_memory = int(parts[1])
            total_memory = int(parts[2])
            gpu_util = int(parts[3])

            print(f"GPU {gpu_id}: Free {free_memory}MB/{total_memory}MB, Utilization: {gpu_util}%")

            # Select GPU with most free memory and low utilization
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_gpu = str(gpu_id)

        if best_gpu is not None:
            print(f"Selected GPU {best_gpu} with {max_free_memory}MB free memory")
            return best_gpu
        else:
            print("No GPU available")
            return None

    except (subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
        print(f"Error detecting GPU: {e}")
        print("Falling back to GPU 0")
        return "0"

# Auto-select the best available GPU
selected_gpu = get_free_gpu()
if selected_gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpu
    print(f"Using GPU {selected_gpu}")
else:
    print("No GPU selected, will use CPU")


def train(config):
    model.train()
    optimizer.zero_grad()

    train_output = {'loss': 0.0}

    # Initialize metric based on task type
    task_type = getattr(config, 'task_type', 'classification')
    if task_type == 'regression':
        train_output['mse'] = 0.0
        train_output['mae'] = 0.0
    else:
        train_output['acc'] = 0.0
        train_output['macro_f1'] = 0.0

    accum_grad = config.training_settings['accum_grad']
    for batch_idx, batch in enumerate(tqdm(train_loader, position=0, leave=True, file=sys.stdout, bar_format="{l_bar}%s{bar:10}%s{r_bar}" % (Fore.GREEN, Fore.RESET))):
        batch = {k: v.to(device=config.device, non_blocking=True) if hasattr(v, 'to') else v for k, v in batch.items()}

        # -- forward pass
        model_output = model(batch)
        loss = model_output['loss'] / accum_grad

        # -- optimization
        loss.backward()
        if ((batch_idx+1) % accum_grad == 0) or (batch_idx+1 == len(train_loader)):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        train_output['loss'] += loss.item()
        # Debug: Check if labels are all zero in first batch
        if batch_idx == 0:
            labels_debug = model_output['labels'].detach().cpu().numpy()
            if np.all(labels_debug == 0):
                print(f"WARNING: All training labels are zero!")
                print(f"Task type: {task_type}")
                print(f"Config target_label: {getattr(config, 'target_label', 'label')}")
                if 'normalized_labels' in model_output:
                    print(f"Normalized labels: {model_output['normalized_labels'].detach().cpu().numpy()}")
            else:
                print(f"Training labels range [{labels_debug.min():.3f}, {labels_debug.max():.3f}]")

        # print('Predictions:', model_output['preds'].detach().cpu().numpy())
        # print('Labels:', model_output['labels'].detach().cpu().numpy())
        # Calculate metrics based on task type
        if task_type == 'regression':
            # Check if this is UPDRS regression with normalized values
            if hasattr(model_output, '__contains__') and 'normalized_preds' in model_output:
                # For UPDRS: use normalized values for loss-related metrics (MSE)
                normalized_preds = model_output['normalized_preds'].detach().cpu().numpy()
                normalized_labels = model_output['normalized_labels'].detach().cpu().numpy()
                train_output['mse'] += mean_squared_error(normalized_labels, normalized_preds)

                # Use denormalized values for reference metrics (MAE)
                denorm_preds = model_output['preds'].detach().cpu().numpy()
                denorm_labels = model_output['labels'].detach().cpu().numpy()
                train_output['mae'] += mean_absolute_error(denorm_labels, denorm_preds)
            else:
                # For non-UPDRS regression tasks, use regular approach
                preds = model_output['preds'].detach().cpu().numpy()
                labels = model_output['labels'].detach().cpu().numpy()
                train_output['mse'] += mean_squared_error(labels, preds)
                train_output['mae'] += mean_absolute_error(labels, preds)
        else:
            preds = model_output['preds'].detach().cpu().numpy()
            labels = model_output['labels'].detach().cpu().numpy()
            train_output['acc'] += accuracy_score(labels, preds)
            train_output['macro_f1'] += f1_score(labels, preds, average='macro')

    train_output['loss'] = train_output['loss'] / (len(train_loader) / accum_grad)

    if task_type == 'regression':
        train_output['mse'] = train_output['mse'] / len(train_loader)
        train_output['mae'] = train_output['mae'] / len(train_loader)
    else:
        train_output['acc'] = (train_output['acc'] / len(train_loader)) * 100.0
        train_output['macro_f1'] = train_output['macro_f1'] / len(train_loader)

    return train_output

def evaluate(config, eval_loader, is_test=False):
    model.eval()
    task_type = getattr(config, 'task_type', 'classification')

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, position=0, leave=True, file=sys.stdout, bar_format="{l_bar}%s{bar:10}%s{r_bar}" % (Fore.BLUE, Fore.RESET))):
            batch = {k: v.to(device=config.device, non_blocking=True) if hasattr(v, 'to') else v for k, v in batch.items()}

            # -- forward pass
            model_output = model(batch)

            # -- evaluation output initialization
            if batch_idx == 0:
                eval_output = {model_output_key:[] for model_output_key in list(model_output.keys())}
                # Add metric keys based on task type
                if task_type == 'regression':
                    eval_output['mse'] = []
                    eval_output['mae'] = []
                else:
                    eval_output['acc'] = []
                    eval_output['macro_f1'] = []
            # print('Predictions:', model_output['preds'].detach().cpu().numpy())
            # print('Labels:', model_output['labels'].detach().cpu().numpy())

            # Debug: Check if all labels are zero
            labels_debug = model_output['labels'].detach().cpu().numpy()
            if np.all(labels_debug == 0):
                print(f"WARNING: All labels are zero in batch {batch_idx}!")
                print(f"Task type: {task_type}")
                print(f"Config target_label: {getattr(config, 'target_label', 'label')}")
                if 'normalized_labels' in model_output:
                    print(f"Normalized labels: {model_output['normalized_labels'].detach().cpu().numpy()}")
                print("First few samples from batch:")
                for i, (subj, sample) in enumerate(zip(batch['subject_id'][:3], batch['sample_id'][:3])):
                    print(f"  Sample {i}: subject={subj}, sample_id={sample}")
            else:
                print(f"Batch {batch_idx}: Labels range [{labels_debug.min():.3f}, {labels_debug.max():.3f}]")
            # -- gathering statistics
            for eval_key in eval_output.keys():
                if eval_key == 'loss':
                    eval_output[eval_key] += [model_output['loss'].item()]
                elif eval_key == 'acc' and task_type != 'regression':
                    eval_output['acc'] += [accuracy_score(
                        model_output['preds'].detach().cpu().numpy(),
                        model_output['labels'].detach().cpu().numpy(),
                    )]
                elif eval_key == 'macro_f1' and task_type != 'regression':
                    eval_output['macro_f1'] += [f1_score(
                        model_output['labels'].detach().cpu().numpy(),
                        model_output['preds'].detach().cpu().numpy(),
                        average='macro'
                    )]
                elif eval_key == 'mse' and task_type == 'regression':
                    # Check if this is UPDRS regression with normalized values
                    if hasattr(model_output, '__contains__') and 'normalized_preds' in model_output:
                        # For UPDRS: use normalized values for MSE (consistent with loss)
                        normalized_preds = model_output['normalized_preds'].detach().cpu().numpy()
                        normalized_labels = model_output['normalized_labels'].detach().cpu().numpy()
                        eval_output['mse'] += [mean_squared_error(normalized_labels, normalized_preds)]
                    else:
                        # For non-UPDRS regression tasks
                        preds = model_output['preds'].detach().cpu().numpy()
                        labels = model_output['labels'].detach().cpu().numpy()
                        eval_output['mse'] += [mean_squared_error(labels, preds)]
                elif eval_key == 'mae' and task_type == 'regression':
                    # Always use denormalized values for MAE (reference metric)
                    preds = model_output['preds'].detach().cpu().numpy()
                    labels = model_output['labels'].detach().cpu().numpy()
                    eval_output['mae'] += [mean_absolute_error(labels, preds)]
                elif eval_key == 'cross_time_mha_scores':
                    time_mha_scores = []
                    for i, time_mha_score_sample in enumerate(model_output[eval_key].detach().cpu().numpy()):
                        time_mha_scores.append( time_mha_score_sample[:, :batch['ssl_lengths'][i], :] )
                    eval_output[eval_key] += time_mha_scores
                elif eval_key in ['self_inf_mha_scores', 'self_ssl_mha_scores', 'cross_embed_mha_scores']:
                    eval_output[eval_key] += [mha_score_sample.detach().cpu().numpy() for mha_score_sample in model_output[eval_key]]
                elif eval_key not in ['mse', 'mae', 'acc']:  # Skip metric keys if not in model_output
                    if eval_key in model_output:
                        eval_output[eval_key] += model_output[eval_key] if not hasattr(model_output[eval_key], 'to') else model_output[eval_key].detach().cpu().numpy().tolist()

            # -- saving embeddings for further analysis
            if is_test and args.save_embeddings:
                for sample_id, embedding in zip(batch['sample_id'], model_output['embeddings']):
                    save_embedding(embedding.detach().cpu().numpy(), args.output_dir, sample_id)

    eval_output['informed_metadata'] = eval_loader.dataset.informed_metadata
    eval_output['loss'] = np.array(eval_output['loss']).mean()

    if task_type == 'regression':
        eval_output['mse'] = np.array(eval_output['mse']).mean()
        eval_output['mae'] = np.array(eval_output['mae']).mean()
    else:
        eval_output['acc'] = np.array(eval_output['acc']).mean() * 100.0
        eval_output['macro_f1'] = np.array(eval_output['macro_f1']).mean()

    return eval_output

def train_single_inner_fold(config, train_dataset, val_dataset, output_dir):
    """Train a single inner fold and return best validation performance"""

    # -- setting seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # -- task and feature filtering
    same_config = task_and_feature_filtering(config, ['ALL'], ['none'])

    # -- building model architecture
    model = build_model(config)
    print(f'Inner fold model parameters: {sum([param.nelement() for param in model.parameters()])}')

    # -- creating dataloaders
    train_loader = get_dataloader(same_config, train_dataset, is_training=True)
    val_loader = get_dataloader(same_config, val_dataset, is_training=False)
    val_loader.dataset.feature_norm_stats = train_loader.dataset.feature_norm_stats

    # -- defining the optimizer and its scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(config, model, train_loader)

    best_val_metric = float('-inf') if getattr(config, 'task_type', 'classification') != 'regression' else float('inf')
    best_epoch = 0
    best_model_state = None
    task_type = getattr(config, 'task_type', 'classification')

    # -- training process
    for epoch in range(1, config.training_settings['epochs']+1):
        # Set global variables for train function
        globals()['model'] = model
        globals()['optimizer'] = optimizer
        globals()['scheduler'] = scheduler
        globals()['train_loader'] = train_loader

        train_stats = train(config)
        val_output = evaluate(config, val_loader)

        # Determine validation metric for early stopping (only after epoch 3 to prevent underfit)
        if epoch >= 3:
            if task_type == 'regression':
                val_metric = val_output['mse']  # Lower is better for MSE
                is_better = val_metric <= best_val_metric
            else:
                val_metric = val_output['macro_f1']  # Higher is better for macro F1
                is_better = val_metric >= best_val_metric

            if is_better:
                best_val_metric = val_metric
                best_epoch = epoch
                best_model_state = model.state_dict().copy()

        # Display metrics
        if task_type == 'regression':
            # Check if we have normalized metrics (UPDRS case)
            config_target = getattr(config, 'target_label', 'label')
            if config_target == 'UPDRS':
                print(f"Epoch {epoch}: TRAIN LOSS={round(train_stats['loss'],4)} TRAIN MSE(norm)={round(train_stats['mse'],4)} TRAIN MAE(orig)={round(train_stats['mae'],1)} || VAL LOSS={round(val_output['loss'],4)} VAL MSE(norm)={round(val_output['mse'],4)} VAL MAE(orig)={round(val_output['mae'],1)}")
            else:
                print(f"Epoch {epoch}: TRAIN LOSS={round(train_stats['loss'],4)} TRAIN MSE={round(train_stats['mse'],4)} TRAIN MAE={round(train_stats['mae'],4)} || VAL LOSS={round(val_output['loss'],4)} VAL MSE={round(val_output['mse'],4)} VAL MAE={round(val_output['mae'],4)}")
        else:
            print(f"Epoch {epoch}: TRAIN LOSS={round(train_stats['loss'],2)} TRAIN ACC={round(train_stats['acc'],4)}% TRAIN F1={round(train_stats['macro_f1'],4)} || VAL LOSS={round(val_output['loss'],4)} VAL ACC={round(val_output['acc'],2)}% VAL F1={round(val_output['macro_f1'],4)}")

    # Save best model
    if best_model_state is not None:
        save_checkpoint_state(best_model_state, output_dir, f'best_model_epoch_{str(best_epoch).zfill(3)}')

    print(f"Best validation {'MSE' if task_type == 'regression' else 'Macro F1'}: {best_val_metric} at epoch {best_epoch}")

    return {
        'best_val_metric': best_val_metric,
        'best_epoch': best_epoch,
        'best_model_state': best_model_state,
        'task_type': task_type
    }

def save_checkpoint_state(state_dict, output_dir, checkpoint_name):
    """Save model state dict to checkpoint file"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f'{checkpoint_name}.pth')
    torch.save(state_dict, checkpoint_path)
    print(f'Saved checkpoint: {checkpoint_path}')

def pipeline_with_inner_cv(args, config, return_dicts=False):
    """Pipeline with inner cross-validation for model selection"""

    # Extract fold information from test dataset path (since we only need test dataset)
    import re
    fold_match = re.search(r'fold_(\d+)', args.test_dataset)
    if not fold_match:
        raise ValueError("Could not extract fold number from test dataset path")

    outer_fold = fold_match.group(1)
    base_splits_dir = args.test_dataset.split(f'fold_{outer_fold}')[0]
    fold_dir = f"{base_splits_dir}fold_{outer_fold}"

    print(f"Running inner CV for outer fold {outer_fold}")
    print(f"Using fold directory: {fold_dir}")

    # Run inner cross-validation
    inner_results = []
    for inner_fold in range(1, 6):  # inner folds 1-5
        print(f"\n=== Inner fold {inner_fold} ===")

        inner_train_dataset = f"{fold_dir}/innerfold_{inner_fold}/train.csv"
        inner_val_dataset = f"{fold_dir}/innerfold_{inner_fold}/val.csv"
        inner_output_dir = f"{args.output_dir}/inner_fold_{inner_fold}"

        # Check if inner fold datasets exist
        if not os.path.exists(inner_train_dataset) or not os.path.exists(inner_val_dataset):
            print(f"Skipping inner fold {inner_fold}: datasets not found")
            continue

        result = train_single_inner_fold(config, inner_train_dataset, inner_val_dataset, inner_output_dir)
        result['inner_fold'] = inner_fold
        result['output_dir'] = inner_output_dir
        inner_results.append(result)

    if not inner_results:
        raise ValueError("No inner fold results available")

    # Select best inner fold based on validation metric
    task_type = inner_results[0]['task_type']
    if task_type == 'regression':
        # For regression, lower MSE is better
        best_inner = min(inner_results, key=lambda x: x['best_val_metric'])
    else:
        # For classification, higher Macro F1 is better
        best_inner = max(inner_results, key=lambda x: x['best_val_metric'])

    print(f"\n=== Best inner fold: {best_inner['inner_fold']} with {'MSE' if task_type == 'regression' else 'Macro F1'}: {best_inner['best_val_metric']} ===")

    # Load best model and evaluate on test set
    model = build_model(config)
    model.load_state_dict(best_inner['best_model_state'])

    # -- task and feature filtering for evaluation
    same_config = task_and_feature_filtering(config, ['ALL'], ['none'])

    diff_eval = False
    if 'none' not in args.filter_evaluation_tasks:
        diff_eval = True
        evaluation_config = task_and_feature_filtering(config, args.filter_evaluation_tasks, ['none'])

    # Create test dataloader using the original test dataset
    test_loader = get_dataloader(evaluation_config if diff_eval else same_config, args.test_dataset, is_training=False)

    # We need feature normalization stats from one of the inner training sets
    temp_train_loader = get_dataloader(same_config, f"{fold_dir}/innerfold_1/train.csv", is_training=True)
    test_loader.dataset.feature_norm_stats = temp_train_loader.dataset.feature_norm_stats

    # Set global model variable for evaluate function
    globals()['model'] = model

    # Evaluate on test set
    test_output = evaluate(config, test_loader, is_test=True)

    # Save test results
    if args.save_output:
        save_model_output(test_output, args.output_dir, 'test', save_attention_scores=args.save_attention_scores)

    # Save best model to main output directory
    save_checkpoint_state(best_inner['best_model_state'], args.output_dir, f'best_model_from_inner_fold_{best_inner["inner_fold"]}')

    # Create reports
    if task_type == 'regression':
        test_report = {
            'mse': test_output['mse'],
            'mae': test_output['mae'],
            'loss': test_output['loss']
        }
        val_report = None
    else:
        # For classification
        num_classes = getattr(config, 'num_classes', None)
        labels_param = list(range(num_classes)) if num_classes else None

        test_report = classification_report(
            test_output['labels'],
            test_output['preds'],
            labels=labels_param,
            target_names=getattr(config, 'class_names', None),
            output_dict=return_dicts,
        )
        val_report = None

    # Add inner CV summary to results
    inner_cv_summary = {
        'best_inner_fold': best_inner['inner_fold'],
        'best_inner_val_metric': best_inner['best_val_metric'],
        'best_inner_epoch': best_inner['best_epoch'],
        'all_inner_results': [(r['inner_fold'], r['best_val_metric'], r['best_epoch']) for r in inner_results]
    }

    return val_report, test_report, inner_cv_summary

def pipeline(args, config, return_dicts=False):
    """Original pipeline function - kept for backward compatibility"""

    # Check if inner CV mode is requested
    if getattr(args, 'use_inner_cv', False):
        return pipeline_with_inner_cv(args, config, return_dicts)

    # -- setting seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # Initialize return variables
    val_report, test_report = None, None

    # -- task and feature filtering
    print(config)
    print(f'Using the following features: {config.features}')
    same_config = task_and_feature_filtering(config, args.filter_tasks, args.exclude_features)

    diff_eval = False
    if 'none' not in args.filter_evaluation_tasks:
        diff_eval = True
        evaluation_config = task_and_feature_filtering(config, args.filter_evaluation_tasks, args.exclude_features)

    # -- building model architecture
    global model, optimizer, scheduler, train_loader, val_loader
    model = build_model(config)
    # print(model)
    # print(f'Model Parameters: {sum([param.nelement() for param in model.parameters()])}')
    # -- loading model checkpoint
    if args.load_checkpoint:
        print(f'Loading checkpoint from {args.load_checkpoint} ...')
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint)

    # -- creating dataloaders
    train_loader = get_dataloader(same_config, args.training_dataset, is_training=True)

    val_loader = get_dataloader(evaluation_config if diff_eval else same_config, args.validation_dataset, is_training=False)
    val_loader.dataset.feature_norm_stats = train_loader.dataset.feature_norm_stats

    test_loader = get_dataloader(evaluation_config if diff_eval else same_config, args.test_dataset, is_training=False)
    test_loader.dataset.feature_norm_stats = train_loader.dataset.feature_norm_stats

    # -- training process
    if args.mode in ['training', 'both']:

        # -- early stopping / model selection (default mode)
        # Minimal, fixed policy as requested:
        # - metric: ROC AUC on validation after every epoch
        # - min epoch: 5, max epoch: 20
        # - patience: 5 epochs without improvement
        min_epochs = 5
        max_epochs = 20
        patience = 5

        # OneCycleLR is initialized with a fixed number of epochs/steps.
        # Since we can run up to `max_epochs` before early stopping, we must
        # configure the scheduler for `epochs=max_epochs` to avoid stepping past
        # its total_steps (the error you saw at epoch 6).
        original_epochs = config.training_settings.get('epochs', None)
        config.training_settings['epochs'] = max_epochs

        # -- defining the optimizer and its scheduler
        optimizer, scheduler = get_optimizer_and_scheduler(config, model, train_loader)

        best_val_auc = float('-inf')
        best_epoch = 0
        best_state_dict = None
        epochs_no_improve = 0

        for epoch in range(1, max_epochs + 1):
            train_stats = train(config)
            val_output = evaluate(config, val_loader)

            # -- compute ROC AUC from probabilities
            # Models in this repo provide model_output['probs'] for classification.
            val_auc = None
            try:
                y_true = np.asarray(val_output.get('labels', []))
                y_prob = np.asarray(val_output.get('probs', []))

                if y_true.size == 0 or y_prob.size == 0:
                    raise ValueError('Missing labels/probs for ROC-AUC computation')

                # Ensure 1D labels
                y_true = y_true.reshape(-1)

                # Normalize binary labels to {0,1} if they come as {1,2} or any two distinct values.
                uniq = np.unique(y_true)
                if uniq.size == 2 and not np.array_equal(uniq, np.array([0, 1])):
                    # Map smaller value -> 0, larger value -> 1
                    y_true = (y_true == uniq.max()).astype(int)

                # Extract positive-class score
                if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    y_score = y_prob[:, 1]
                elif y_prob.ndim == 1:
                    # Already a single score per sample
                    y_score = y_prob
                else:
                    raise ValueError(f'Unexpected probs shape for binary ROC-AUC: {y_prob.shape}')

                # roc_auc_score requires both classes present
                if np.unique(y_true).size < 2:
                    raise ValueError('Only one class present in y_true')

                val_auc = float(roc_auc_score(y_true, y_score))
            except Exception as e:
                # Typical failure: only one class present in y_true
                val_auc = float('-inf')
                print(f"WARNING: Could not compute ROC-AUC on validation at epoch {epoch}: {e}")

            print(f"Epoch {epoch}: VAL ROC-AUC={val_auc:.6f}")

            # -- saving model checkpoint
            save_checkpoint(model, args.output_dir, f'epoch_{str(epoch).zfill(3)}')

            # -- track best checkpoint by ROC-AUC
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0

                # Save best checkpoint
                save_checkpoint(model, args.output_dir, f'best_epoch_{str(best_epoch).zfill(3)}')
            else:
                epochs_no_improve += 1

            # -- early stopping condition
            if epoch >= min_epochs and epochs_no_improve >= patience:
                print(
                    f"Early stopping at epoch {epoch} (no ROC-AUC improvement for {patience} epochs). "
                    f"Best epoch: {best_epoch} with ROC-AUC={best_val_auc:.6f}"
                )
                break

            # Display metrics based on task type
            task_type = getattr(config, 'task_type', 'classification')
            if task_type == 'regression':
                # Check if we have normalized metrics (UPDRS case)
                config_target = getattr(config, 'target_label', 'label')
                if config_target == 'UPDRS':
                    print(f"Epoch {epoch}: TRAIN LOSS={round(train_stats['loss'],4)} TRAIN MSE(norm)={round(train_stats['mse'],4)} TRAIN MAE(orig)={round(train_stats['mae'],1)} || VAL LOSS={round(val_output['loss'],4)} VAL MSE(norm)={round(val_output['mse'],4)} VAL MAE(orig)={round(val_output['mae'],1)}")
                else:
                    print(f"Epoch {epoch}: TRAIN LOSS={round(train_stats['loss'],4)} TRAIN MSE={round(train_stats['mse'],4)} TRAIN MAE={round(train_stats['mae'],4)} || VAL LOSS={round(val_output['loss'],4)} VAL MSE={round(val_output['mse'],4)} VAL MAE={round(val_output['mae'],4)}")
            else:
                print(f"Epoch {epoch}: TRAIN LOSS={round(train_stats['loss'],2)} TRAIN ACC={round(train_stats['acc'],4)}% TRAIN F1={round(train_stats['macro_f1'],4)} || VAL LOSS={round(val_output['loss'],4)} VAL ACC={round(val_output['acc'],2)}% VAL F1={round(val_output['macro_f1'],4)}")

        # -- after training, keep only the best checkpoint and load best weights for evaluation
        # Restore original YAML epochs value (scheduler already constructed).
        if original_epochs is not None:
            config.training_settings['epochs'] = original_epochs

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

            ckpt_dir = os.path.join(args.output_dir, 'model_checkpoints')
            best_fname = f"best_epoch_{str(best_epoch).zfill(3)}.pth"
            best_path = os.path.join(ckpt_dir, best_fname)

            # Remove all other epoch checkpoints, keep only the best checkpoint file.
            try:
                if os.path.isdir(ckpt_dir):
                    for fname in os.listdir(ckpt_dir):
                        fpath = os.path.join(ckpt_dir, fname)
                        if not os.path.isfile(fpath):
                            continue
                        if fname == best_fname:
                            continue
                        if fname.startswith('epoch_') and fname.endswith('.pth'):
                            os.remove(fpath)
                        elif fname.startswith('best_epoch_') and fname.endswith('.pth') and fpath != best_path:
                            os.remove(fpath)
            except Exception as e:
                print(f"WARNING: Failed to clean up checkpoints in {ckpt_dir}: {e}")

    if args.mode in ['evaluation', 'both']:

        val_output = evaluate(config, val_loader)
        test_output = evaluate(config, test_loader, is_test=True)

        # --- Binary classification threshold tuning (after best checkpoint)
        # Reference: BDHPD/train.py style: choose threshold maximizing positive-class F1 on validation,
        # then apply this threshold to test predictions.
        task_type = getattr(config, 'task_type', 'classification')
        num_classes = getattr(config, 'num_classes', None)
        if task_type == 'classification' and num_classes == 2:
            thresholds_path = os.path.join(args.output_dir, 'tuned_thresholds.json')

            tuned_threshold = None
            tuned_strategy = 'max_pos_f1'
            tuned_grid = 101

            # If running evaluation-only, try to reuse saved threshold.
            if args.mode == 'evaluation' and os.path.isfile(thresholds_path):
                try:
                    with open(thresholds_path, 'r', encoding='utf-8') as f:
                        payload = json.load(f)
                    tuned_threshold = float(payload.get('threshold', 0.5))
                    tuned_strategy = str(payload.get('strategy', tuned_strategy))
                    tuned_grid = int(payload.get('grid', tuned_grid))
                    print(f"[THRESHOLD] Loaded tuned threshold={tuned_threshold:.6f} from {thresholds_path}")
                except Exception as e:
                    print(f"WARNING: Failed to load {thresholds_path}: {e}")
                    tuned_threshold = None

            # Otherwise (or if load failed), tune on current validation output.
            if tuned_threshold is None:
                try:
                    y_true, y_score = _extract_binary_labels_and_scores_from_eval_output(val_output)
                    tuned_threshold = _select_threshold_max_pos_f1(y_true, y_score, num_thresholds=tuned_grid)

                    # Save threshold for reproducibility
                    os.makedirs(args.output_dir, exist_ok=True)
                    with open(thresholds_path, 'w', encoding='utf-8') as f:
                        json.dump(
                            {
                                'threshold': float(tuned_threshold),
                                'strategy': tuned_strategy,
                                'grid': tuned_grid,
                            },
                            f,
                            indent=2,
                            sort_keys=True,
                        )
                    print(f"[THRESHOLD] Tuned on val: thr={tuned_threshold:.6f} (strategy={tuned_strategy}, grid={tuned_grid})")
                    print(f"[THRESHOLD] Saved tuned thresholds to: {thresholds_path}")
                except Exception as e:
                    print(f"WARNING: Threshold tuning failed; using 0.5. Reason: {e}")
                    tuned_threshold = 0.5

            # Apply to val/test outputs (so reports use thresholded preds)
            try:
                _apply_binary_threshold_to_eval_output(val_output, tuned_threshold)
                _apply_binary_threshold_to_eval_output(test_output, tuned_threshold)
            except Exception as e:
                print(f"WARNING: Failed to apply tuned threshold to outputs: {e}")

            # Write detailed TSV results based on the input test TSV
            try:
                _write_test_detailed_results_tsv(args.test_dataset, test_output, args.output_dir)
            except Exception as e:
                print(f"WARNING: Failed to write detailed test TSV: {e}")

        if args.save_output:
            save_model_output(val_output, args.output_dir, 'validation', save_attention_scores=args.save_attention_scores)
            save_model_output(test_output, args.output_dir, 'test', save_attention_scores=args.save_attention_scores)

        # -- displaying final report
        task_type = getattr(config, 'task_type', 'classification')

        if task_type == 'regression':
            # For regression, create simple metric reports
            val_report = {
                'mse': val_output['mse'],
                'mae': val_output['mae'],
                'loss': val_output['loss']
            }
            test_report = {
                'mse': test_output['mse'],
                'mae': test_output['mae'],
                'loss': test_output['loss']
            }
        else:
            # For classification and ordinal regression
            # Generate labels parameter to handle missing classes
            num_classes = getattr(config, 'num_classes', None)
            labels_param = list(range(num_classes)) if num_classes else None

            val_report = classification_report(
                val_output['labels'],
                val_output['preds'],
                labels=labels_param,
                target_names=getattr(config, 'class_names', None),
                output_dict=return_dicts,
            )

            test_report = classification_report(
                test_output['labels'],
                test_output['preds'],
                labels=labels_param,
                target_names=getattr(config, 'class_names', None),
                output_dict=return_dicts,
            )

    return val_report, test_report

if __name__ == "__main__":

    # -- command-line arguments
    parser = argparse.ArgumentParser(description='Training and/or evaluation of models.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', required=True, type=str, help='Configuration file to build, train, and evaluate the model')

    parser.add_argument('--training-dataset', required=False, type=str, help='CSV file representing the training dataset (not needed for inner CV)')
    parser.add_argument('--validation-dataset', required=False, type=str, help='CSV file representing the validation dataset (not needed for inner CV)')
    parser.add_argument('--test-dataset', required=True, type=str, help='CSV file representing the test dataset')

    parser.add_argument('--mode', default='both', type=str, help='Choose between: "training", "evaluation", or "both"')

    parser.add_argument("--exclude-features", nargs='+', default=['none'], help="Choose the features you don't want to use for ablation studies.")
    parser.add_argument('--filter-tasks', nargs='+', default=['ALL'], type=str, help='Choose the task to train')
    parser.add_argument('--filter-evaluation-tasks', nargs='+', default=['none'], type=str, help='Choose the task to evaluate on, otherwise we evaluate on the --filter-tasks')

    parser.add_argument("--yaml-overrides", metavar="CONF:[KEY]:VALUE", nargs='*', help="Set a number of conf-key-value pairs for modifying the yaml config file on the fly.")

    parser.add_argument('--load-checkpoint', default='', type=str, help='Choose between: "training", "evaluation", or "both"')
    parser.add_argument('--save-output', type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument('--save-attention-scores', type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument('--save-embeddings', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--output-dir', required=True, type=str, help='Path where to save model checkpoints and predictions')

    # New argument for inner CV
    parser.add_argument('--use-inner-cv', type=lambda x: bool(strtobool(x)), default=False, help='Use inner cross-validation for model selection')

    args = parser.parse_args()

    # Validate arguments based on mode
    if not args.use_inner_cv:
        if not args.training_dataset:
            parser.error("--training-dataset is required when not using inner CV (--use-inner-cv False)")
        if not args.validation_dataset:
            parser.error("--validation-dataset is required when not using inner CV (--use-inner-cv False)")

    # -- loading configuration file
    config_file = Path(args.config)
    with config_file.open('r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config = override_yaml(config, args.yaml_overrides)
    config = argparse.Namespace(**config)

    # -- pipeline process
    if args.use_inner_cv:
        val_report, test_report, inner_cv_summary = pipeline_with_inner_cv(args, config)

        print('\n--- INNER CV SUMMARY ---')
        print(f"Best inner fold: {inner_cv_summary['best_inner_fold']}")
        print(f"Best validation metric: {inner_cv_summary['best_inner_val_metric']}")
        print(f"Best epoch: {inner_cv_summary['best_inner_epoch']}")
        print("All inner fold results:")
        for fold, metric, epoch in inner_cv_summary['all_inner_results']:
            print(f"  Inner fold {fold}: metric={metric}, epoch={epoch}")

        print('\n--- TEST RESULTS ---')
        print(test_report)
    else:
        val_report, test_report = pipeline(args, config)

        print('\n--- VALIDATION ---')
        print(val_report)

        print('\n--- TEST ---')
        print(test_report)
