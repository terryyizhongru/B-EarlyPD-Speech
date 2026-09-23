import torch
import numpy as np
from yaml_config_manager import load_config
from tqdm import tqdm
import os
import csv
import json

from model_classes.audio_classification_model import AudioClassificationModel
from additional_classes.checkpoint_manager import CheckpointManager

from utils import get_dataset, get_device, create_model, get_single_dataloader
from utils import get_classification_loss, compute_metrics, save_results_file, save_confusion_matrix


def save_detailed_results_tsv(config, dataset_name, test_metadata_path, audio_paths, probabilities, true_labels, predictions):
    """Save positive-class probabilities and labels aligned to the test TSV."""
    if test_metadata_path is None or not os.path.exists(test_metadata_path):
        raise FileNotFoundError(f"Test metadata TSV for {dataset_name} not found: {test_metadata_path}")

    lengths = (len(audio_paths), len(probabilities), len(true_labels), len(predictions))
    if len(set(lengths)) != 1:
        raise ValueError(f"Prediction array lengths differ for {dataset_name}: {lengths}")

    # A path may occur more than once; consume its predictions in input order.
    path_to_entries = {}
    for path, probability_row, y_true, y_pred in zip(audio_paths, probabilities, true_labels, predictions):
        path_to_entries.setdefault(path, []).append((probability_row, int(y_true), int(y_pred)))

    with open(test_metadata_path, "r", encoding="utf-8") as fin:
        reader = csv.reader(fin, delimiter="\t")
        header = next(reader, None)
        if not header:
            raise ValueError(f"Test metadata TSV for {dataset_name} is empty: {test_metadata_path}")

        num_classes = config.model.num_classes
        probability_cols = ["probability"] if num_classes == 2 else [
            f"probability_{i}" for i in range(num_classes)
        ]
        new_header = header + probability_cols + ["true_label", "predicted_label"]
        rows = []
        for row in reader:
            if not row:
                continue
            if len(row) < 2:
                raise ValueError(f"Test metadata row has no AUDIOFILE: {row}")
            audio_path = row[1]
            entries = path_to_entries.get(audio_path)
            if not entries:
                raise ValueError(f"No prediction for {dataset_name} AUDIOFILE: {audio_path}")
            probability_row, y_true, y_pred = entries.pop(0)
            if num_classes == 2:
                probability_vals = [float(probability_row if np.ndim(probability_row) == 0 else probability_row[0])]
            else:
                probability_vals = [float(x) for x in probability_row]
            rows.append(row + [str(v) for v in probability_vals] + [str(y_true), str(y_pred)])

    unused = sum(len(entries) for entries in path_to_entries.values())
    if unused:
        raise ValueError(f"{unused} predictions for {dataset_name} have no matching test metadata row")

    output_tsv = os.path.join(config.training.checkpoint_dir, f"{dataset_name}_detailed_results.tsv")
    with open(output_tsv, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout, delimiter="\t")
        writer.writerow(new_header)
        writer.writerows(rows)
    print(f"[INFO] Saved detailed results TSV for {dataset_name} to {output_tsv}")


def load_tuned_thresholds(checkpoint_dir: str) -> dict:
    """Load thresholds saved by train.py (if present)."""
    path = os.path.join(checkpoint_dir, "tuned_thresholds.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("thresholds", {}) if isinstance(payload, dict) else {}
    except Exception as e:
        print(f"[WARN] Failed to load tuned thresholds from {path}: {e}")
        return {}


def evaluate_test_set(
    config,
    model,
    dataloader,
    device,
    criterions,
    return_embeddings=False,
    return_logits_and_paths=False,
    threshold: float = 0.5,
):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    all_scores = []
    all_embeddings = []
    all_sample_types = []
    all_logits = []
    all_paths = []

    with torch.no_grad():
        p_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing", leave=False)
        for i, batch in p_bar:
            # Keep a copy of non-tensor fields (e.g., paths) before moving tensors to device
            raw_paths = None
            if return_logits_and_paths and "audio_path" in batch:
                raw_paths = batch["audio_path"]

            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = model(batch)
            if config.model.num_classes == 2:
                # outputs = outputs.squeeze(1)
                logits = outputs["logits"].squeeze(1)
                targets = batch["labels"].float()
            else:
                logits = outputs["logits"]
                targets = batch["labels"]
            loss = criterions["classification"](logits, targets)
            running_loss += loss.item()

            if config.model.num_classes == 2:
                prob = torch.sigmoid(logits)
                current_scores = prob.detach().cpu().numpy()
                all_scores.extend(current_scores.tolist())
                current_predictions = np.where(current_scores >= float(threshold), 1, 0)
            else:
                prob = torch.softmax(logits, dim=-1)
                current_predictions = prob.argmax(dim=-1).cpu().numpy()

            all_labels.extend(batch["labels"].cpu().numpy())
            all_predictions.extend(current_predictions)

            if return_logits_and_paths:
                all_logits.append(prob.detach().cpu().numpy())
                if raw_paths is not None:
                    # assume list[str] or list
                    all_paths.extend(list(raw_paths))

            if return_embeddings:
                all_embeddings.append(outputs["embeddings"].cpu().numpy())
                if "sample_type" in batch:
                    all_sample_types.extend(batch["sample_type"].cpu().numpy().tolist())
                else:
                    # Default to speech (0) if sample_type is not provided
                    all_sample_types.extend([0] * len(batch["labels"]))
                    print("Warning: 'sample_type' not found in batch. Defaulting to speech type (0).")

    metrics = compute_metrics(
        all_labels,
        all_predictions,
        is_binary_classification=config.model.num_classes == 2,
        y_scores=all_scores if config.model.num_classes == 2 else None,
    )
    metrics["loss"] = running_loss / len(dataloader)
    if config.model.num_classes == 2:
        metrics["threshold"] = float(threshold)

    if return_embeddings:
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        result = (metrics, all_embeddings, all_labels, all_sample_types)
    else:
        result = (metrics,)

    if return_logits_and_paths:
        if all_logits:
            all_logits = np.concatenate(all_logits, axis=0)
        result = result + (all_logits, all_paths, all_predictions)

    if len(result) == 1:
        return result[0]
    return result

def main(config):
    # Load the test datasets based on active configuration
    test_ewadb = None
    test_pcgita = None

    if config.ewadb.active:
        test_ewadb = get_dataset(config, "test", "ewadb", domain_id=0)
        print("Test EWADB dataset length: ", len(test_ewadb))
    else:
        print("EWADB dataset is not active, skipping EWADB testing")

    if config.pc_gita.active:
        test_pcgita = get_dataset(config, "test", "pc_gita", domain_id=1)
        print("Test PCGITA dataset length: ", len(test_pcgita))
    else:
        print("PC-GITA dataset is not active, skipping PC-GITA testing")

    if hasattr(config, 'Neurovoz_and_PC_GITA') and config.Neurovoz_and_PC_GITA.active:
        test_neurovoz_pcgita = get_dataset(config, "test", "Neurovoz_and_PC_GITA", domain_id=1)
        print("Test Neurovoz_and_PC_GITA dataset length: ", len(test_neurovoz_pcgita))
    else:
        print("Neurovoz_and_PC_GITA dataset is not active, skipping Neurovoz_and_PC_GITA testing")

    if not config.ewadb.active and not config.pc_gita.active and not (hasattr(config, 'Neurovoz_and_PC_GITA') and config.Neurovoz_and_PC_GITA.active):
        raise ValueError("At least one dataset should be active for testing")

    print("Test datasets loaded successfully")

    # Initialize device, model, and dataloader
    device = get_device(config)
    print(f"Using device: {device}")

    # set number of domains
    config.model.num_domains = 2

    model = create_model(config, device)

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config.training.checkpoint_dir,
        model=model,
        optimizer=None,
        scheduler=None,
        device=device,
        lower_is_better=config.training.validation.metric_lower_is_better
    )

    # Load the best model from checkpoint
    checkpoint_manager.load_best_model()
    print("Loaded best model from checkpoint")

    # Create separate test dataloaders for each active dataset
    test_dl_ewadb = None
    test_dl_pcgita = None
    test_dl_neurovoz_pcgita = None

    if config.ewadb.active and test_ewadb is not None:
        test_dl_ewadb = get_single_dataloader(config, test_ewadb, "test")

    if config.pc_gita.active and test_pcgita is not None:
        test_dl_pcgita = get_single_dataloader(config, test_pcgita, "test")

    if hasattr(config, 'Neurovoz_and_PC_GITA') and config.Neurovoz_and_PC_GITA.active and test_neurovoz_pcgita is not None:
        test_dl_neurovoz_pcgita = get_single_dataloader(config, test_neurovoz_pcgita, "test")

    criterions = {}
    criterions["classification"] = get_classification_loss(config.model.num_classes)
    # criterions["domain_classification"] = get_classification_loss(config.model.num_domains)


    tuned_thresholds = load_tuned_thresholds(config.training.checkpoint_dir)
    if test_dl_neurovoz_pcgita is not None and "Neurovoz_and_PC_GITA" not in tuned_thresholds:
        raise FileNotFoundError(
            "Missing Neurovoz_and_PC_GITA validation threshold in "
            f"{config.training.checkpoint_dir}/tuned_thresholds.json"
        )
    if tuned_thresholds:
        print(f"[THRESHOLD] Loaded tuned thresholds: {tuned_thresholds}")
    else:
        print("[THRESHOLD] No tuned_thresholds.json found; using default threshold=0.5")

    # Evaluate the model on the test sets with embeddings and save detailed TSVs
    if test_dl_ewadb is not None:
        thr = float(tuned_thresholds.get("ewadb", 0.5))
        test_metrics_ewadb, embeddings_ewadb, labels_ewadb, sample_types_ewadb, logits_ewadb, paths_ewadb, preds_ewadb = evaluate_test_set(
            config,
            model,
            test_dl_ewadb,
            device,
            criterions,
            return_embeddings=True,
            return_logits_and_paths=True,
            threshold=thr,
        )
        print(f"[EWADB] Test Metrics:")
        for m in test_metrics_ewadb:
            print(f"Test {m}: {test_metrics_ewadb[m]}")

        # Save the results and confusion matrices for EWADB
        save_results_file(config.training.checkpoint_dir, test_metrics_ewadb, prefix="ewadb_")
        save_confusion_matrix(config.training.checkpoint_dir, test_metrics_ewadb["confusion_matrix"], prefix="ewadb_")

        # Save detailed per-sample results TSV if metadata path is available in config
        tsv_path_ewadb = getattr(config.ewadb, "test_metadata_path", None) if hasattr(config, "ewadb") else None
        save_detailed_results_tsv(config, "ewadb", tsv_path_ewadb, paths_ewadb, logits_ewadb, labels_ewadb, preds_ewadb)

    if test_dl_pcgita is not None:
        thr = float(tuned_thresholds.get("pc_gita", 0.5))
        test_metrics_pcgita, embeddings_pcgita, labels_pcgita, sample_types_pcgita, logits_pcgita, paths_pcgita, preds_pcgita = evaluate_test_set(
            config,
            model,
            test_dl_pcgita,
            device,
            criterions,
            return_embeddings=True,
            return_logits_and_paths=True,
            threshold=thr,
        )
        print(f"[PCGITA] Test Metrics:")
        for m in test_metrics_pcgita:
            print(f"Test {m}: {test_metrics_pcgita[m]}")

        # Save the results and confusion matrices for PC-GITA
        save_results_file(config.training.checkpoint_dir, test_metrics_pcgita, prefix="pcgita_")
        save_confusion_matrix(config.training.checkpoint_dir, test_metrics_pcgita["confusion_matrix"], prefix="pcgita_")

        tsv_path_pcgita = getattr(config.pc_gita, "test_metadata_path", None) if hasattr(config, "pc_gita") else None
        save_detailed_results_tsv(config, "pcgita", tsv_path_pcgita, paths_pcgita, logits_pcgita, labels_pcgita, preds_pcgita)

    if test_dl_neurovoz_pcgita is not None:
        thr = float(tuned_thresholds.get("Neurovoz_and_PC_GITA", 0.5))
        test_metrics_neurovoz_pcgita, embeddings_neurovoz_pcgita, labels_neurovoz_pcgita, sample_types_neurovoz_pcgita, logits_neurovoz_pcgita, paths_neurovoz_pcgita, preds_neurovoz_pcgita = evaluate_test_set(
            config,
            model,
            test_dl_neurovoz_pcgita,
            device,
            criterions,
            return_embeddings=True,
            return_logits_and_paths=True,
            threshold=thr,
        )
        print(f"[Neurovoz_and_PC_GITA] Test Metrics:")
        for m in test_metrics_neurovoz_pcgita:
            print(f"Test {m}: {test_metrics_neurovoz_pcgita[m]}")

        # Save the results and confusion matrices for Neurovoz_and_PC_GITA
        save_results_file(config.training.checkpoint_dir, test_metrics_neurovoz_pcgita, prefix="neurovoz_pcgita_")
        save_confusion_matrix(config.training.checkpoint_dir, test_metrics_neurovoz_pcgita["confusion_matrix"], prefix="neurovoz_pcgita_")

        tsv_path_neurovoz_pcgita = getattr(config.Neurovoz_and_PC_GITA, "test_metadata_path", None) if hasattr(config, "Neurovoz_and_PC_GITA") else None
        save_detailed_results_tsv(config, "Neurovoz_and_PC_GITA", tsv_path_neurovoz_pcgita, paths_neurovoz_pcgita, logits_neurovoz_pcgita, labels_neurovoz_pcgita, preds_neurovoz_pcgita)


if __name__ == "__main__":
    config = load_config()
    main(config)
