import comet_ml
import torch
# functional
import torch.nn.functional as F
import numpy as np

from yaml_config_manager import load_config
from tqdm import tqdm
import os
import json
from typing import Optional

from data_classes.ewadb_dataset import EWADBDataset
from model_classes.audio_classification_model import AudioClassificationModel
from miners.cl_miner import CLMiner

from additional_classes.checkpoint_manager import CheckpointManager
from list_dataloaders import ListDataLoaders

from utils import compute_metrics, get_dataset, create_optimizer_and_scheduler
from utils import get_device, create_model, get_experiment
from utils import get_classification_loss, resume_from_checkpoint_if_needed
from utils import save_results_file, save_confusion_matrix, get_single_dataloader
from utils import set_all_seeds_for_reproducibility, plot_embeddings, plot_embeddings_3d
from matplotlib import pyplot as plt

from pytorch_metric_learning import losses


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _select_threshold_max_f1(y_true: np.ndarray, y_score: np.ndarray, num_thresholds: int = 101) -> float:
    """Select a single threshold that maximizes positive-class F1 on (y_true, y_score)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if y_true.size == 0 or y_score.size == 0:
        return 0.5

    # If only one class present, threshold tuning is ill-defined.
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

    return best_t

def compute_classification_loss(config, criterion, batch, outputs):
    if config.model.num_classes == 2:
        logits = outputs["logits"].squeeze(1)
        targets = batch["labels"].float()
    else:
        logits = outputs["logits"]
        targets = batch["labels"]
    classification_loss = criterion(logits, targets)
    return classification_loss

def compute_contrastive_loss(config, criterion, batch, outputs, miner):
    embeddings = outputs["embeddings"]
    labels = batch["labels"]
    if miner is None:
        contrastive_loss = criterion(embeddings, labels)
    else:
        miner_output = miner.mine(embeddings, labels, batch["sample_type"], batch["domain_labels"])
        contrastive_loss = criterion(embeddings, labels, miner_output)

    # multiplier
    contrastive_loss *= config.training.contrastive_loss.multiplier
    return contrastive_loss

def train_one_epoch(config, model, dataloader, optimizer, scheduler, device, criterions, epoch, experiment, miner=None):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    all_scores = []

    p_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}", leave=False)
    for i, batch in p_bar:

        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        # if "domain_labels" in batch:
        #     unique_domains = batch["domain_labels"].unique()
        #     print(f"[DEBUG] Batch {i} domain_labels unique: {unique_domains.tolist()}, num_domains: {config.model.num_domains}")

        optimizer.zero_grad()
        outputs = model(batch)

        postfix_dict = {}
        loss = 0.0

        if config.training.cross_entropy_loss.active:
            classification_loss = compute_classification_loss(config, criterions["classification"], batch, outputs)
            postfix_dict = {"CLF-L": classification_loss.item()}
        else: classification_loss = None

        if config.training.contrastive_loss.active:
            contrastive_loss = compute_contrastive_loss(config, criterions["contrastive"], batch, outputs, miner)
            postfix_dict["CTR-L"] = contrastive_loss.item()
        else: contrastive_loss = None

        if classification_loss is not None: loss += classification_loss
        if contrastive_loss is not None: loss += contrastive_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        if config.model.num_classes == 2:
            logits = outputs["logits"].squeeze(1)

            # apply sigmoid to binary classification
            current_scores = torch.sigmoid(logits).detach().cpu().numpy()
            current_predictions = np.where(current_scores > 0.5, 1, 0)
            all_scores.extend(current_scores.tolist())
        else:
            # apply softmax to multiclass classification
            current_predictions = torch.softmax(logits, dim=-1).argmax(dim=-1).cpu().numpy()

        all_labels.extend(batch["labels"].cpu().numpy())
        all_predictions.extend(current_predictions)

        p_bar.set_postfix(postfix_dict)

    metrics = compute_metrics(
        all_labels,
        all_predictions,
        is_binary_classification=config.model.num_classes == 2,
        y_scores=all_scores if config.model.num_classes == 2 else None,
    )
    metrics["loss"] = running_loss / len(dataloader)
    return metrics

def evaluate_one_epoch(
    config,
    model,
    dataloader,
    device,
    criterions,
    epoch,
    experiment,
    return_embeddings: bool = False,
    threshold: Optional[float] = None,
    threshold_strategy: str = "fixed_0.5",
    threshold_grid: int = 101,
):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    all_scores = []
    all_embeddings = []
    all_sample_types = []

    with torch.no_grad():
        p_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Validation {epoch}", leave=False)
        for i, batch in p_bar:
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
                # collect sigmoid scores for binary classification
                current_scores = torch.sigmoid(logits).detach().cpu().numpy()
                all_scores.extend(current_scores.tolist())
            else:
                # apply softmax to multiclass classification
                current_predictions = torch.softmax(logits, dim=-1).argmax(dim=-1).cpu().numpy()
                all_predictions.extend(current_predictions)

            all_labels.extend(batch["labels"].cpu().numpy())
            all_embeddings.extend(outputs["embeddings"].cpu().numpy())
            all_sample_types.extend(batch["sample_type"].cpu().numpy())

            p_bar.set_postfix({"loss": running_loss / (i + 1)})

    used_threshold = None
    if config.model.num_classes == 2:
        y_true = np.asarray(all_labels).astype(int)
        y_score = np.asarray(all_scores).astype(float)

        def _pos_f1_from_pred(y_true_arr: np.ndarray, y_pred_arr: np.ndarray) -> float:
            tp = int(((y_pred_arr == 1) & (y_true_arr == 1)).sum())
            fp = int(((y_pred_arr == 1) & (y_true_arr == 0)).sum())
            fn = int(((y_pred_arr == 0) & (y_true_arr == 1)).sum())

            precision = _safe_div(tp, tp + fp)
            recall = _safe_div(tp, tp + fn)
            return _safe_div(2.0 * precision * recall, precision + recall)

        if threshold is not None:
            used_threshold = float(threshold)
        else:
            if str(threshold_strategy).lower() == "max_f1":
                used_threshold = _select_threshold_max_f1(y_true, y_score, num_thresholds=threshold_grid)
            else:
                used_threshold = 0.5

        y_pred_used = (y_score >= used_threshold).astype(int)
        all_predictions = y_pred_used.tolist()

    metrics = compute_metrics(
        all_labels,
        all_predictions,
        is_binary_classification=config.model.num_classes == 2,
        y_scores=all_scores if config.model.num_classes == 2 else None,
    )
    metrics["loss"] = running_loss / len(dataloader)
    if used_threshold is not None:
        metrics["threshold"] = used_threshold

    # Extra threshold-comparison logging data (only when tuning threshold with max_f1).
    if config.model.num_classes == 2 and used_threshold is not None:
        y_true_arr = np.asarray(all_labels).astype(int)
        y_score_arr = np.asarray(all_scores).astype(float)

        y_pred_used_arr = (y_score_arr >= float(used_threshold)).astype(int)
        metrics["pos_f1"] = _pos_f1_from_pred(y_true_arr, y_pred_used_arr)

        if threshold is None and str(threshold_strategy).lower() == "max_f1":
            y_pred_05_arr = (y_score_arr >= 0.5).astype(int)
            metrics_05 = compute_metrics(
                all_labels,
                y_pred_05_arr.tolist(),
                is_binary_classification=True,
                y_scores=all_scores,
            )
            metrics["f1_at_0.5"] = metrics_05.get("f1")
            metrics["pos_f1_at_0.5"] = _pos_f1_from_pred(y_true_arr, y_pred_05_arr)
    if return_embeddings:
        return metrics, all_embeddings, all_labels, all_sample_types
    return metrics



def main(config):

    # load train, validation, and test datasets
    # train_dataset = get_dataset(config, "train")
    # test_dataset = get_dataset(config, "test")
    # validation_dataset = get_dataset(config, "validation")

    # Use configurable seed so repeated runs can vary deterministically.
    set_all_seeds_for_reproducibility(getattr(config.training, "seed", 42))

    if config.ewadb.active:
        train_ewadb = get_dataset(config, "train", "ewadb", domain_id=0)
        validation_ewadb = get_dataset(config, "validation", "ewadb", domain_id=0)
    else:
        train_ewadb = None
        validation_ewadb = None

    if config.pc_gita.active:
        train_pcgita = get_dataset(config, "train", "pc_gita", domain_id=1)
        validation_pcgita = get_dataset(config, "validation", "pc_gita", domain_id=1)
    else:
        train_pcgita = None
        validation_pcgita = None

    if hasattr(config, 'Neurovoz_and_PC_GITA') and config.Neurovoz_and_PC_GITA.active:
        train_neurovoz_pcgita = get_dataset(config, "train", "Neurovoz_and_PC_GITA", domain_id=1)
        validation_neurovoz_pcgita = get_dataset(config, "validation", "Neurovoz_and_PC_GITA", domain_id=1)
    else:
        train_neurovoz_pcgita = None
        validation_neurovoz_pcgita = None

    if config.ewadb.active:
        test_ewadb = get_dataset(config, "test", "ewadb", domain_id=0)
    else:
        test_ewadb = None

    if config.pc_gita.active:
        test_pcgita = get_dataset(config, "test", "pc_gita", domain_id=1)
    else:
        test_pcgita = None

    if hasattr(config, 'Neurovoz_and_PC_GITA') and config.Neurovoz_and_PC_GITA.active:
        test_neurovoz_pcgita = get_dataset(config, "test", "Neurovoz_and_PC_GITA", domain_id=1)
    else:
        test_neurovoz_pcgita = None

    # # merge train datasets using torch.utils.data.ConcatDataset
    active_datasets = []
    if config.ewadb.active:
        active_datasets.append(train_ewadb)
    if config.pc_gita.active:
        active_datasets.append(train_pcgita)
    if hasattr(config, 'Neurovoz_and_PC_GITA') and config.Neurovoz_and_PC_GITA.active:
        active_datasets.append(train_neurovoz_pcgita)

    if len(active_datasets) > 1:
        train_dataset = torch.utils.data.ConcatDataset(active_datasets)
    elif len(active_datasets) == 1:
        train_dataset = active_datasets[0]
    else:
        neurovoz_active = hasattr(config, 'Neurovoz_and_PC_GITA') and config.Neurovoz_and_PC_GITA.active
        raise ValueError(f"At least one of the datasets should be active: pc_gita: {config.pc_gita.active}, ewadb: {config.ewadb.active}, Neurovoz_and_PC_GITA: {neurovoz_active}")

    # create dataloader
    train_dl = get_single_dataloader(config, train_dataset, "train", balance_dataloader=config.training.balance_dataloaders)

    # create dataloader
    # Create validation dataloaders
    val_dl_ewadb = None
    val_dl_pcgita = None
    val_dl_neurovoz_pcgita = None

    if config.ewadb.active:
        val_dl_ewadb = get_single_dataloader(config, validation_ewadb, "validation")
    if config.pc_gita.active:
        val_dl_pcgita = get_single_dataloader(config, validation_pcgita, "validation")
    if hasattr(config, 'Neurovoz_and_PC_GITA') and config.Neurovoz_and_PC_GITA.active:
        val_dl_neurovoz_pcgita = get_single_dataloader(config, validation_neurovoz_pcgita, "validation")

    # Create test dataloaders
    test_dl_ewadb = None
    test_dl_pcgita = None
    test_dl_neurovoz_pcgita = None

    if config.ewadb.active:
        test_dl_ewadb = get_single_dataloader(config, test_ewadb, "test")
    if config.pc_gita.active:
        test_dl_pcgita = get_single_dataloader(config, test_pcgita, "test")
    if hasattr(config, 'Neurovoz_and_PC_GITA') and config.Neurovoz_and_PC_GITA.active:
        test_dl_neurovoz_pcgita = get_single_dataloader(config, test_neurovoz_pcgita, "test")

    # train_dl = ListDataLoaders([train_dl_ewadb, train_dl_pcgita], weight_by_num_samples=True)

    print ("Datasets loaded successfully")
    if config.ewadb.active:  print ("Train EWADB dataset length: ", len(train_ewadb))
    if config.pc_gita.active: print ("Train PCGITA dataset length: ", len(train_pcgita))
    if hasattr(config, 'Neurovoz_and_PC_GITA') and config.Neurovoz_and_PC_GITA.active: print ("Train Neurovoz_and_PC_GITA dataset length: ", len(train_neurovoz_pcgita))

    if config.ewadb.active:  print ("Validation EWADB dataset length: ", len(validation_ewadb))
    if config.pc_gita.active: print ("Validation PCGITA dataset length: ", len(validation_pcgita))
    if hasattr(config, 'Neurovoz_and_PC_GITA') and config.Neurovoz_and_PC_GITA.active: print ("Validation Neurovoz_and_PC_GITA dataset length: ", len(validation_neurovoz_pcgita))

    if config.ewadb.active: print ("Test EWADB dataset length: ", len(test_ewadb))
    if config.pc_gita.active: print ("Test PCGITA dataset length: ", len(test_pcgita))
    if hasattr(config, 'Neurovoz_and_PC_GITA') and config.Neurovoz_and_PC_GITA.active: print ("Test Neurovoz_and_PC_GITA dataset length: ", len(test_neurovoz_pcgita))

    # set number of domains dynamically based on active datasets
    # num_domains = 0
    # if config.ewadb.active:
    #     num_domains += 1
    # if config.pc_gita.active:
    #     num_domains += 1
    # if hasattr(config, 'Neurovoz_and_PC_GITA') and config.Neurovoz_and_PC_GITA.active:
    #     num_domains += 1
    config.model.num_domains = 2

    experiment = get_experiment(config)
    device = get_device(config)
    print(f"\t\tUsing device: {device}")
    model = create_model(config, device)
    len_for_opt_and_sched = 0
    if config.ewadb.active: len_for_opt_and_sched += len(train_ewadb)
    if config.pc_gita.active: len_for_opt_and_sched += len(train_pcgita)
    if hasattr(config, 'Neurovoz_and_PC_GITA') and config.Neurovoz_and_PC_GITA.active: len_for_opt_and_sched += len(train_neurovoz_pcgita)
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, len_for_opt_and_sched)

    checkpoint_manager = CheckpointManager(
        checkpoint_dir = config.training.checkpoint_dir,
        model = model,
        optimizer = optimizer,
        scheduler = scheduler,
        device = device,
        lower_is_better = config.training.validation.metric_lower_is_better
    )

    criterions = {}
    criterions["classification"] = get_classification_loss(config.model.num_classes)
    # contrastive learning loss
    if config.training.contrastive_loss.active:
        criterions["contrastive"] = losses.ContrastiveLoss()
        miner = CLMiner()
    else:
        miner = None



    # resume if needed
    start_epoch, best_metric = resume_from_checkpoint_if_needed(config, checkpoint_manager)

    # train loop
    max_epochs_without_improvement = 5
    current_epochs_without_improvement = 0
    for epoch in range(start_epoch, config.training.num_epochs):
        print(f"Epoch {epoch} started")

        train_metrics = train_one_epoch(config, model, train_dl, optimizer, scheduler, device, criterions, epoch, experiment, miner)

        if val_dl_ewadb is not None: val_metrics_ewadb = evaluate_one_epoch(config, model, val_dl_ewadb, device, criterions, epoch, experiment)
        else: val_metrics_ewadb = None

        if val_dl_pcgita is not None: val_metrics_pcgita = evaluate_one_epoch(config, model, val_dl_pcgita, device, criterions, epoch, experiment)
        else: val_metrics_pcgita = None

        if val_dl_neurovoz_pcgita is not None:
            val_metrics_neurovoz = evaluate_one_epoch(config, model, val_dl_neurovoz_pcgita, device, criterions, epoch, experiment)
        else:
            val_metrics_neurovoz = None

        print(f"Epoch {epoch} Train Loss: {train_metrics['loss']}")
        print(f"Epoch {epoch} Train Metrics: {train_metrics}")

        if val_metrics_ewadb is not None:
            print(f"[EWADB] Epoch {epoch} metrics:")
            for m in val_metrics_ewadb: print(f"Val {m}: {val_metrics_ewadb[m]}")
            for m in val_metrics_ewadb: experiment.log(f"val_ewadb_{m}", val_metrics_ewadb[m])

        if val_metrics_pcgita is not None:
            print(f"[PCGITA] Epoch {epoch} metrics:")
            for m in val_metrics_pcgita: print(f"Val {m}: {val_metrics_pcgita[m]}")
            for m in val_metrics_pcgita: experiment.log(f"val_pcgita_{m}", val_metrics_pcgita[m])

        if val_metrics_neurovoz is not None:
            print(f"[Neurovoz_and_PC_GITA] Epoch {epoch} metrics:")
            for m in val_metrics_neurovoz: print(f"Val {m}: {val_metrics_neurovoz[m]}")
            for m in val_metrics_neurovoz: experiment.log(f"val_neurovoz_{m}", val_metrics_neurovoz[m])

        # add epoch_ as prefix to all metrics
        epoch_metrics = { f"epoch_{k}": v for k, v in train_metrics.items() }

        if "+" in config.training.validation.metric:
            m_to_be_used = config.training.validation.metric.split("+")
        else:
            m_to_be_used = [config.training.validation.metric]

        current_metric = []
        if val_metrics_ewadb is not None:
            for m in m_to_be_used: current_metric.append(val_metrics_ewadb[m])
        if val_metrics_pcgita is not None:
            for m in m_to_be_used: current_metric.append(val_metrics_pcgita[m])
        if val_metrics_neurovoz is not None:
            for m in m_to_be_used: current_metric.append(val_metrics_neurovoz[m])

        if len(current_metric) == 0:
            # 没有任何验证集，退化为使用训练指标
            for m in m_to_be_used:
                current_metric.append(train_metrics[m])

        current_metric = sum(current_metric) / len(current_metric)

        # current metric is the average of the metrics from both datasets
        print(f"Current Metric: {current_metric}")
        print(f"Previous best metric: {checkpoint_manager.get_current_best_metric()}")
        current_is_best = checkpoint_manager.save_checkpoint(epoch, current_metric)
        if current_is_best:
            current_epochs_without_improvement = 0
        else:
            current_epochs_without_improvement += 1
            print(f"Current epochs without improvement: {current_epochs_without_improvement}")
            if current_epochs_without_improvement >= max_epochs_without_improvement:
                print(f"Early stopping at epoch {epoch} - no improvement for {max_epochs_without_improvement} epochs")
                break

    # load best model
    checkpoint_manager.load_best_model()

    # Tune a single threshold on validation (best model), then apply to test.
    tuned_thresholds = {}
    threshold_strategy = getattr(getattr(config.training, "validation", object()), "threshold_strategy", "max_f1")
    threshold_grid = int(getattr(getattr(config.training, "validation", object()), "threshold_grid", 101))

    if config.model.num_classes == 2:
        if val_dl_ewadb is not None:
            m = evaluate_one_epoch(
                config,
                model,
                val_dl_ewadb,
                device,
                criterions,
                "val(best)",
                experiment,
                return_embeddings=False,
                threshold=None,
                threshold_strategy=threshold_strategy,
                threshold_grid=threshold_grid,
            )
            tuned_thresholds["ewadb"] = m.get("threshold", 0.5)
            if str(threshold_strategy).lower() == "max_f1":
                print(
                    f"[THRESHOLD][ewadb] best_thr={m.get('threshold', 0.5):.4f}, "
                    f"pos_f1@best={m.get('pos_f1')}, pos_f1@0.5={m.get('pos_f1_at_0.5')}, "
                    f"f1@best={m.get('f1')}, f1@0.5={m.get('f1_at_0.5')}"
                )
        if val_dl_pcgita is not None:
            m = evaluate_one_epoch(
                config,
                model,
                val_dl_pcgita,
                device,
                criterions,
                "val(best)",
                experiment,
                return_embeddings=False,
                threshold=None,
                threshold_strategy=threshold_strategy,
                threshold_grid=threshold_grid,
            )
            tuned_thresholds["pc_gita"] = m.get("threshold", 0.5)
            if str(threshold_strategy).lower() == "max_f1":
                print(
                    f"[THRESHOLD][pc_gita] best_thr={m.get('threshold', 0.5):.4f}, "
                    f"pos_f1@best={m.get('pos_f1')}, pos_f1@0.5={m.get('pos_f1_at_0.5')}, "
                    f"f1@best={m.get('f1')}, f1@0.5={m.get('f1_at_0.5')}"
                )
        if val_dl_neurovoz_pcgita is not None:
            m = evaluate_one_epoch(
                config,
                model,
                val_dl_neurovoz_pcgita,
                device,
                criterions,
                "val(best)",
                experiment,
                return_embeddings=False,
                threshold=None,
                threshold_strategy=threshold_strategy,
                threshold_grid=threshold_grid,
            )
            tuned_thresholds["Neurovoz_and_PC_GITA"] = m.get("threshold", 0.5)
            if str(threshold_strategy).lower() == "max_f1":
                print(
                    f"[THRESHOLD][Neurovoz_and_PC_GITA] best_thr={m.get('threshold', 0.5):.4f}, "
                    f"pos_f1@best={m.get('pos_f1')}, pos_f1@0.5={m.get('pos_f1_at_0.5')}, "
                    f"f1@best={m.get('f1')}, f1@0.5={m.get('f1_at_0.5')}"
                )

    if tuned_thresholds:
        print(f"[THRESHOLD] Tuned on val (strategy={threshold_strategy}, grid={threshold_grid}): {tuned_thresholds}")

        # Persist tuned thresholds so standalone test.py can reuse them.
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
        thresholds_path = os.path.join(config.training.checkpoint_dir, "tuned_thresholds.json")
        with open(thresholds_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "thresholds": tuned_thresholds,
                    "strategy": threshold_strategy,
                    "grid": threshold_grid,
                },
                f,
                indent=2,
                sort_keys=True,
            )
        print(f"[THRESHOLD] Saved tuned thresholds to: {thresholds_path}")

    # separate evaluation for test datasets
    if config.ewadb.active:
        test_threshold = tuned_thresholds.get("ewadb", 0.5)
        test_metrics_ewadb, embeddings_ewadb, labels_ewadb, sample_types_ewadb = evaluate_one_epoch(
            config,
            model,
            test_dl_ewadb,
            device,
            criterions,
            "test",
            experiment,
            return_embeddings=True,
            threshold=test_threshold,
            threshold_strategy="fixed",
        )
        print(f"[EWADB] Test Metrics")
        for m in test_metrics_ewadb: print(f"Test {m}: {test_metrics_ewadb[m]}")

        # store results for EWADB test dataset
        save_results_file(config.training.checkpoint_dir, test_metrics_ewadb, prefix="ewadb_")
        save_confusion_matrix(config.training.checkpoint_dir, test_metrics_ewadb["confusion_matrix"], prefix="ewadb_")
        plot_embeddings(config.training.checkpoint_dir, embeddings_ewadb, labels_ewadb, sample_types_ewadb, prefix="ewadb_")
        plot_embeddings_3d(config.training.checkpoint_dir, embeddings_ewadb, labels_ewadb, sample_types_ewadb, prefix="ewadb_")

    if config.pc_gita.active:
        test_threshold = tuned_thresholds.get("pc_gita", 0.5)
        test_metrics_pcgita, embedding_pcgita, labels_pcgita, sample_types_pcgita = evaluate_one_epoch(
            config,
            model,
            test_dl_pcgita,
            device,
            criterions,
            "test",
            experiment,
            return_embeddings=True,
            threshold=test_threshold,
            threshold_strategy="fixed",
        )
        print(f"[PCGITA] Test Metrics")
        for m in test_metrics_pcgita: print(f"Test {m}: {test_metrics_pcgita[m]}")

        # store results for PCGITA test dataset
        save_results_file(config.training.checkpoint_dir, test_metrics_pcgita, prefix="pcgita_")
        save_confusion_matrix(config.training.checkpoint_dir, test_metrics_pcgita["confusion_matrix"], prefix="pcgita_")
        plot_embeddings(config.training.checkpoint_dir, embedding_pcgita, labels_pcgita, sample_types_pcgita, prefix="pcgita_")
        plot_embeddings_3d(config.training.checkpoint_dir, embedding_pcgita, labels_pcgita, sample_types_pcgita, prefix="pcgita_")

    if hasattr(config, 'Neurovoz_and_PC_GITA') and config.Neurovoz_and_PC_GITA.active:
        test_threshold = tuned_thresholds.get("Neurovoz_and_PC_GITA", 0.5)
        test_metrics_neurovoz, embeddings_neurovoz, labels_neurovoz, sample_types_neurovoz = evaluate_one_epoch(
            config,
            model,
            test_dl_neurovoz_pcgita,
            device,
            criterions,
            "test",
            experiment,
            return_embeddings=True,
            threshold=test_threshold,
            threshold_strategy="fixed",
        )
        print(f"[Neurovoz_and_PC_GITA] Test Metrics")
        for m in test_metrics_neurovoz:
            print(f"Test {m}: {test_metrics_neurovoz[m]}")

        save_results_file(config.training.checkpoint_dir, test_metrics_neurovoz, prefix="neurovoz_pcgita_")
        save_confusion_matrix(config.training.checkpoint_dir, test_metrics_neurovoz["confusion_matrix"], prefix="neurovoz_pcgita_")
        # plot_embeddings(config.training.checkpoint_dir, embeddings_neurovoz, labels_neurovoz, sample_types_neurovoz, prefix="neurovoz_pcgita_")
        # plot_embeddings_3d(config.training.checkpoint_dir, embeddings_neurovoz, labels_neurovoz, sample_types_neurovoz, prefix="neurovoz_pcgita_")


if __name__ == "__main__":
    config = load_config()
    main(config)
