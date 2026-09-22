"""Runtime helpers required by the RECA-PD binary-classification pipeline."""

import copy
import os
import pickle
import sys
from os import path

import numpy as np
import torch

sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

from datasets import ParkinsonDataset
from models import RECAPD_Model


def build_model(config):
    if config.model != "RECAPD":
        raise ValueError(f"This integration supports only RECAPD, got: {config.model}")
    model = RECAPD_Model(config)
    return model.to(dtype=getattr(torch, config.dtype), device=config.device)


def task_and_feature_filtering(old_config, filter_tasks, exclude_features, verbose=True):
    config = copy.copy(old_config)

    if "ALL" not in filter_tasks:
        filtered_tasks = []
        consequent_features = []
        for task in config.tasks:
            if task["name"] in filter_tasks:
                filtered_tasks.append(task)
                consequent_features += task["features"]
        config.tasks = filtered_tasks
        consequent_features = list(set(consequent_features))
    else:
        consequent_features = []
        for task in config.tasks:
            consequent_features += task["features"]

    config.features = [
        feature
        for feature in config.features
        if feature["name"] in consequent_features
        and feature["name"] not in exclude_features
    ]

    assert config.features, "No informed features remain after filtering"
    assert config.tasks, "No tasks remain after filtering"

    if verbose:
        print(
            "Using the following features: "
            + " | ".join(feature["name"].upper() for feature in config.features)
        )
        print(
            "Using the following tasks: "
            + " | ".join(task["name"].upper() for task in config.tasks)
        )

    return config


def get_dataloader(config, dataset_path, is_training=True):
    dataset = ParkinsonDataset(config, dataset_path, is_training=is_training)
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=is_training,
        batch_size=config.training_settings["batch_size"],
        num_workers=config.training_settings["num_workers"],
        pin_memory=True,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )


def get_optimizer_and_scheduler(config, model, train_loader):
    if config.training_settings["optimizer"] != "adamw":
        raise ValueError(
            f"Unknown optimizer: {config.training_settings['optimizer']}"
        )

    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        config.training_settings["learning_rate"] / 10,
    )

    if config.training_settings["scheduler"] != "onecycle":
        raise ValueError(
            f"Unknown scheduler: {config.training_settings['scheduler']}"
        )

    accum_grad = max(int(config.training_settings.get("accum_grad", 1)), 1)
    steps_per_epoch = int(np.ceil(len(train_loader) / accum_grad))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.training_settings["learning_rate"],
        epochs=config.training_settings["epochs"],
        steps_per_epoch=steps_per_epoch,
        anneal_strategy=config.training_settings["anneal_strategy"],
    )
    return optimizer, scheduler


def save_checkpoint(model, output_dir, suffix):
    checkpoint_dir = os.path.join(output_dir, "model_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{suffix}.pth")
    print(f"Saving model checkpoint in {checkpoint_path}...")
    torch.save(model.state_dict(), checkpoint_path)


def save_embedding(embedding, output_dir, sample_id):
    embeddings_dir = os.path.join(output_dir, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(embeddings_dir, f"{sample_id}.npz"),
        data=embedding.reshape(1, -1),
    )


def save_model_output(output_stats, output_dir, suffix, save_attention_scores=True):
    model_outputs_dir = os.path.join(output_dir, "model_output")
    os.makedirs(model_outputs_dir, exist_ok=True)
    print(f"Saving model output in {model_outputs_dir}...")

    mha_scores_output = {}
    for key in output_stats.copy():
        if "mha_scores" in key or "metadata" in key:
            mha_scores_output[key] = output_stats.pop(key, None)
    mha_scores_output["labels"] = output_stats["labels"].copy()
    mha_scores_output["sample_id"] = output_stats["sample_id"].copy()

    with open(
        os.path.join(model_outputs_dir, f"{suffix}_classification.pkl"), "wb"
    ) as handle:
        pickle.dump(output_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if save_attention_scores:
        with open(
            os.path.join(model_outputs_dir, f"{suffix}_mha_scores.pkl"), "wb"
        ) as handle:
            pickle.dump(mha_scores_output, handle, protocol=pickle.HIGHEST_PROTOCOL)


def override_yaml(yaml_config, to_override):
    if to_override is None:
        return yaml_config

    for new_setting in to_override:
        if new_setting.count(":") == 1:
            key, value = new_setting.split(":")
            value_type = type(yaml_config[key])
            yaml_config[key] = value == "true" if value_type is bool else value_type(value)
        elif new_setting.count(":") == 2:
            section, key, value = new_setting.split(":")
            value_type = type(yaml_config[section][key])
            yaml_config[section][key] = (
                value == "true" if value_type is bool else value_type(value)
            )
        else:
            raise ValueError(f"Invalid YAML override: {new_setting}")

    return yaml_config
