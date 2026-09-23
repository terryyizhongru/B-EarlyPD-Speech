#!/usr/bin/env python
# coding: utf-8

# Inception Model for Parkinson's Disease detection from voice spectral data.
# The Inception V3 model, pretrained on Imagenet is adapted using transfer learning to
# extract features from spectrogram images of the sustained vowel /a/ to distinguish people
# with Parkinson’s Disease (PwPD) from healthy controls (HC).
#
# Audio files were preprocessed using the R packages Create_Liner(Mel)Spectrograms_(dataset) which are available in https://github.com/uams-tri/PD-Voice.
#
# Spectra files must be oranized into the directory structure required by the Keras ImageDatatGenerator() class when using the
# class_mode- 'categorical' option as described in
# https://vijayabhaskar96.medium.com/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720.
# The pathname to this directory strucutre must be changed for each data set being analyzed. Output file names should be modified per experiment.
# This approach was used to improve understandability.
# For comparison with other analyses performeed on these data, training for transfer learing is re-initialized on each of 100 iterations retaining ROC data.


# Copyright (C) 2024 University of Arkansas for Medical Sciences
# Author: Anu Iyer, Fred Prior, PhD FWPrior@uams.edu
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


import pandas as pd
import numpy as np
import tensorflow as tf
import timeit
import datetime
import math
import re
import os
import json
import argparse
import random
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _select_threshold_max_f1(y_true: np.ndarray, y_score: np.ndarray, num_thresholds: int = 101) -> float:
    """Select a single threshold that maximizes (binary) F1 on (y_true, y_score)."""
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


def wav_to_jpg_path(wav_path: str) -> str:
    # User rule: wavpath.replace('fortrain', 'fortrain_jpg').replace('.wav', '.jpg')
    jpg_path = wav_path.replace('fortrain', 'fortrain_jpg')
    jpg_path = re.sub(r"\.wav$", ".jpg", jpg_path, flags=re.IGNORECASE)
    return jpg_path


def normalize_label(label: str) -> str:
    s = str(label).strip().lower()
    if s in {"healthy", "hc", "control", "controls"}:
        return "healthy"
    if s in {"parkinson", "pd", "parkinsons", "parkinson's"}:
        return "parkinson"
    return s


def load_tsv(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep='\t')
    df = df.copy()
    df["filename"] = df["AUDIOFILE"].astype(str).apply(wav_to_jpg_path)
    df["label"] = df["DIAGNOSIS"].astype(str).apply(normalize_label)
    return df


def stable_binary_classes_from_df(df: pd.DataFrame) -> list[str]:
    labels = set(df["label"].astype(str).tolist())
    if {"healthy", "parkinson"}.issubset(labels):
        return ["healthy", "parkinson"]
    return sorted(labels)


def build_data_generators(data_path, img_rows, img_cols, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        fill_mode='nearest',
        validation_split=0.3,
    )
    train_generator = train_datagen.flow_from_directory(
        data_path,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
    )
    validation_generator = train_datagen.flow_from_directory(
        data_path,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
    )
    return train_generator, validation_generator


def build_data_generators_from_tsv(
    train_tsv_path,
    img_rows,
    img_cols,
    batch_size,
    *,
    val_tsv_path=None,
    validation_split=0.3,
    seed=123,
):
    if val_tsv_path is None:
        df = load_tsv(train_tsv_path)
        classes = stable_binary_classes_from_df(df)
        datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            fill_mode='nearest',
            validation_split=validation_split,
        )
        train_generator = datagen.flow_from_dataframe(
            df,
            x_col="filename",
            y_col="label",
            classes=classes,
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True,
            subset='training',
            seed=seed,
        )
        validation_generator = datagen.flow_from_dataframe(
            df,
            x_col="filename",
            y_col="label",
            classes=classes,
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False,
            subset='validation',
            seed=seed,
        )
        return train_generator, validation_generator

    df_train = load_tsv(train_tsv_path)
    df_val = load_tsv(val_tsv_path)
    classes = stable_binary_classes_from_df(df_train)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        fill_mode='nearest',
    )
    train_generator = train_datagen.flow_from_dataframe(
        df_train,
        x_col="filename",
        y_col="label",
        classes=classes,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True,
        seed=seed,
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        fill_mode='nearest',
    )
    validation_generator = val_datagen.flow_from_dataframe(
        df_val,
        x_col="filename",
        y_col="label",
        classes=classes,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False,
        seed=seed,
    )
    return train_generator, validation_generator


def build_eval_generator_from_tsv(tsv_path, img_rows, img_cols, batch_size, classes):
    df = load_tsv(tsv_path)
    datagen = ImageDataGenerator(rescale=1.0 / 255, fill_mode='nearest')
    return datagen.flow_from_dataframe(
        df,
        x_col="filename",
        y_col="label",
        classes=classes,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False,
    )


def build_model(img_rows: int, img_cols: int):
    pre_trained = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3), pooling='avg')
    for layer in pre_trained.layers:
        layer.trainable = False

    x = pre_trained.output
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=pre_trained.input, outputs=predictions)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc', curve='ROC')],
    )
    return model


def run_single_experiment(
    train_generator,
    validation_generator,
    batch_size,
    epochs,
    *,
    img_rows: int,
    img_cols: int,
    checkpoint_path: str,
    log_csv_path: str | None = None,
    early_stop_patience: int = 5,
    min_epochs: int = 5,
):
    model = build_model(img_rows, img_cols)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            save_weights_only=False,
            monitor='val_auc',
            mode='max',
        )
    ]
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=early_stop_patience,
            start_from_epoch=min_epochs,
            restore_best_weights=False,
            verbose=1,
        )
    )
    if log_csv_path is not None:
        callbacks.append(tf.keras.callbacks.CSVLogger(log_csv_path))

    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, math.ceil(train_generator.samples / batch_size)),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=max(1, math.ceil(validation_generator.samples / batch_size)),
        callbacks=callbacks,
    )

    return model, history


def parse_args():
    p = argparse.ArgumentParser(description='Train InceptionV3 on spectrogram JPGs using TSV splits.')
    p.add_argument('--train_tsv', type=str, required=True, help='Train TSV (columns: ID, AUDIOFILE, DIAGNOSIS)')
    p.add_argument('--val_tsv', type=str, required=True, help='Validation TSV (columns: ID, AUDIOFILE, DIAGNOSIS)')
    p.add_argument('--test_tsv', type=str, default=None, help='Optional test TSV for final evaluation + detailed TSV output')
    p.add_argument('--output_dir', type=str, required=True, help='Output directory for this fold/run (files written directly here)')

    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--early_stop_patience', type=int, default=5)
    p.add_argument('--min_epochs', type=int, default=5, help='Do not allow early stopping before this many epochs')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--img_size', type=int, default=600)
    p.add_argument('--n_runs', type=int, default=1)
    p.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return p.parse_args()


def compute_auc_from_generator(model, generator, batch_size: int) -> float:
    n = int(getattr(generator, "samples", len(getattr(generator, "classes", []))))
    steps = max(1, math.ceil(n / batch_size))
    y_prob = model.predict(generator, batch_size=batch_size, steps=steps, verbose=0).reshape(-1)
    y_true = np.asarray(generator.classes).reshape(-1)
    y_prob = y_prob[:n]
    y_true = y_true[:n]
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return float(roc_auc_score(y_true, y_prob))


def compute_best_threshold_from_generator(model, generator, batch_size: int, *, threshold_grid: int = 101) -> float:
    n = int(getattr(generator, "samples", len(getattr(generator, "classes", []))))
    steps = max(1, math.ceil(n / batch_size))
    y_prob = model.predict(generator, batch_size=batch_size, steps=steps, verbose=0).reshape(-1)[:n]
    y_true = np.asarray(generator.classes).reshape(-1)[:n]
    return float(_select_threshold_max_f1(y_true, y_prob, num_thresholds=threshold_grid))


def write_detailed_results_tsv(
    *,
    model,
    test_tsv_path: str,
    classes: list[str],
    img_rows: int,
    img_cols: int,
    batch_size: int,
    out_path: str,
    threshold: float = 0.5,
):
    df = load_tsv(test_tsv_path)
    class_to_index = {c: i for i, c in enumerate(classes)}
    df["true_label"] = df["label"].map(class_to_index)
    if df["true_label"].isna().any():
        bad = df.loc[df["true_label"].isna(), "label"].unique().tolist()
        raise ValueError(f"Found labels not in classes mapping: {bad}. classes={classes}")
    df["true_label"] = df["true_label"].astype(int)

    gen = ImageDataGenerator(rescale=1.0 / 255, fill_mode='nearest').flow_from_dataframe(
        df,
        x_col="filename",
        y_col="label",
        classes=classes,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False,
    )
    n = int(getattr(gen, "samples", len(df)))
    steps = max(1, math.ceil(n / batch_size))
    # The sigmoid output is the positive-class probability in [0, 1].
    y_prob = model.predict(gen, batch_size=batch_size, steps=steps, verbose=0).reshape(-1)[:n]
    # y_prob = np.clip(y_prob, 0.0, 1.0)
    y_pred = (y_prob >= float(threshold)).astype(int)

    # Output: original test TSV columns + appended columns (ordered)
    df_out = df[["ID", "AUDIOFILE", "DIAGNOSIS"]].copy()
    df_out["probability"] = y_prob
    df_out["true_label"] = df["true_label"].values[:n]
    df_out["predicted_label"] = y_pred
    df_out.to_csv(out_path, sep='\t', index=False)


def summarize_results(model, eval_generator, batch_size, all_auc, *, threshold: float = 0.5, out_summary_path: str | None = None):
    print("\n")
    n = int(getattr(eval_generator, "samples", len(getattr(eval_generator, "classes", []))))
    steps = max(1, math.ceil(n / batch_size))
    Y_prob = model.predict(eval_generator, batch_size=batch_size, steps=steps, verbose=0).reshape(-1)[:n]
    y_true = np.asarray(eval_generator.classes).reshape(-1)[:n]
    y_pred = (Y_prob >= float(threshold)).astype(int)
    print('Confusion Matrix')
    print(confusion_matrix(y_true, y_pred))
    print('Classification Report:')
    class_indices = getattr(eval_generator, "class_indices", {})
    if class_indices:
        target_names = [name for name, idx in sorted(class_indices.items(), key=lambda x: x[1])]
    else:
        target_names = ['healthy', 'parkinson']
    print(classification_report(y_true, y_pred, target_names=target_names))
    print("avg auc: {} ({})".format(np.round(np.average(all_auc), 4), np.round(np.std(all_auc), 3)))

    if out_summary_path is not None:
        summary = {
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "threshold": float(threshold),
            "f1": float(metrics.f1_score(y_true, y_pred)) if len(np.unique(y_true)) >= 2 else None,
            "avg_auc": float(np.nanmean(all_auc)) if len(all_auc) else None,
            "std_auc": float(np.nanstd(all_auc)) if len(all_auc) else None,
            "n": int(n),
        }
        with open(out_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


def plot_history(history, all_auc):
    hist = getattr(history, "history", {}) or {}

    accs = list(hist.get('accuracy', []))
    val_accs = list(hist.get('val_accuracy', []))
    if accs:
        plt.plot(range(1, len(accs) + 1), accs, label='Training_accuracy')
    if val_accs:
        plt.plot(range(1, len(val_accs) + 1), val_accs, label='Validation_accuracy')
    if accs or val_accs:
        plt.legend()
        plt.show()

    losses = list(hist.get('loss', []))
    val_losses = list(hist.get('val_loss', []))
    if losses:
        plt.plot(range(1, len(losses) + 1), losses, label='Training_loss')
    if val_losses:
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation_loss')
    if losses or val_losses:
        plt.legend()
        plt.show()

    print('AUC vector: ', all_auc)
    DF = pd.DataFrame(all_auc)
    DF.to_csv("mPowerHiFreqData2AUC.csv")


def main():
    args = parse_args()
    start = timeit.default_timer()

    # Reproducibility: seed Python/NumPy/TensorFlow and Keras utils.
    seed = int(getattr(args, "seed", 42))
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    # Optional TSV inputs (minimal-change support):
    # TSV columns: ID, AUDIOFILE, DIAGNOSIS
    # AUDIOFILE is wav path; jpg path is derived by:
    #   wavpath.replace('fortrain', 'fortrain_jpg').replace('.wav', '.jpg')
    train_tsv_path = args.train_tsv
    val_tsv_path = args.val_tsv
    test_tsv_path = args.test_tsv

    run_dir = args.output_dir
    os.makedirs(run_dir, exist_ok=True)
    best_ckpt_path = os.path.join(run_dir, "best_model.keras")
    train_log_csv = os.path.join(run_dir, "train_log.csv")
    eval_summary_json = os.path.join(run_dir, "eval_summary.json")
    detailed_results_path = os.path.join(run_dir, "Neurovoz_and_PC_GITA_detailed_results.tsv")
    tuned_threshold_path = os.path.join(run_dir, "tuned_threshold.json")

    img_rows = args.img_size
    img_cols = args.img_size
    batch_size = args.batch_size
    epochs = args.epochs
    n_runs = args.n_runs

    all_auc = []
    last_history = None
    last_model = None
    train_generator = None
    validation_generator = None
    eval_generator = None
    tsv_classes = None

    for run in range(n_runs):
        print("run={}".format(run))
        if train_tsv_path is None:
            train_generator, validation_generator = build_data_generators(
                data_path, img_rows, img_cols, batch_size
            )
        else:
            train_generator, validation_generator = build_data_generators_from_tsv(
                train_tsv_path,
                img_rows,
                img_cols,
                batch_size,
                val_tsv_path=val_tsv_path,
                seed=seed,
            )
            tsv_classes = [name for name, idx in sorted(train_generator.class_indices.items(), key=lambda x: x[1])]
            if test_tsv_path is not None:
                eval_generator = build_eval_generator_from_tsv(
                    test_tsv_path, img_rows, img_cols, batch_size, classes=tsv_classes
                )
            else:
                eval_generator = validation_generator

        model, history = run_single_experiment(
            train_generator,
            validation_generator,
            batch_size,
            epochs,
            img_rows=img_rows,
            img_cols=img_cols,
            checkpoint_path=best_ckpt_path,
            log_csv_path=train_log_csv,
            early_stop_patience=args.early_stop_patience,
            min_epochs=args.min_epochs,
        )

        # Evaluate AUC on validation using probabilities
        all_auc.append(compute_auc_from_generator(model, validation_generator, batch_size))
        last_history = history
        last_model = model

    if last_model is not None and validation_generator is not None:
        # Load best checkpoint (selected by val_auc) for final test evaluation
        best_model = tf.keras.models.load_model(best_ckpt_path)

        # Tune threshold on validation set (best model), then apply to test evaluation.
        tuned_threshold = compute_best_threshold_from_generator(best_model, validation_generator, batch_size, threshold_grid=101)
        with open(tuned_threshold_path, "w", encoding="utf-8") as f:
            json.dump({"threshold": float(tuned_threshold), "strategy": "max_f1", "grid": 101}, f, indent=2)
        print(f"[THRESHOLD] Tuned on val (max_f1, grid=101): {tuned_threshold:.4f}")
        print(f"[THRESHOLD] Saved to: {tuned_threshold_path}")

        # Recompute ROC-AUC on the full validation set using sklearn for correctness/consistency
        all_auc = [compute_auc_from_generator(best_model, validation_generator, batch_size)]
        summarize_results(
            best_model,
            eval_generator or validation_generator,
            batch_size,
            all_auc,
            threshold=tuned_threshold,
            out_summary_path=eval_summary_json,
        )

        if test_tsv_path is not None and tsv_classes is not None:
            write_detailed_results_tsv(
                model=best_model,
                test_tsv_path=test_tsv_path,
                classes=tsv_classes,
                img_rows=img_rows,
                img_cols=img_cols,
                batch_size=batch_size,
                out_path=detailed_results_path,
                threshold=tuned_threshold,
            )

    stop = timeit.default_timer()
    print('RunTime: ', round(stop - start, 2), 'Seconds')


if __name__ == "__main__":
    main()
