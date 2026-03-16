#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from datasets import Dataset, load_from_disk
from sklearn.metrics import accuracy_score, f1_score
import torch
from transformers import (BertConfig, BertForSequenceClassification,
                          DefaultDataCollator, Trainer, TrainingArguments)


def _resolve_label_column(dataset: Dataset) -> str:
    if "labels" in dataset.column_names:
        return "labels"
    if "label" in dataset.column_names:
        return "label"
    raise ValueError(
        "Dataset must contain a 'labels' or 'label' column for classification."
    )


def _normalize_dataset(dataset: Dataset, label_column: str) -> Dataset:
    required = {"input_ids", label_column}
    missing = sorted(required - set(dataset.column_names))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
        input_ids = np.asarray(row["input_ids"], dtype=np.int64).reshape(-1)
        attention_mask = row.get("attention_mask")
        if attention_mask is None:
            attention = (input_ids != 0).astype(np.int64)
        else:
            attention = np.asarray(attention_mask, dtype=np.int64).reshape(-1)
        return {
            "input_ids": input_ids.tolist(),
            "attention_mask": attention.tolist(),
            "labels": int(row[label_column]),
        }

    return dataset.map(
        _normalize_row,
        remove_columns=dataset.column_names,
        desc="Normalizing dataset columns")


def _infer_num_labels(train_dataset: Dataset) -> int:
    labels = np.asarray(train_dataset["labels"], dtype=np.int64)
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        raise ValueError("At least two classes are required for classification.")
    return int(unique_labels.size)


def _freeze_first_two_layers(model: BertForSequenceClassification) -> None:
    for layer in model.bert.encoder.layer[:2]:
        for parameter in layer.parameters():
            parameter.requires_grad = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Native Geneformer benchmark with Hugging Face Trainer.")
    parser.add_argument("--train-path",
                        default="./data/hf/train",
                        help="Path to train dataset saved with load_from_disk.")
    parser.add_argument("--test-path",
                        default="./data/hf/test",
                        help="Path to test dataset saved with load_from_disk.")
    parser.add_argument("--output-dir",
                        default="./runs/bench_native",
                        help="Trainer output directory.")
    return parser.parse_args()


def main() -> Tuple[float, float]:
    args = _parse_args()
    train_path = Path(args.train_path)
    test_path = Path(args.test_path)
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected datasets at '{train_path}' and '{test_path}'.")

    if not torch.cuda.is_available():
        raise RuntimeError("fp16=True requires CUDA-enabled runtime.")

    train_raw = load_from_disk(str(train_path))
    test_raw = load_from_disk(str(test_path))
    if not isinstance(train_raw, Dataset) or not isinstance(test_raw, Dataset):
        raise TypeError("Both train and test paths must load Hugging Face Dataset objects.")

    train_label_column = _resolve_label_column(train_raw)
    test_label_column = _resolve_label_column(test_raw)
    train_dataset = _normalize_dataset(train_raw, train_label_column)
    test_dataset = _normalize_dataset(test_raw, test_label_column)

    num_labels = _infer_num_labels(train_dataset)
    config = BertConfig.from_pretrained("ctheodoris/Geneformer",
                                        subfolder="Geneformer-V1-10M",
                                        num_labels=num_labels)
    model = BertForSequenceClassification.from_pretrained(
        "ctheodoris/Geneformer",
        subfolder="Geneformer-V1-10M",
        config=config,
        ignore_mismatched_sizes=True)
    _freeze_first_two_layers(model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        fp16=True,
        learning_rate=5e-5,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        num_train_epochs=1,
        report_to=[],
        logging_strategy="no",
        save_strategy="no",
    )

    def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro")),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate(eval_dataset=test_dataset)

    accuracy = float(metrics["eval_accuracy"])
    macro_f1 = float(metrics["eval_macro_f1"])
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Macro F1: {macro_f1:.6f}")
    return accuracy, macro_f1


if __name__ == "__main__":
    main()
