
import os
import json
import time
import math
import random
import shutil
import warnings
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from transformers.utils import logging as hf_logging


# keep notebook output clean
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*Some weights of .* were not initialized.*")
hf_logging.set_verbosity_error()


model_id = "thomas-sounack/BioClinical-ModernBERT-base"
default_max_length = 512


# small utility helpers

def print_divider(char="=", width=88):
    print(char * width)


def print_stage_header(stage_title, stage_explainer):
    print()
    print_divider()
    print(stage_title.upper())
    print_divider()
    print(stage_explainer)
    print()


def short_name(name, max_len=44):
    return name if len(name) <= max_len else name[: max_len - 3] + "..."


def save_json(obj, path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# tokenizer + tokenisation

def get_tokenizer(model_name=model_id):
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_pair_batch(batch, tokenizer, max_length=default_max_length):
    """
    Tokenise question + context pairs.

    We always preserve the question and only truncate the context when needed.
    """
    return tokenizer(
        batch["QUESTION"],
        batch["CONTEXTS"],
        truncation="only_second",
        max_length=max_length,
    )


def _columns_to_remove(dataset_columns, keep_id=False):
    keep_cols = {"input_ids", "attention_mask", "token_type_ids", "labels", "example_weights", "soft_labels"}
    if keep_id:
        keep_cols.add("ID")
    return [col for col in dataset_columns if col not in keep_cols]


def tokenise_dataset_dict(dataset_dict, tokenizer, max_length=default_max_length, keep_id=False):
    tokenized = dataset_dict.map(
        lambda batch: tokenize_pair_batch(batch, tokenizer=tokenizer, max_length=max_length),
        batched=True,
    )

    cleaned = {}
    for split_name, split_dataset in tokenized.items():
        remove_cols = _columns_to_remove(split_dataset.column_names, keep_id=keep_id)
        if len(remove_cols) > 0:
            cleaned[split_name] = split_dataset.remove_columns(remove_cols)
        else:
            cleaned[split_name] = split_dataset

    return DatasetDict(cleaned)


def tokenise_single_dataset(dataset, tokenizer, max_length=default_max_length, keep_id=False):
    tokenized = dataset.map(
        lambda batch: tokenize_pair_batch(batch, tokenizer=tokenizer, max_length=max_length),
        batched=True,
    )
    remove_cols = _columns_to_remove(tokenized.column_names, keep_id=keep_id)
    if len(remove_cols) > 0:
        tokenized = tokenized.remove_columns(remove_cols)
    return tokenized


class BetterDataCollatorWithPadding:
    """
    Pads transformer inputs while carrying over labels, soft labels, and per-example weights.
    """

    def __init__(self, tokenizer):
        self.base_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, features):
        labels = [feature.pop("labels", None) for feature in features]
        soft_labels = [feature.pop("soft_labels", None) for feature in features]
        example_weights = [feature.pop("example_weights", None) for feature in features]
        ids = [feature.pop("ID", None) for feature in features]

        batch = self.base_collator(features)

        if any(label is not None for label in labels):
            batch["labels"] = torch.tensor(labels, dtype=torch.long)

        if any(weight is not None for weight in example_weights):
            filled = [1.0 if weight is None else float(weight) for weight in example_weights]
            batch["example_weights"] = torch.tensor(filled, dtype=torch.float)

        if any(soft is not None for soft in soft_labels):
            first_non_none = next((soft for soft in soft_labels if soft is not None), None)
            dim = len(first_non_none) if first_non_none is not None else 0
            filled = []
            for soft in soft_labels:
                if soft is None:
                    filled.append([0.0] * dim)
                else:
                    filled.append([float(x) for x in soft])
            batch["soft_labels"] = torch.tensor(filled, dtype=torch.float)

        if any(example_id is not None for example_id in ids):
            batch["ID"] = ids

        return batch


def get_data_collator(tokenizer):
    return BetterDataCollatorWithPadding(tokenizer)


# dataset helpers

def build_class_weights_from_labels(labels, num_classes):
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for class_id in range(num_classes):
        count = counts.get(class_id, 0)
        if count == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float)


def build_label_balanced_sample_weights(labels):
    counts = Counter(labels)
    sample_weights = [1.0 / counts[label] for label in labels]
    return torch.tensor(sample_weights, dtype=torch.double)


def describe_label_counts(dataset, id_to_label):
    counts = Counter(dataset["labels"])
    return {id_to_label[k]: counts.get(k, 0) for k in sorted(id_to_label)}


def save_dataset_split(dataset_dict, save_path):
    folder = os.path.dirname(save_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    dataset_dict.save_to_disk(save_path)


def load_dataset_split(save_path):
    return load_from_disk(save_path)


def export_model_artifact(checkpoint_path, tokenizer, export_dir, metadata=None):
    os.makedirs(export_dir, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.save_pretrained(export_dir)
    tokenizer.save_pretrained(export_dir)

    if metadata is not None:
        save_json(metadata, os.path.join(export_dir, "metadata.json"))


# metrics + reporting

def compute_classification_metrics_from_arrays(logits, labels, ordered_label_ids):
    preds = np.argmax(logits, axis=-1)

    per_class_f1 = f1_score(
        labels,
        preds,
        average=None,
        labels=ordered_label_ids,
        zero_division=0,
    )

    results = {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }

    return results, per_class_f1, preds


def add_confusion_metrics(results, labels, preds, id_to_label):
    matrix = confusion_matrix(labels, preds, labels=list(sorted(id_to_label)))
    results["confusion_matrix"] = matrix.tolist()
    return results


def mean_metrics_across_runs(runs):
    metric_names = sorted(runs[0]["final_metrics"].keys())
    return {
        metric_name: float(np.mean([run["final_metrics"][metric_name] for run in runs]))
        for metric_name in metric_names
    }


def std_metrics_across_runs(runs):
    metric_names = sorted(runs[0]["final_metrics"].keys())
    return {
        metric_name: float(np.std([run["final_metrics"][metric_name] for run in runs]))
        for metric_name in metric_names
    }


def pick_best_run(runs, metric_name="macro_f1"):
    return max(runs, key=lambda x: x["final_metrics"][metric_name])


def print_results_table(results, title):
    print()
    print_divider()
    print(title)
    print_divider()

    for row in results:
        metrics = row["final_metrics"]
        ordered_keys = [key for key in ["accuracy", "macro_f1", "f1_no", "f1_maybe", "f1_yes"] if key in metrics]
        metrics_text = " | ".join([f"{key}={metrics[key]:.4f}" for key in ordered_keys])
        print(f"{row['run_name']} | {metrics_text}")


# generic trainer

class GenericTrainer(Trainer):
    """
    Custom Trainer supporting:
        - weighted loss
        - weighted sampler
        - soft labels
        - per-example loss weights
    """

    def __init__(
        self,
        *args,
        class_weights=None,
        sample_weights=None,
        use_weighted_loss=False,
        use_weighted_sampler=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.sample_weights = sample_weights
        self.use_weighted_loss = use_weighted_loss
        self.use_weighted_sampler = use_weighted_sampler

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)
        soft_labels = inputs.pop("soft_labels", None)
        example_weights = inputs.pop("example_weights", None)

        outputs = model(**inputs)
        logits = outputs.logits

        if soft_labels is not None:
            soft_labels = soft_labels.to(logits.device)
            per_example_loss = -(soft_labels * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        else:
            labels = labels.to(logits.device)
            class_weights = None
            if self.use_weighted_loss and self.class_weights is not None:
                class_weights = self.class_weights.to(logits.device)

            per_example_loss = F.cross_entropy(
                logits,
                labels,
                weight=class_weights,
                reduction="none",
            )

        if example_weights is not None:
            example_weights = example_weights.to(logits.device)
            per_example_loss = per_example_loss * example_weights

        loss = per_example_loss.mean()
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        if not self.use_weighted_sampler:
            return super().get_train_dataloader()

        sampler = WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.sample_weights),
            replacement=True,
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=False,
        )


def build_training_args(
    run_name,
    seed,
    learning_rate,
    train_batch_size,
    eval_batch_size,
    gradient_accumulation_steps,
    num_train_epochs,
    runs_root="./runs",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    logging_steps=None,
    disable_tqdm=False,
    early_stopping_patience=2,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    dataloader_num_workers=2,
):
    output_dir = os.path.join(runs_root, run_name)

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        logging_strategy=logging_strategy,
        logging_steps=logging_steps,
        logging_first_step=(logging_strategy == "steps"),
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_total_limit=1,
        remove_unused_columns=False,
        dataloader_num_workers=dataloader_num_workers,
        report_to="none",
        seed=seed,
        data_seed=seed,
        disable_tqdm=disable_tqdm,
        save_only_model = True
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    return args, callbacks


# calibration helpers

def logits_to_probs(logits):
    logits_tensor = torch.tensor(logits, dtype=torch.float)
    probs = torch.softmax(logits_tensor, dim=-1).cpu().numpy()
    return probs


def apply_temperature_to_logits(logits, temperature):
    return logits / float(temperature)


def fit_temperature_scaler(logits, labels, max_iter=200):
    """
    Learns one scalar temperature on a hold-out split by minimising cross-entropy.
    """
    logits_tensor = torch.tensor(logits, dtype=torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    log_temperature = torch.nn.Parameter(torch.zeros(1))
    optimiser = torch.optim.LBFGS([log_temperature], lr=0.01, max_iter=max_iter)

    def closure():
        optimiser.zero_grad()
        temperature = torch.exp(log_temperature).clamp(min=1e-3, max=100.0)
        loss = F.cross_entropy(logits_tensor / temperature, labels_tensor)
        loss.backward()
        return loss

    optimiser.step(closure)
    learned_temperature = float(torch.exp(log_temperature).detach().cpu().item())
    return max(learned_temperature, 1e-3)


# prediction helpers

def _move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def predict_logits(model, tokenized_dataset, tokenizer, batch_size=8):
    device = get_device()
    model = model.to(device)
    model.eval()

    collator = get_data_collator(tokenizer)
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False,
    )

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels", None)
            batch.pop("example_weights", None)
            batch.pop("soft_labels", None)
            batch.pop("ID", None)

            batch = _move_batch_to_device(batch, device)
            outputs = model(**batch)
            logits = outputs.logits.detach().cpu().numpy()
            all_logits.append(logits)

            if labels is not None:
                all_labels.append(labels.detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0) if len(all_labels) > 0 else None
    return logits, labels


def predict_probs_from_checkpoint(
    checkpoint_path,
    dataset,
    tokenizer,
    temperature=None,
    batch_size=8,
    max_length=default_max_length,
):
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    tokenized_dataset = tokenise_single_dataset(
        dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        keep_id=False,
    )
    logits, labels = predict_logits(model, tokenized_dataset, tokenizer=tokenizer, batch_size=batch_size)

    if temperature is not None:
        logits = apply_temperature_to_logits(logits, temperature)

    probs = logits_to_probs(logits)
    return probs, labels


def evaluate_probs(probs, labels, ordered_label_ids, id_to_label=None):
    preds = np.argmax(probs, axis=-1)
    results = {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }

    per_class_f1 = f1_score(
        labels,
        preds,
        average=None,
        labels=ordered_label_ids,
        zero_division=0,
    )

    for idx, class_id in enumerate(ordered_label_ids):
        class_name = id_to_label[class_id] if id_to_label is not None else str(class_id)
        results[f"f1_{class_name}"] = float(per_class_f1[idx])

    if id_to_label is not None:
        add_confusion_metrics(results, labels, preds, id_to_label)

    return results, preds
