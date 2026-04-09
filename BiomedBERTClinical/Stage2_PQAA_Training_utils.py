# Functions needed for the task

import os
import json
import time
import shutil
import warnings
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from datasets import Dataset, DatasetDict, load_from_disk
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from transformers.utils import logging as hf_logging


# environment / logging

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore", message=".*Some weights of .* were not initialized.*")
hf_logging.set_verbosity_error()


#  config
model_id = "thomas-sounack/BioClinical-ModernBERT-base"

label_to_id = {"no": 0, "yes": 1}
id_to_label = {0: "no", 1: "yes"}


#
# small print helpers

def print_divider(char="=", width=88):
    print(char * width)


def print_stage_header(title, text):
    print()
    print_divider()
    print(title)
    print_divider()
    print(text)
    print()


# loading + preprocessing

def load_weak_json(path):
    """
    Load weak-label json into a Hugging Face Dataset.

     fields per row:
        - QUESTION
        - CONTEXTS
        - final_decision

    Weak stage is binary only: yes / no.
    """
    with open(path, "r") as f:
        raw = json.load(f)

    rows = []

    if isinstance(raw, dict):
        iterable = raw.items()
    else:
        iterable = enumerate(raw)

    for example_id, row in iterable:
        label = row["final_decision"].strip().lower()

        rows.append(
            {
                "ID": str(example_id),
                "QUESTION": row["QUESTION"],
                "CONTEXTS": " ".join(row["CONTEXTS"]),
                "final_decision": label,
            }
        )

    return Dataset.from_list(rows)


def validate_weak_dataset(dataset):
    """
    Check that weak labels are binary and clean.
    """
    
    labels = [x["final_decision"] for x in dataset]
    unique_labels = sorted(set(labels))

    if any(label not in label_to_id for label in unique_labels):
        raise ValueError(
            f"Weak dataset must contain only {list(label_to_id.keys())}, got {unique_labels}"
        )

    print_divider()
    print("WEAK DATA CHECK")
    print_divider()
    print(f"Examples      : {len(dataset)}")
    print(f"Label counts  : {dict(Counter(labels))}")
    print(f"Unique labels : {unique_labels}")


def add_binary_label(example):
    """
    Map string label to integer label id expected by Trainer.
    """
    example["labels"] = label_to_id[example["final_decision"]]
    return example


def build_full_split(dataset, val_size=0.1, seed=7):
    """
    Build one fixed full split that all later work reuses.
    """
    split = dataset.train_test_split(test_size=val_size, seed=seed)

    return DatasetDict(
        {
            "train": split["train"],
            "validation": split["test"],
        }
    )


def build_pilot_split(full_split, pilot_fraction=0.1, seed=7):
    """
    Build a smaller pilot version from the already-created full split.

    Using this to know which sampling technique to use to handle class imbalance 
    in this binary problem.
    """
    pilot_train_n = max(1, int(len(full_split["train"]) * pilot_fraction))
    pilot_val_n = max(1, int(len(full_split["validation"]) * pilot_fraction))

    pilot_train = full_split["train"].shuffle(seed=seed).select(range(pilot_train_n))
    pilot_val = full_split["validation"].shuffle(seed=seed).select(range(pilot_val_n))

    return DatasetDict(
        {
            "train": pilot_train,
            "validation": pilot_val,
        }
    )


def save_dataset_split(dataset_dict, save_path):
    """
    Save split.
    """
    dataset_dict.save_to_disk(save_path)


def load_dataset_split(save_path):
    """
    Reload a previously saved split.
    """
    return load_from_disk(save_path)


# tokenisation

def get_tokenizer(model_id=model_id):
    return AutoTokenizer.from_pretrained(model_id)


def tokenize_batch(batch, tokenizer, max_length=512):
    """
    Tokenise question + context pairs.

    We keep the question intact and only truncate the context side.
    """
    return tokenizer(
        batch["QUESTION"],
        batch["CONTEXTS"],
        truncation="only_second",
        max_length=max_length,
    )


def tokenize_split(dataset_dict, tokenizer, max_length=512):
    """
    Tokenise all splits and remove raw text columns afterward.
    """
    tokenized = dataset_dict.map(
        lambda batch: tokenize_batch(batch, tokenizer=tokenizer, max_length=max_length),
        batched=True,
    )

    keep_cols = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    existing_cols = set(tokenized["train"].column_names)
    remove_cols = [c for c in existing_cols if c not in keep_cols]

    tokenized = tokenized.remove_columns(remove_cols)

    return tokenized


def get_data_collator(tokenizer):
    """
    Dynamic padding during batch creation.
    """
    return DataCollatorWithPadding(tokenizer=tokenizer)


# class summaries / weighting

def get_label_counts(dataset):
    counts = Counter(dataset["labels"])
    return {id_to_label[k]: counts.get(k, 0) for k in sorted(id_to_label)}


def print_split_summary(dataset_dict, title="DATA SUMMARY"):
    print()
    print_divider()
    print(title)
    print_divider()

    for split_name in dataset_dict.keys():
        print(f"{split_name} examples : {len(dataset_dict[split_name])}")
        if "labels" in dataset_dict[split_name].column_names:
            print(f"{split_name} label counts : {get_label_counts(dataset_dict[split_name])}")
        print()


def build_class_weights(train_dataset):
    """
    Inverse-frequency class weights for weighted-loss experiments.
    """
    label_counts = Counter(train_dataset["labels"])

    total = sum(label_counts.values())
    num_classes = len(label_to_id)

    weights = []
    for class_id in range(num_classes):
        count = label_counts[class_id]
        weight = total / (num_classes * count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float)


def build_sample_weights(train_dataset):
    """
    Sample-level weights for WeightedRandomSampler.
    """
    label_counts = Counter(train_dataset["labels"])

    sample_weights = []
    for label in train_dataset["labels"]:
        sample_weights.append(1.0 / label_counts[label])

    return torch.tensor(sample_weights, dtype=torch.double)


# model

def make_binary_model(model_id=model_id):
    return AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        id2label=id_to_label,
        label2id=label_to_id,
    )


# metrics

def compute_binary_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    per_class_f1 = f1_score(
        labels,
        preds,
        average=None,
        labels=[0, 1],
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_no": per_class_f1[0],
        "f1_yes": per_class_f1[1],
    }


# custom trainer

class WeakTrainer(Trainer):
    """
    Custom Trainer supporting:
        - weighted loss
        - weighted sampler
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
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.use_weighted_loss:
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device)
            )
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits, labels)
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

# training args

def build_training_args(
    output_dir,
    learning_rate,
    num_train_epochs,
    train_batch_size,
    eval_batch_size,
    gradient_accumulation_steps,
    dataloader_num_workers=2,
    early_stopping_patience=2,
    seed=7,
):
    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_total_limit=1,
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=dataloader_num_workers,
        report_to="none",
        disable_tqdm=False,
    )

    callbacks = [
    EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=0.001,
    )
]
    return args, callbacks


# one experiment run

def run_weak_experiment(
    run_name,
    tokenized_split,
    tokenizer,
    learning_rate,
    num_train_epochs,
    train_batch_size,
    eval_batch_size,
    gradient_accumulation_steps,
    class_weights,
    sample_weights,
    use_weighted_loss,
    use_weighted_sampler,
    seed=7,
    runs_root="./weak_runs",
):
    model = make_binary_model()

    data_collator = get_data_collator(tokenizer)

    output_dir = os.path.join(runs_root, run_name)
    training_args, callbacks = build_training_args(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=seed,
    )

    trainer = WeakTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_split["train"],
        eval_dataset=tokenized_split["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_binary_metrics,
        callbacks=callbacks,
        class_weights=class_weights,
        sample_weights=sample_weights,
        use_weighted_loss=use_weighted_loss,
        use_weighted_sampler=use_weighted_sampler,
    )

    start = time.time()
    trainer.train()
    runtime_minutes = (time.time() - start) / 60

    metrics = trainer.evaluate()

    return {
        "run_name": run_name,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "use_weighted_loss": use_weighted_loss,
        "use_weighted_sampler": use_weighted_sampler,
        "runtime_minutes": runtime_minutes,
        "final_metrics": {
            "accuracy": metrics["eval_accuracy"],
            "macro_f1": metrics["eval_macro_f1"],
            "f1_no": metrics["eval_f1_no"],
            "f1_yes": metrics["eval_f1_yes"],
        },
        "best_checkpoint": trainer.state.best_model_checkpoint,
    }


# results printing

def print_results_table(results, title):
    print()
    print_divider()
    print(title)
    print_divider()

    for row in results:
        m = row["final_metrics"]
        print(
            f"{row['run_name']}"
            f" | acc={m['accuracy']:.4f}"
            f" | macro_f1={m['macro_f1']:.4f}"
            f" | f1_no={m['f1_no']:.4f}"
            f" | f1_yes={m['f1_yes']:.4f}"
            f" | loss={row['use_weighted_loss']}"
            f" | sampler={row['use_weighted_sampler']}"
        )


def choose_best_result(results, key="macro_f1"):
    return max(results, key=lambda x: x["final_metrics"][key])


# final export for Stage 2 reuse in later ablations

def export_best_weak_model(best_checkpoint_path, tokenizer, export_dir, metadata=None):
    """
    Export best artifact for later Stage 2 reuse.
    """
    model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint_path)
    os.makedirs(export_dir, exist_ok=True)

    model.save_pretrained(export_dir)
    tokenizer.save_pretrained(export_dir)

    if metadata is not None:
        with open(os.path.join(export_dir, "weak_training_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)


def delete_non_best_run_dirs(results, best_run_name, runs_root="./weak_runs"):
    """
    Keep only the best weak-training run on disk.
    """
    for row in results:
        run_name = row["run_name"]
        run_dir = os.path.join(runs_root, run_name)

        if run_name != best_run_name and os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        