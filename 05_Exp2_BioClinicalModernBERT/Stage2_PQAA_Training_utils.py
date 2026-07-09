from Stage2_common_training_utils import *

_common_save_dataset_split = save_dataset_split
_common_load_dataset_split = load_dataset_split
_common_build_training_args = build_training_args


# fixed weak-stage config

model_id = model_id
label_to_id = {"no": 0, "yes": 1}
id_to_label = {0: "no", 1: "yes"}


# loading + preprocessing

def load_weak_json(path):
    """
    Load PQA-A json into a Hugging Face Dataset.

    Fields per row:
        - QUESTION
        - CONTEXTS
        - final_decision
        
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
    example["labels"] = label_to_id[example["final_decision"]]
    return example


def build_full_split(dataset, val_size=0.1, seed=7):
    split = dataset.train_test_split(test_size=val_size, seed=seed)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


def build_pilot_split(full_split, pilot_fraction=0.1, seed=7):
    pilot_train_n = max(1, int(len(full_split["train"]) * pilot_fraction))
    pilot_val_n = max(1, int(len(full_split["validation"]) * pilot_fraction))

    pilot_train = full_split["train"].shuffle(seed=seed).select(range(pilot_train_n))
    pilot_val = full_split["validation"].shuffle(seed=seed).select(range(pilot_val_n))

    return DatasetDict({"train": pilot_train, "validation": pilot_val})


def save_dataset_split(dataset_dict, save_path):
    return _common_save_dataset_split(dataset_dict, save_path)


def load_dataset_split(save_path):
    return _common_load_dataset_split(save_path)


# tokenisation

def get_tokenizer(model_id=model_id):
    return AutoTokenizer.from_pretrained(model_id)


def tokenize_batch(batch, tokenizer, max_length=512):
    return tokenize_pair_batch(batch, tokenizer=tokenizer, max_length=max_length)


def tokenize_split(dataset_dict, tokenizer, max_length=512):
    return tokenise_dataset_dict(dataset_dict, tokenizer=tokenizer, max_length=max_length, keep_id=False)


def get_data_collator(tokenizer):
    return BetterDataCollatorWithPadding(tokenizer)


# summaries / weighting

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
    return build_class_weights_from_labels(train_dataset["labels"], num_classes=2)


def build_sample_weights(train_dataset):
    return build_label_balanced_sample_weights(train_dataset["labels"])


# model + metrics

def make_binary_model(model_id=model_id):
    return AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        id2label=id_to_label,
        label2id=label_to_id,
    )


def compute_binary_metrics(eval_pred):
    logits, labels = eval_pred
    results, per_class_f1, _ = compute_classification_metrics_from_arrays(
        logits=logits,
        labels=labels,
        ordered_label_ids=[0, 1],
    )
    results["f1_no"] = float(per_class_f1[0])
    results["f1_yes"] = float(per_class_f1[1])
    return results


class WeakTrainer(GenericTrainer):
    pass


def build_training_args(
    output_dir,
    learning_rate,
    num_train_epochs,
    train_batch_size,
    eval_batch_size,
    gradient_accumulation_steps,
    seed=7,
    disable_tqdm=False,
    logging_strategy="epoch",
    logging_steps=None,
    early_stopping_patience=2,
):
    run_name = os.path.basename(output_dir.rstrip("/"))
    runs_root = os.path.dirname(output_dir.rstrip("/")) or "."

    return _common_build_training_args(
        run_name=run_name,
        seed=seed,
        learning_rate=learning_rate,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        runs_root=runs_root,
        disable_tqdm=disable_tqdm,
        logging_strategy=logging_strategy,
        logging_steps=logging_steps,
        early_stopping_patience=early_stopping_patience,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
    )


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
    disable_tqdm=False,
    logging_strategy="epoch",
    logging_steps=None,
    early_stopping_patience=2,
):
    set_all_seeds(seed)

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
        disable_tqdm=disable_tqdm,
        logging_strategy=logging_strategy,
        logging_steps=logging_steps,
        early_stopping_patience=early_stopping_patience,
    )

    trainer = WeakTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_split["train"],
        eval_dataset=tokenized_split["validation"],
        processing_class=tokenizer,
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
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "final_metrics": {
            "accuracy": metrics["eval_accuracy"],
            "macro_f1": metrics["eval_macro_f1"],
            "f1_no": metrics["eval_f1_no"],
            "f1_yes": metrics["eval_f1_yes"],
        },
    }


def print_results_table(results, title):
    print()
    print_divider()
    print(title)
    print_divider()

    for row in results:
        metrics = row["final_metrics"]
        print(
            f"{row['run_name']}"
            f" | acc={metrics['accuracy']:.4f}"
            f" | macro_f1={metrics['macro_f1']:.4f}"
            f" | f1_no={metrics['f1_no']:.4f}"
            f" | f1_yes={metrics['f1_yes']:.4f}"
            f" | loss={row['use_weighted_loss']}"
            f" | sampler={row['use_weighted_sampler']}"
        )


def choose_best_result(results, key="macro_f1"):
    return max(results, key=lambda x: x["final_metrics"][key])


def export_best_weak_model(best_checkpoint_path, tokenizer, export_dir, metadata=None):
    export_model_artifact(
        checkpoint_path=best_checkpoint_path,
        tokenizer=tokenizer,
        export_dir=export_dir,
        metadata=metadata,
    )


def delete_non_best_run_dirs(results, best_run_name, runs_root="./weak_runs"):
    for row in results:
        run_name = row["run_name"]
        run_dir = os.path.join(runs_root, run_name)
        if run_name != best_run_name and os.path.exists(run_dir):
            shutil.rmtree(run_dir)
