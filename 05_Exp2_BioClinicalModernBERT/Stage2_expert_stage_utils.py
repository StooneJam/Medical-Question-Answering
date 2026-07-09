from Stage2_common_training_utils import *

import math
import random
from collections import defaultdict

# fixed expert-stage config
model_id = model_id
label_to_id = {"no": 0, "maybe": 1, "yes": 2}
id_to_label = {0: "no", 1: "maybe", 2: "yes"}


# loading + preprocessing

def load_expert_json(path):
    with open(path, "r") as f:
        raw = json.load(f)

    rows = []
    if isinstance(raw, dict):
        iterable = raw.items()
    else:
        iterable = enumerate(raw)

    for example_id, row in iterable:
        rows.append(
            {
                "ID": str(example_id),
                "QUESTION": row["QUESTION"],
                "CONTEXTS": " ".join(row["CONTEXTS"]),
                "final_decision": row["final_decision"].strip().lower(),
            }
        )

    return Dataset.from_list(rows)


def load_unlabelled_json(path):
    with open(path, "r") as f:
        raw = json.load(f)

    rows = []
    if isinstance(raw, dict):
        iterable = raw.items()
    else:
        iterable = enumerate(raw)

    for example_id, row in iterable:
        rows.append(
            {
                "ID": str(example_id),
                "QUESTION": row["QUESTION"],
                "CONTEXTS": " ".join(row["CONTEXTS"]),
            }
        )

    return Dataset.from_list(rows)


def add_expert_label(example):
    example["labels"] = label_to_id[example["final_decision"]]
    return example


def validate_expert_dataset(dataset):
    labels = [x["final_decision"] for x in dataset]
    unique_labels = sorted(set(labels))

    if any(label not in label_to_id for label in unique_labels):
        raise ValueError(
            f"Expert dataset must contain only {list(label_to_id.keys())}, got {unique_labels}"
        )

    print_divider()
    print("EXPERT DATA CHECK")
    print_divider()
    print(f"Examples      : {len(dataset)}")
    print(f"Label counts  : {dict(Counter(labels))}")
    print(f"Unique labels : {unique_labels}")


def build_official_500_500_split(dataset, seed=0):
    """
    Build the labelled split to exactly match the official PubMedQA logic
    inside a Hugging Face Dataset.

    """

    rng = random.Random(seed)

    id_to_label = {0: "no", 1: "maybe", 2: "yes"}

    def to_label_name(x):
        if isinstance(x, str):
            return x
        return id_to_label[int(x)]

    def split_label(indices, fold):
        indices = list(indices)
        rng.shuffle(indices)
        num_all = len(indices)
        num_split = math.ceil(num_all / fold)

        output = []
        for i in range(fold):
            if i == fold - 1:
                output.append(indices[i * num_split :])
            else:
                output.append(indices[i * num_split : (i + 1) * num_split])
        return output

    def official_split(indices_by_label, fold=2):
        # split each label separately
        split_by_label = {
            label: split_label(indices, fold)
            for label, indices in indices_by_label.items()
        }

        # recombine same-position chunks across labels
        output = []
        for i in range(fold):
            chunk = []
            for label in ["yes", "no", "maybe"]:
                chunk.extend(split_by_label.get(label, [[], []])[i])
            output.append(chunk)

        #  balancing fix:
        if len(output[-1]) != len(output[0]):
            for i in range(fold - 1):
                picked = rng.choice(output[i])
                output[-1].append(picked)
                output[i].remove(picked)

        return output

    labels = dataset["labels"] if "labels" in dataset.column_names else dataset["label"]

    indices_by_label = defaultdict(list)
    for idx, label in enumerate(labels):
        indices_by_label[to_label_name(label)].append(idx)

    split_chunks = official_split(indices_by_label, fold=2)

    dev_idx = sorted(split_chunks[0])
    test_idx = sorted(split_chunks[1])

    dev_500 = dataset.select(dev_idx)
    test_500 = dataset.select(test_idx)

    return dev_500, test_500


def build_outer_cv_folds(dev_dataset, n_splits=10, seed=42):
    """
    Outer stratified 10-fold CV on the 500-development half.

    Each outer fold becomes:
        - 400 train
        - 50 calibration / model-selection
        - 50 untouched fold evaluation
    """
    labels = np.array(dev_dataset["labels"])
    all_indices = np.arange(len(dev_dataset))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []

    for fold_idx, (build_idx, eval_idx) in enumerate(skf.split(all_indices, labels), start=1):
        build_idx = np.array(build_idx)
        eval_idx = np.array(eval_idx)

        build_labels = labels[build_idx]
        inner_split = StratifiedShuffleSplit(
            n_splits=1,
            test_size=50,
            random_state=seed + fold_idx,
        )

        inner_train_pos, calib_pos = next(inner_split.split(build_idx, build_labels))
        train_idx = build_idx[inner_train_pos]
        calib_idx = build_idx[calib_pos]

        fold = {
            "fold_id": fold_idx,
            "train_400": dev_dataset.select(sorted(train_idx.tolist())),
            "calibration_50": dev_dataset.select(sorted(calib_idx.tolist())),
            "eval_50": dev_dataset.select(sorted(eval_idx.tolist())),
        }
        folds.append(fold)

    return folds


def build_final_committee_folds(dev_dataset, n_splits=10, seed=42):
    """
    Final deployment committee over the 500-development half.

    Each committee member uses:
        - 450 train
        - 50 calibration
    """
    labels = np.array(dev_dataset["labels"])
    all_indices = np.arange(len(dev_dataset))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    for fold_idx, (train_idx, calib_idx) in enumerate(skf.split(all_indices, labels), start=1):
        folds.append(
            {
                "member_id": fold_idx,
                "train_450": dev_dataset.select(sorted(train_idx.tolist())),
                "calibration_50": dev_dataset.select(sorted(calib_idx.tolist())),
            }
        )

    return folds


def print_labelled_split_summary(dev_500, test_500, title="LABELLED SPLIT SUMMARY"):
    print()
    print_divider()
    print(title)
    print_divider()
    print(f"dev_500 examples  : {len(dev_500)}")
    print(f"dev_500 counts    : {describe_label_counts(dev_500, id_to_label)}")
    print()
    print(f"test_500 examples : {len(test_500)}")
    print(f"test_500 counts   : {describe_label_counts(test_500, id_to_label)}")


def print_outer_fold_summary(folds, title="OUTER CV FOLDS"):
    print()
    print_divider()
    print(title)
    print_divider()

    for fold in folds:
        print(f"fold {fold['fold_id']}")
        print(f"  train_400       : {len(fold['train_400'])} | {describe_label_counts(fold['train_400'], id_to_label)}")
        print(f"  calibration_50  : {len(fold['calibration_50'])} | {describe_label_counts(fold['calibration_50'], id_to_label)}")
        print(f"  eval_50         : {len(fold['eval_50'])} | {describe_label_counts(fold['eval_50'], id_to_label)}")
        print()


# metrics

def compute_three_class_metrics(eval_pred):
    logits, labels = eval_pred
    results, per_class_f1, _ = compute_classification_metrics_from_arrays(
        logits=logits,
        labels=labels,
        ordered_label_ids=[0, 1, 2],
    )
    results["f1_no"] = float(per_class_f1[0])
    results["f1_maybe"] = float(per_class_f1[1])
    results["f1_yes"] = float(per_class_f1[2])
    return results


# weighting + modelling

def build_class_weights(train_dataset):
    return build_class_weights_from_labels(train_dataset["labels"], num_classes=3)


def build_sample_weights(train_dataset):
    return build_label_balanced_sample_weights(train_dataset["labels"])


def make_warm_started_three_class_model(stage1_checkpoint_path):
    """
    Load the PQA-A trained checkpoint and replace the binary head with a fresh 3-class head.
    """
    return AutoModelForSequenceClassification.from_pretrained(
        stage1_checkpoint_path,
        num_labels=3,
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
    )


# low-level training stages

def run_three_class_training_stage(
    run_name,
    train_dataset,
    selection_dataset,
    tokenizer,
    init_checkpoint_path,
    seed,
    learning_rate,
    num_train_epochs,
    train_batch_size,
    eval_batch_size,
    gradient_accumulation_steps,
    use_weighted_loss=False,
    use_weighted_sampler=True,
    runs_root="./stage2_cv_runs",
    disable_tqdm=False,
    logging_strategy="epoch",
    logging_steps=None,
    early_stopping_patience=2,
):
    set_all_seeds(seed)

    tokenized = tokenise_dataset_dict(
        DatasetDict({"train": train_dataset, "validation": selection_dataset}),
        tokenizer=tokenizer,
        max_length=512,
        keep_id=False,
    )

    class_weights = build_class_weights(tokenized["train"])
    sample_weights = build_sample_weights(tokenized["train"])
    model = make_warm_started_three_class_model(init_checkpoint_path)
    data_collator = get_data_collator(tokenizer)

    args, callbacks = build_training_args(
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

    trainer = GenericTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_three_class_metrics,
        callbacks=callbacks,
        class_weights=class_weights,
        sample_weights=sample_weights,
        use_weighted_loss=use_weighted_loss,
        use_weighted_sampler=use_weighted_sampler,
    )

    start = time.time()
    trainer.train()
    runtime_minutes = (time.time() - start) / 60
    selection_metrics = trainer.evaluate()

    return {
        "run_name": run_name,
        "seed": seed,
        "runtime_minutes": runtime_minutes,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "selection_metrics": {
            "accuracy": selection_metrics["eval_accuracy"],
            "macro_f1": selection_metrics["eval_macro_f1"],
            "f1_no": selection_metrics["eval_f1_no"],
            "f1_maybe": selection_metrics["eval_f1_maybe"],
            "f1_yes": selection_metrics["eval_f1_yes"],
        },
    }


def fit_checkpoint_temperature(checkpoint_path, calibration_dataset, tokenizer, batch_size=8):
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    tokenized_calib = tokenise_single_dataset(
        calibration_dataset,
        tokenizer=tokenizer,
        max_length=512,
        keep_id=False,
    )
    logits, labels = predict_logits(model, tokenized_calib, tokenizer=tokenizer, batch_size=batch_size)
    temperature = fit_temperature_scaler(logits, labels)
    return temperature, logits, labels


def evaluate_calibrated_checkpoint(
    checkpoint_path,
    eval_dataset,
    tokenizer,
    temperature,
    batch_size=8,
):
    probs, labels = predict_probs_from_checkpoint(
        checkpoint_path=checkpoint_path,
        dataset=eval_dataset,
        tokenizer=tokenizer,
        temperature=temperature,
        batch_size=batch_size,
        max_length=512,
    )
    metrics, preds = evaluate_probs(
        probs=probs,
        labels=labels,
        ordered_label_ids=[0, 1, 2],
        id_to_label=id_to_label,
    )
    return metrics, preds, probs


def predict_calibrated_probs_for_dataset(
    checkpoint_path,
    dataset,
    tokenizer,
    temperature,
    batch_size=8,
):
    probs, _ = predict_probs_from_checkpoint(
        checkpoint_path=checkpoint_path,
        dataset=dataset,
        tokenizer=tokenizer,
        temperature=temperature,
        batch_size=batch_size,
        max_length=512,
    )
    return probs


# pseudo-label selection

def build_class_target_plan(
    reference_dataset,
    pseudo_multiplier=1.50,
    maybe_boost=1.25,
    minimum_per_class=0,
):
    """
    Build class-aware pseudo-label quotas from the train dataset distribution.

    We anchor the pseudo-label mix to the expert train-fold distribution,
    then give 'maybe' class a modest boost because it is the fragile class under macro F1.
    """
    counts = Counter(reference_dataset["labels"])

    targets = {}
    for class_id in sorted(id_to_label):
        base_target = counts.get(class_id, 0) * pseudo_multiplier
        if class_id == 1:
            base_target *= maybe_boost
        targets[class_id] = max(int(round(base_target)), minimum_per_class)

    return targets


def _selector_score(confidence, margin):
    return (0.7 * confidence) + (0.3 * margin)


def _make_empty_pseudo_dataset():
    return Dataset.from_dict(
        {
            "ID": [],
            "QUESTION": [],
            "CONTEXTS": [],
            "final_decision": [],
            "labels": [],
            "soft_labels": [],
            "example_weights": [],
            "selector_score": [],
            "pseudo_confidence": [],
            "pseudo_margin": [],
            "data_source": [],
        }
    )


def select_pseudo_labels_best(
    unlabelled_dataset,
    probs_teacher_a,
    probs_teacher_b,
    class_targets,
    min_confidence_by_class=None,
    min_margin_by_class=None,
    base_pseudo_weight=0.35,
):
    """
    Best selector used in the advanced pipeline.

    Rules:
        1. Both teachers must agree on the top class
        2. Mean calibrated confidence must exceed a class-specific threshold
        3.Top1 - Top2 margin must exceed a class-specific minimum
        4. Examples are then ranked per class and clipped by class-aware quotas (To protect minority class 'maybe')

    """
    if min_confidence_by_class is None:
        min_confidence_by_class = {0: 0.78, 1: 0.62, 2: 0.78}
    if min_margin_by_class is None:
        min_margin_by_class = {0: 0.12, 1: 0.05, 2: 0.12}

    preds_a = np.argmax(probs_teacher_a, axis=1)
    preds_b = np.argmax(probs_teacher_b, axis=1)
    mean_probs = (probs_teacher_a + probs_teacher_b) / 2.0

    candidates_by_class = defaultdict(list)

    for idx in range(len(unlabelled_dataset)):
        if int(preds_a[idx]) != int(preds_b[idx]):
            continue

        pred_class = int(np.argmax(mean_probs[idx]))
        if pred_class != int(preds_a[idx]):
            continue

        sorted_probs = np.sort(mean_probs[idx])[::-1]
        confidence = float(sorted_probs[0])
        margin = float(sorted_probs[0] - sorted_probs[1])

        if confidence < float(min_confidence_by_class[pred_class]):
            continue
        if margin < float(min_margin_by_class[pred_class]):
            continue

        score = _selector_score(confidence, margin)

        candidates_by_class[pred_class].append(
            {
                "idx": idx,
                "pred_class": pred_class,
                "confidence": confidence,
                "margin": margin,
                "score": score,
                "mean_probs": mean_probs[idx].tolist(),
            }
        )

    selected_rows = []
    selection_stats = {}

    for class_id in sorted(id_to_label):
        class_candidates = sorted(
            candidates_by_class[class_id],
            key=lambda row: (row["score"], row["confidence"], row["margin"]),
            reverse=True,
        )

        target_k = class_targets.get(class_id, 0)
        chosen = class_candidates[:target_k]

        selection_stats[id_to_label[class_id]] = {
            "available_candidates": int(len(class_candidates)),
            "selected": int(len(chosen)),
            "target": int(target_k),
            "min_confidence": float(min_confidence_by_class[class_id]),
            "min_margin": float(min_margin_by_class[class_id]),
        }

        for row in chosen:
            raw = unlabelled_dataset[row["idx"]]

            conf_floor = float(min_confidence_by_class[class_id])
            margin_floor = float(min_margin_by_class[class_id])

            conf_strength = (row["confidence"] - conf_floor) / max(1e-8, 1.0 - conf_floor)
            margin_strength = (row["margin"] - margin_floor) / max(1e-8, 1.0 - margin_floor)
            conf_strength = float(np.clip(conf_strength, 0.0, 1.0))
            margin_strength = float(np.clip(margin_strength, 0.0, 1.0))

            weight_strength = (0.5 * conf_strength) + (0.5 * margin_strength)
            example_weight = base_pseudo_weight * (0.75 + 0.25 * weight_strength)

            selected_rows.append(
                {
                    "ID": raw["ID"],
                    "QUESTION": raw["QUESTION"],
                    "CONTEXTS": raw["CONTEXTS"],
                    "final_decision": id_to_label[class_id],
                    "labels": int(class_id),
                    "soft_labels": [float(x) for x in row["mean_probs"]],
                    "example_weights": float(example_weight),
                    "selector_score": float(row["score"]),
                    "pseudo_confidence": float(row["confidence"]),
                    "pseudo_margin": float(row["margin"]),
                    "data_source": "pseudo",
                }
            )

    if len(selected_rows) == 0:
        return _make_empty_pseudo_dataset(), selection_stats

    selected_dataset = Dataset.from_list(selected_rows)
    return selected_dataset, selection_stats


def build_mixed_train_dataset(expert_train_dataset, pseudo_dataset, seed=42):
    """
    Build the 'expert x pseudo-label' mixed dataset.

    Expert PQA-L examples:
        - weight 1.0
        - one-hot soft labels so the mixed stage can use one unified soft-target loss

    Pseudo examples:
        - already contain soft labels and downweighted example_weights
    """
    expert_rows = []
    for row in expert_train_dataset:
        one_hot = [0.0, 0.0, 0.0]
        one_hot[int(row["labels"])] = 1.0

        expert_rows.append(
            {
                "ID": row["ID"],
                "QUESTION": row["QUESTION"],
                "CONTEXTS": row["CONTEXTS"],
                "final_decision": row["final_decision"],
                "labels": int(row["labels"]),
                "soft_labels": one_hot,
                "example_weights": 1.0,
                "selector_score": 1.0,
                "pseudo_confidence": 1.0,
                "pseudo_margin": 1.0,
                "data_source": "expert",
            }
        )

    expert_soft_dataset = Dataset.from_list(expert_rows)

    if len(pseudo_dataset) == 0:
        return expert_soft_dataset.shuffle(seed=seed)

    mixed = concatenate_datasets([expert_soft_dataset, pseudo_dataset]).shuffle(seed=seed)
    return mixed


# fold runners

def run_simple_fold(
    fold,
    stage1_checkpoint_path,
    tokenizer,
    config,
):
    model_seed = int(config.get("base_seed", 42)) + (fold["fold_id"] * 10)

    train_result = run_three_class_training_stage(
        run_name=f"simple_fold_{fold['fold_id']}",
        train_dataset=fold["train_400"],
        selection_dataset=fold["calibration_50"],
        tokenizer=tokenizer,
        init_checkpoint_path=stage1_checkpoint_path,
        seed=model_seed,
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        train_batch_size=config["train_batch_size"],
        eval_batch_size=config["eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        use_weighted_loss=config["use_weighted_loss"],
        use_weighted_sampler=config["use_weighted_sampler"],
        runs_root=config["runs_root"],
        disable_tqdm=config.get("disable_tqdm", False),
        logging_strategy=config.get("logging_strategy", "epoch"),
        logging_steps=config.get("logging_steps", None),
        early_stopping_patience=config.get("early_stopping_patience", 2),
    )

    temperature, _, _ = fit_checkpoint_temperature(
        checkpoint_path=train_result["best_checkpoint"],
        calibration_dataset=fold["calibration_50"],
        tokenizer=tokenizer,
        batch_size=config["eval_batch_size"],
    )

    fold_metrics, fold_preds, _ = evaluate_calibrated_checkpoint(
        checkpoint_path=train_result["best_checkpoint"],
        eval_dataset=fold["eval_50"],
        tokenizer=tokenizer,
        temperature=temperature,
        batch_size=config["eval_batch_size"],
    )

    return {
        "run_name": f"simple_fold_{fold['fold_id']}",
        "fold_id": fold["fold_id"],
        "best_checkpoint": train_result["best_checkpoint"],
        "temperature": float(temperature),
        "selection_metrics": train_result["selection_metrics"],
        "final_metrics": {
            "accuracy": float(fold_metrics["accuracy"]),
            "macro_f1": float(fold_metrics["macro_f1"]),
            "f1_no": float(fold_metrics["f1_no"]),
            "f1_maybe": float(fold_metrics["f1_maybe"]),
            "f1_yes": float(fold_metrics["f1_yes"]),
        },
    }


def run_advanced_fold(
    fold,
    stage1_checkpoint_path,
    unlabelled_dataset,
    tokenizer,
    config,
):
    seed_a = int(config.get("teacher_seed_a", 7)) + (fold["fold_id"] * 10)
    seed_b = int(config.get("teacher_seed_b", 69)) + (fold["fold_id"] * 10)

    teacher_a = run_three_class_training_stage(
        run_name=f"advanced_fold_{fold['fold_id']}_teacher_a",
        train_dataset=fold["train_400"],
        selection_dataset=fold["calibration_50"],
        tokenizer=tokenizer,
        init_checkpoint_path=stage1_checkpoint_path,
        seed=seed_a,
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        train_batch_size=config["train_batch_size"],
        eval_batch_size=config["eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        use_weighted_loss=config["use_weighted_loss"],
        use_weighted_sampler=config["use_weighted_sampler"],
        runs_root=config["runs_root"],
        disable_tqdm=config.get("disable_tqdm", False),
        logging_strategy=config.get("logging_strategy", "epoch"),
        logging_steps=config.get("logging_steps", None),
        early_stopping_patience=config.get("early_stopping_patience", 2),
    )

    teacher_b = run_three_class_training_stage(
        run_name=f"advanced_fold_{fold['fold_id']}_teacher_b",
        train_dataset=fold["train_400"],
        selection_dataset=fold["calibration_50"],
        tokenizer=tokenizer,
        init_checkpoint_path=stage1_checkpoint_path,
        seed=seed_b,
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        train_batch_size=config["train_batch_size"],
        eval_batch_size=config["eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        use_weighted_loss=config["use_weighted_loss"],
        use_weighted_sampler=config["use_weighted_sampler"],
        runs_root=config["runs_root"],
        disable_tqdm=config.get("disable_tqdm", False),
        logging_strategy=config.get("logging_strategy", "epoch"),
        logging_steps=config.get("logging_steps", None),
        early_stopping_patience=config.get("early_stopping_patience", 2),
    )

    temp_a, _, _ = fit_checkpoint_temperature(
        checkpoint_path=teacher_a["best_checkpoint"],
        calibration_dataset=fold["calibration_50"],
        tokenizer=tokenizer,
        batch_size=config["eval_batch_size"],
    )
    temp_b, _, _ = fit_checkpoint_temperature(
        checkpoint_path=teacher_b["best_checkpoint"],
        calibration_dataset=fold["calibration_50"],
        tokenizer=tokenizer,
        batch_size=config["eval_batch_size"],
    )

    probs_a = predict_calibrated_probs_for_dataset(
        checkpoint_path=teacher_a["best_checkpoint"],
        dataset=unlabelled_dataset,
        tokenizer=tokenizer,
        temperature=temp_a,
        batch_size=config["eval_batch_size"],
    )
    probs_b = predict_calibrated_probs_for_dataset(
        checkpoint_path=teacher_b["best_checkpoint"],
        dataset=unlabelled_dataset,
        tokenizer=tokenizer,
        temperature=temp_b,
        batch_size=config["eval_batch_size"],
    )

    class_targets = build_class_target_plan(
        reference_dataset=fold["train_400"],
        pseudo_multiplier=config.get("pseudo_multiplier", 1.50),
        maybe_boost=config.get("maybe_boost", 1.25),
        minimum_per_class=config.get("minimum_pseudo_per_class", 0),
    )

    pseudo_dataset, selection_stats = select_pseudo_labels_best(
        unlabelled_dataset=unlabelled_dataset,
        probs_teacher_a=probs_a,
        probs_teacher_b=probs_b,
        class_targets=class_targets,
        min_confidence_by_class=config.get("min_confidence_by_class", None),
        min_margin_by_class=config.get("min_margin_by_class", None),
        base_pseudo_weight=config.get("base_pseudo_weight", 0.35),
    )

    mixed_train = build_mixed_train_dataset(
        expert_train_dataset=fold["train_400"],
        pseudo_dataset=pseudo_dataset,
        seed=int(config.get("base_seed", 42)) + fold["fold_id"],
    )

    better_teacher = teacher_a
    if teacher_b["selection_metrics"]["macro_f1"] > teacher_a["selection_metrics"]["macro_f1"]:
        better_teacher = teacher_b

    student_seed = int(config.get("student_base_seed", 420)) + (fold["fold_id"] * 10)
    student_result = run_three_class_training_stage(
        run_name=f"advanced_fold_{fold['fold_id']}_student",
        train_dataset=mixed_train,
        selection_dataset=fold["calibration_50"],
        tokenizer=tokenizer,
        init_checkpoint_path=better_teacher["best_checkpoint"],
        seed=student_seed,
        learning_rate=config.get("student_learning_rate", config["learning_rate"]),
        num_train_epochs=config.get("student_num_train_epochs", config["num_train_epochs"]),
        train_batch_size=config["train_batch_size"],
        eval_batch_size=config["eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        use_weighted_loss=False,
        use_weighted_sampler=False,
        runs_root=config["runs_root"],
        disable_tqdm=config.get("disable_tqdm", False),
        logging_strategy=config.get("logging_strategy", "epoch"),
        logging_steps=config.get("logging_steps", None),
        early_stopping_patience=config.get("early_stopping_patience", 2),
    )

    student_temperature, _, _ = fit_checkpoint_temperature(
        checkpoint_path=student_result["best_checkpoint"],
        calibration_dataset=fold["calibration_50"],
        tokenizer=tokenizer,
        batch_size=config["eval_batch_size"],
    )

    fold_metrics, fold_preds, _ = evaluate_calibrated_checkpoint(
        checkpoint_path=student_result["best_checkpoint"],
        eval_dataset=fold["eval_50"],
        tokenizer=tokenizer,
        temperature=student_temperature,
        batch_size=config["eval_batch_size"],
    )

    return {
        "run_name": f"advanced_fold_{fold['fold_id']}",
        "fold_id": fold["fold_id"],
        "teacher_a_checkpoint": teacher_a["best_checkpoint"],
        "teacher_b_checkpoint": teacher_b["best_checkpoint"],
        "student_checkpoint": student_result["best_checkpoint"],
        "teacher_a_temperature": float(temp_a),
        "teacher_b_temperature": float(temp_b),
        "student_temperature": float(student_temperature),
        "pseudo_count": int(len(pseudo_dataset)),
        "class_targets": class_targets,
        "selection_stats": selection_stats,
        "final_metrics": {
            "accuracy": float(fold_metrics["accuracy"]),
            "macro_f1": float(fold_metrics["macro_f1"]),
            "f1_no": float(fold_metrics["f1_no"]),
            "f1_maybe": float(fold_metrics["f1_maybe"]),
            "f1_yes": float(fold_metrics["f1_yes"]),
        },
    }


# pipeline runners

def run_simple_pipeline_cv(
    folds,
    stage1_checkpoint_path,
    tokenizer,
    config,
):
    print_stage_header(
        "Pipeline S",
        (
            "Strong supervised baseline. "
            "For each outer fold we train on 400 expert examples, calibrate on 50 "
            "and report performance on the untouched outer 50."
        ),
    )

    fold_results = []
    for fold in folds:
        result = run_simple_fold(
            fold=fold,
            stage1_checkpoint_path=stage1_checkpoint_path,
            tokenizer=tokenizer,
            config=config,
        )
        fold_results.append(result)
        metrics = result["final_metrics"]
        print(
            f"fold={fold['fold_id']} | macro_f1={metrics['macro_f1']:.4f} "
            f"| acc={metrics['accuracy']:.4f} | f1_no={metrics['f1_no']:.4f} "
            f"| f1_maybe={metrics['f1_maybe']:.4f} | f1_yes={metrics['f1_yes']:.4f}"
        )

    return {
        "pipeline_name": "simple_supervised",
        "fold_results": fold_results,
        "mean_outer_metrics": mean_metrics_across_runs(fold_results),
        "std_outer_metrics": std_metrics_across_runs(fold_results),
        "best_fold": pick_best_run(fold_results, metric_name="macro_f1"),
    }


def run_advanced_pipeline_cv(
    folds,
    stage1_checkpoint_path,
    unlabelled_dataset,
    tokenizer,
    config,
):
    print_stage_header(
        "Pipeline E",
        (
            "Advanced semi-supervised pipeline. "
            "For each outer fold we train two teachers on 400 expert examples, "
            "calibrate them on 50, score the full PQA-U unlabeled pool, select class-aware "
            "soft pseudo-labels, train the model, recalibrate it, and "
            "report performance on the untouched outer 50."
        ),
    )

    fold_results = []
    for fold in folds:
        result = run_advanced_fold(
            fold=fold,
            stage1_checkpoint_path=stage1_checkpoint_path,
            unlabelled_dataset=unlabelled_dataset,
            tokenizer=tokenizer,
            config=config,
        )
        fold_results.append(result)
        metrics = result["final_metrics"]
        print(
            f"fold={fold['fold_id']} | macro_f1={metrics['macro_f1']:.4f} "
            f"| acc={metrics['accuracy']:.4f} | f1_no={metrics['f1_no']:.4f} "
            f"| f1_maybe={metrics['f1_maybe']:.4f} | f1_yes={metrics['f1_yes']:.4f} "
            f"| pseudo={result['pseudo_count']}"
        )

    return {
        "pipeline_name": "advanced_ssl",
        "fold_results": fold_results,
        "mean_outer_metrics": mean_metrics_across_runs(fold_results),
        "std_outer_metrics": std_metrics_across_runs(fold_results),
        "best_fold": pick_best_run(fold_results, metric_name="macro_f1"),
    }


def compare_pipeline_summaries(simple_summary, advanced_summary):
    print()
    print_divider()
    print("PIPELINE COMPARISON ON DEVELOPMENT SIDE")
    print_divider()

    print("simple mean outer metrics")
    for key, value in simple_summary["mean_outer_metrics"].items():
        print(f"  {key}: {value:.4f}")

    print()
    print("advanced mean outer metrics")
    for key, value in advanced_summary["mean_outer_metrics"].items():
        print(f"  {key}: {value:.4f}")

    simple_mean = simple_summary["mean_outer_metrics"]
    advanced_mean = advanced_summary["mean_outer_metrics"]

    if advanced_mean["macro_f1"] > simple_mean["macro_f1"]:
        winner = advanced_summary
        loser = simple_summary
    elif advanced_mean["macro_f1"] < simple_mean["macro_f1"]:
        winner = simple_summary
        loser = advanced_summary
    else:
        if advanced_mean["f1_maybe"] > simple_mean["f1_maybe"]:
            winner = advanced_summary
            loser = simple_summary
        elif advanced_mean["f1_maybe"] < simple_mean["f1_maybe"]:
            winner = simple_summary
            loser = advanced_summary
        else:
            if advanced_mean["accuracy"] >= simple_mean["accuracy"]:
                winner = advanced_summary
                loser = simple_summary
            else:
                winner = simple_summary
                loser = advanced_summary

    print()
    print("Choose winner before evaluating on official test set")
    print(f"  pipeline_name : {winner['pipeline_name']}")
    print(f"  mean_macro_f1 : {winner['mean_outer_metrics']['macro_f1']:.4f}")
    print(f"  mean_f1_maybe : {winner['mean_outer_metrics']['f1_maybe']:.4f}")

    return {"winner": winner, "loser": loser}


# final deployment on official test

def build_test_committee_member_simple(
    member_fold,
    stage1_checkpoint_path,
    tokenizer,
    config,
):
    member_seed = int(config.get("base_seed", 42)) + (member_fold["member_id"] * 10)

    train_result = run_three_class_training_stage(
        run_name=f"final_simple_member_{member_fold['member_id']}",
        train_dataset=member_fold["train_450"],
        selection_dataset=member_fold["calibration_50"],
        tokenizer=tokenizer,
        init_checkpoint_path=stage1_checkpoint_path,
        seed=member_seed,
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        train_batch_size=config["train_batch_size"],
        eval_batch_size=config["eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        use_weighted_loss=config["use_weighted_loss"],
        use_weighted_sampler=config["use_weighted_sampler"],
        runs_root=config["runs_root"],
        disable_tqdm=config.get("disable_tqdm", False),
        logging_strategy=config.get("logging_strategy", "epoch"),
        logging_steps=config.get("logging_steps", None),
        early_stopping_patience=config.get("early_stopping_patience", 2),
    )

    temperature, _, _ = fit_checkpoint_temperature(
        checkpoint_path=train_result["best_checkpoint"],
        calibration_dataset=member_fold["calibration_50"],
        tokenizer=tokenizer,
        batch_size=config["eval_batch_size"],
    )

    return {
        "best_checkpoint": train_result["best_checkpoint"],
        "temperature": float(temperature),
    }


def build_test_committee_member_advanced(
    member_fold,
    stage1_checkpoint_path,
    unlabelled_dataset,
    tokenizer,
    config,
):
    seed_a = int(config.get("teacher_seed_a", 7)) + (member_fold["member_id"] * 10)
    seed_b = int(config.get("teacher_seed_b", 69)) + (member_fold["member_id"] * 10)

    teacher_a = run_three_class_training_stage(
        run_name=f"final_advanced_member_{member_fold['member_id']}_teacher_a",
        train_dataset=member_fold["train_450"],
        selection_dataset=member_fold["calibration_50"],
        tokenizer=tokenizer,
        init_checkpoint_path=stage1_checkpoint_path,
        seed=seed_a,
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        train_batch_size=config["train_batch_size"],
        eval_batch_size=config["eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        use_weighted_loss=config["use_weighted_loss"],
        use_weighted_sampler=config["use_weighted_sampler"],
        runs_root=config["runs_root"],
        disable_tqdm=config.get("disable_tqdm", False),
        logging_strategy=config.get("logging_strategy", "epoch"),
        logging_steps=config.get("logging_steps", None),
        early_stopping_patience=config.get("early_stopping_patience", 2),
    )

    teacher_b = run_three_class_training_stage(
        run_name=f"final_advanced_member_{member_fold['member_id']}_teacher_b",
        train_dataset=member_fold["train_450"],
        selection_dataset=member_fold["calibration_50"],
        tokenizer=tokenizer,
        init_checkpoint_path=stage1_checkpoint_path,
        seed=seed_b,
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        train_batch_size=config["train_batch_size"],
        eval_batch_size=config["eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        use_weighted_loss=config["use_weighted_loss"],
        use_weighted_sampler=config["use_weighted_sampler"],
        runs_root=config["runs_root"],
        disable_tqdm=config.get("disable_tqdm", False),
        logging_strategy=config.get("logging_strategy", "epoch"),
        logging_steps=config.get("logging_steps", None),
        early_stopping_patience=config.get("early_stopping_patience", 2),
    )

    temp_a, _, _ = fit_checkpoint_temperature(
        checkpoint_path=teacher_a["best_checkpoint"],
        calibration_dataset=member_fold["calibration_50"],
        tokenizer=tokenizer,
        batch_size=config["eval_batch_size"],
    )
    temp_b, _, _ = fit_checkpoint_temperature(
        checkpoint_path=teacher_b["best_checkpoint"],
        calibration_dataset=member_fold["calibration_50"],
        tokenizer=tokenizer,
        batch_size=config["eval_batch_size"],
    )

    probs_a = predict_calibrated_probs_for_dataset(
        checkpoint_path=teacher_a["best_checkpoint"],
        dataset=unlabelled_dataset,
        tokenizer=tokenizer,
        temperature=temp_a,
        batch_size=config["eval_batch_size"],
    )
    probs_b = predict_calibrated_probs_for_dataset(
        checkpoint_path=teacher_b["best_checkpoint"],
        dataset=unlabelled_dataset,
        tokenizer=tokenizer,
        temperature=temp_b,
        batch_size=config["eval_batch_size"],
    )

    class_targets = build_class_target_plan(
        reference_dataset=member_fold["train_450"],
        pseudo_multiplier=config.get("pseudo_multiplier", 1.50),
        maybe_boost=config.get("maybe_boost", 1.25),
        minimum_per_class=config.get("minimum_pseudo_per_class", 0),
    )

    pseudo_dataset, selection_stats = select_pseudo_labels_best(
        unlabelled_dataset=unlabelled_dataset,
        probs_teacher_a=probs_a,
        probs_teacher_b=probs_b,
        class_targets=class_targets,
        min_confidence_by_class=config.get("min_confidence_by_class", None),
        min_margin_by_class=config.get("min_margin_by_class", None),
        base_pseudo_weight=config.get("base_pseudo_weight", 0.35),
    )

    mixed_train = build_mixed_train_dataset(
        expert_train_dataset=member_fold["train_450"],
        pseudo_dataset=pseudo_dataset,
        seed=int(config.get("base_seed", 42)) + member_fold["member_id"],
    )

    better_teacher = teacher_a
    if teacher_b["selection_metrics"]["macro_f1"] > teacher_a["selection_metrics"]["macro_f1"]:
        better_teacher = teacher_b

    student_seed = int(config.get("student_base_seed", 420)) + (member_fold["member_id"] * 10)
    student_result = run_three_class_training_stage(
        run_name=f"final_advanced_member_{member_fold['member_id']}_student",
        train_dataset=mixed_train,
        selection_dataset=member_fold["calibration_50"],
        tokenizer=tokenizer,
        init_checkpoint_path=better_teacher["best_checkpoint"],
        seed=student_seed,
        learning_rate=config.get("student_learning_rate", config["learning_rate"]),
        num_train_epochs=config.get("student_num_train_epochs", config["num_train_epochs"]),
        train_batch_size=config["train_batch_size"],
        eval_batch_size=config["eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        use_weighted_loss=False,
        use_weighted_sampler=False,
        runs_root=config["runs_root"],
        disable_tqdm=config.get("disable_tqdm", False),
        logging_strategy=config.get("logging_strategy", "epoch"),
        logging_steps=config.get("logging_steps", None),
        early_stopping_patience=config.get("early_stopping_patience", 2),
    )

    student_temperature, _, _ = fit_checkpoint_temperature(
        checkpoint_path=student_result["best_checkpoint"],
        calibration_dataset=member_fold["calibration_50"],
        tokenizer=tokenizer,
        batch_size=config["eval_batch_size"],
    )

    return {
        "best_checkpoint": student_result["best_checkpoint"],
        "temperature": float(student_temperature),
        "pseudo_count": int(len(pseudo_dataset)),
        "selection_stats": selection_stats,
    }


def run_final_test_committee(
    pipeline_name,
    stage1_checkpoint_path,
    dev_500,
    test_500,
    tokenizer,
    config,
    unlabelled_dataset=None,
):
    print_stage_header(
        "FINAL TEST DEPLOYMENT",
        (
            "Build a 10-member calibrated soft-voting committee on the full 500-development half, "
            "then evaluate the chosen pipeline once on the untouched official 500-test set."
        ),
    )

    committee_folds = build_final_committee_folds(
        dev_dataset=dev_500,
        n_splits=10,
        seed=config.get("cv_seed", 42),
    )

    member_probabilities = []
    member_info = []
    for member_fold in committee_folds:
        if pipeline_name == "simple_supervised":
            member = build_test_committee_member_simple(
                member_fold=member_fold,
                stage1_checkpoint_path=stage1_checkpoint_path,
                tokenizer=tokenizer,
                config=config,
            )
        elif pipeline_name == "advanced_ssl":
            member = build_test_committee_member_advanced(
                member_fold=member_fold,
                stage1_checkpoint_path=stage1_checkpoint_path,
                unlabelled_dataset=unlabelled_dataset,
                tokenizer=tokenizer,
                config=config,
            )
        else:
            raise ValueError(f"Unknown pipeline_name: {pipeline_name}")

        member_probs = predict_calibrated_probs_for_dataset(
            checkpoint_path=member["best_checkpoint"],
            dataset=test_500,
            tokenizer=tokenizer,
            temperature=member["temperature"],
            batch_size=config["eval_batch_size"],
        )

        member_probabilities.append(member_probs)
        member_info.append(member)
        print(f"committee_member={member_fold['member_id']} ready")

    mean_probs = np.mean(np.stack(member_probabilities, axis=0), axis=0)
    labels = np.array(test_500["labels"])

    test_metrics, preds = evaluate_probs(
        probs=mean_probs,
        labels=labels,
        ordered_label_ids=[0, 1, 2],
        id_to_label=id_to_label,
    )

    print()
    print("final official test metrics")
    for key, value in test_metrics.items():
        if key != "confusion_matrix":
            print(f"  {key}: {value:.4f}")

    return {
        "pipeline_name": pipeline_name,
        "committee_members": member_info,
        "test_metrics": test_metrics,
        "test_preds": preds.tolist(),
        "mean_probs": mean_probs.tolist(),
    }