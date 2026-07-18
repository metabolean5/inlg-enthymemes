import os
import csv
import numpy as np
from sklearn.model_selection import StratifiedKFold
from datasets import concatenate_datasets


def build_cv_dataset(tokenized_datasets):
    """Combine the train + validation splits into a single dataset for cross-validation."""
    return concatenate_datasets([
        tokenized_datasets["train"],
        tokenized_datasets["validation"],
    ])


def create_stratified_folds(full_dataset, n_splits=5, seed=42):
    labels = np.array(full_dataset["labels"])
    indices = np.arange(len(full_dataset))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    for train_idx, val_idx in skf.split(indices, labels):
        folds.append((train_idx.tolist(), val_idx.tolist()))

    return folds


def initialize_cv_trial_results_csv(cv_trial_results_csv, n_splits=5):
    if not os.path.exists(cv_trial_results_csv):
        with open(cv_trial_results_csv, mode="w", newline="") as file:
            writer = csv.writer(file)

            header = ["trial_number", "mean_f1_cv", "std_f1_cv"]
            for i in range(1, n_splits + 1):
                header.append(f"fold_{i}_f1")
            header.extend([
                "learning_rate",
                "weight_decay",
                "warmup_steps",
                "num_train_epochs",
                "batch_size",
            ])
            writer.writerow(header)


def save_cv_trial_result_to_csv(
    cv_trial_results_csv,
    trial_number,
    mean_f1_cv,
    std_f1_cv,
    fold_f1_scores,
    learning_rate,
    weight_decay,
    warmup_steps,
    num_train_epochs,
    batch_size,
):
    with open(cv_trial_results_csv, mode="a", newline="") as file:
        writer = csv.writer(file)
        row = [trial_number, mean_f1_cv, std_f1_cv]
        row.extend(fold_f1_scores)
        row.extend([learning_rate, weight_decay, warmup_steps, num_train_epochs, batch_size])
        writer.writerow(row)
