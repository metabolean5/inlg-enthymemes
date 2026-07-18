import os
import gc
import csv
import torch
import evaluate
from optuna.importance import (
    get_param_importances,
    FanovaImportanceEvaluator,
    PedAnovaImportanceEvaluator,
)

accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    import numpy as np
    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(predictions=preds, references=labels)
    precision = precision_metric.compute(predictions=preds, references=labels, average="binary")
    recall = recall_metric.compute(predictions=preds, references=labels, average="binary")
    f1 = f1_metric.compute(predictions=preds, references=labels, average="binary")

    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"],
    }


def cleanup_cuda(*objs):
    """Frees as much GPU memory as possible between consecutive Optuna trials/folds."""
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def initialize_trial_results_csv(trial_results_csv):
    if not os.path.exists(trial_results_csv):
        with open(trial_results_csv, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "trial_number",
                "f1_score",
                "learning_rate",
                "weight_decay",
                "warmup_steps",
                "num_train_epochs",
                "batch_size",
            ])


def save_trial_result_to_csv(
    trial_results_csv,
    trial_number,
    f1_score,
    learning_rate,
    weight_decay,
    warmup_steps,
    num_train_epochs,
    batch_size,
):
    with open(trial_results_csv, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            trial_number,
            f1_score,
            learning_rate,
            weight_decay,
            warmup_steps,
            num_train_epochs,
            batch_size,
        ])


def save_importances_to_csv(study, importance_results_csv):
    fanova_importances = get_param_importances(
        study, evaluator=FanovaImportanceEvaluator(seed=42)
    )
    pedanova_importances = get_param_importances(
        study, evaluator=PedAnovaImportanceEvaluator(target_quantile=0.2)
    )

    all_params = sorted(set(fanova_importances.keys()) | set(pedanova_importances.keys()))

    with open(importance_results_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["hyperparameter", "fanova_importance", "pedanova_importance"])
        for param in all_params:
            writer.writerow([
                param,
                fanova_importances.get(param, 0.0),
                pedanova_importances.get(param, 0.0),
            ])

    print("\n=== fANOVA parameter importances ===")
    for key, value in fanova_importances.items():
        print(f"{key}: {value}")

    print("\n=== PED-ANOVA parameter importances ===")
    for key, value in pedanova_importances.items():
        print(f"{key}: {value}")


def trial_report_callback(study, trial):
    print(f"Trial {trial.number} finished with value: {trial.value} and parameters:")
    for key, value in trial.params.items():
        print(f"    '{key}': {value},")
    print()
    print(f"Best is trial {study.best_trial.number} with value: {study.best_value}.")
    print("-" * 60)
