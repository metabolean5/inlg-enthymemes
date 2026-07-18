"""
Optuna hyperparameter optimization for the enthymeme detector, with 5-fold
stratified cross-validation inside each trial (Section 4.1.1 / Appendix A).

Tunes: learning rate, weight decay, warmup ratio, batch size, and number of
epochs, maximizing the mean macro-F1 across folds. The trial-level results
and fANOVA / PED-ANOVA hyperparameter importances are logged to CSV.

Usage:
    python hpo_cv.py --checkpoint microsoft/deberta-v3-base --n-trials 30

By default this points at the small CSV sample shipped in data/sample/, which
is only enough data to smoke-test the pipeline; set ENTHYMEME_DATA to your
own copy of the full annotated dataset to reproduce the paper's numbers.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import optuna
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset

from utils.hpo_helpers import cleanup_cuda, save_importances_to_csv, trial_report_callback
from utils.cv_helpers import create_stratified_folds, initialize_cv_trial_results_csv, save_cv_trial_result_to_csv

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = os.environ.get("ENTHYMEME_DATA", str(REPO_ROOT / "data" / "sample" / "enthymeme_annotations_sample.csv"))
MAX_LEN = 80


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["majority_label"] = df["majority_label"].replace({"premise": "implicit", "conclusion": "implicit"})
    df = df[df["majority_label"].isin(["implicit", "none"])].copy()
    df["label"] = (df["majority_label"] == "implicit").astype(int)
    print(f"Dataset: {len(df)} samples | implicit={df['label'].sum()} none={(df['label']==0).sum()}")
    return df


def make_dataset(df, tokenizer, max_len):
    enc = tokenizer(
        df["tweet_text"].tolist(),
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="np",
    )
    ds = Dataset.from_dict({
        "input_ids": enc["input_ids"].tolist(),
        "attention_mask": enc["attention_mask"].tolist(),
        "labels": df["label"].values.astype(np.int64).tolist(),
    })
    ds.set_format("torch")
    return ds


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
    }


def main():
    parser = argparse.ArgumentParser(description="Optuna HPO with 5-fold CV for enthymeme detection")
    parser.add_argument("--checkpoint", default="microsoft/deberta-v3-base")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "hpo"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cv_trial_results_csv = os.path.join(args.output_dir, "cv_trial_results.csv")
    importance_results_csv = os.path.join(args.output_dir, "cv_param_importances.csv")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    df = load_data()

    def tokenize(batch_df):
        return make_dataset(batch_df, tokenizer, MAX_LEN)

    full_dataset = tokenize(df)
    folds = create_stratified_folds(full_dataset, n_splits=args.n_splits, seed=42)
    initialize_cv_trial_results_csv(cv_trial_results_csv, n_splits=args.n_splits)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def objective(trial):
        cleanup_cuda()

        per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16])
        num_train_epochs = trial.suggest_int("num_train_epochs", 3, 8)
        learning_rate = trial.suggest_float("learning_rate", 8e-6, 5e-5, log=True)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.12)
        warmup_ratio = trial.suggest_float("warmup_steps", 0.0, 0.15)

        fold_f1_scores = []
        try:
            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                cleanup_cuda()
                model = None
                trainer = None
                try:
                    train_dataset = full_dataset.select(train_idx)
                    val_dataset = full_dataset.select(val_idx)

                    model = AutoModelForSequenceClassification.from_pretrained(
                        args.checkpoint,
                        num_labels=2,
                        id2label={0: "none", 1: "enthymeme"},
                        label2id={"none": 0, "enthymeme": 1},
                    )

                    training_args = TrainingArguments(
                        output_dir=os.path.join(args.output_dir, "tmp", f"trial_{trial.number}_fold_{fold_idx}"),
                        per_device_train_batch_size=per_device_train_batch_size,
                        per_device_eval_batch_size=8,
                        eval_strategy="epoch",
                        save_strategy="no",
                        num_train_epochs=num_train_epochs,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        warmup_ratio=warmup_ratio,
                        logging_strategy="epoch",
                        seed=42,
                        data_seed=42,
                        gradient_checkpointing=True,
                        report_to="none",
                        disable_tqdm=True,
                    )

                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=val_dataset,
                        data_collator=data_collator,
                        compute_metrics=compute_metrics,
                    )
                    trainer.train()

                    eval_f1_values = [e["eval_f1"] for e in trainer.state.log_history if "eval_f1" in e]
                    fold_f1_scores.append(max(eval_f1_values) if eval_f1_values else float("-inf"))
                finally:
                    if trainer is not None:
                        try:
                            del trainer.model
                        except Exception:
                            pass
                    cleanup_cuda(trainer, model)

            mean_f1_cv = float(np.mean(fold_f1_scores))
            std_f1_cv = float(np.std(fold_f1_scores))

            save_cv_trial_result_to_csv(
                cv_trial_results_csv, trial.number, mean_f1_cv, std_f1_cv, fold_f1_scores,
                learning_rate, weight_decay, warmup_ratio, num_train_epochs, per_device_train_batch_size,
            )
            return mean_f1_cv

        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            while len(fold_f1_scores) < args.n_splits:
                fold_f1_scores.append(float("-inf"))
            save_cv_trial_result_to_csv(
                cv_trial_results_csv, trial.number, float("-inf"), float("nan"), fold_f1_scores,
                learning_rate, weight_decay, warmup_ratio, num_train_epochs, per_device_train_batch_size,
            )
            return float("-inf")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials, callbacks=[trial_report_callback])

    print("Best trial:")
    print(study.best_trial)
    print("Best params:")
    print(study.best_trial.params)

    save_importances_to_csv(study, importance_results_csv)


if __name__ == "__main__":
    main()
