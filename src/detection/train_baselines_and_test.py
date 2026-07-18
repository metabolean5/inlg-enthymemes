"""
Binary enthymeme detection (none vs. implicit premise/conclusion collapsed):
trains on the full development set and evaluates once on the untouched,
148-tweet held-out test set. Includes two lexical baselines (TF-IDF + linear
SVM, TF-IDF + logistic regression) alongside a single fine-tuned DeBERTa-v3
model, for comparison against the ensembles in train_ensemble_cv.py.

A 10% stratified split is carved out of the training data solely to provide
an early-stopping signal for DeBERTa; it is never used for evaluation.

Usage:
    python train_baselines_and_test.py
"""
import copy
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from datasets import Dataset
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from transformers import (
    DebertaV2Config,
    DebertaV2Model,
    DebertaV2PreTrainedModel,
    DebertaV2Tokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import ContextPooler, StableDropout

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED = 42
EPOCHS = 20
BATCH = 16
GRAD_ACCUM = 2
MAX_LEN = 128
LR = 2e-5
WARMUP = 0.1
PATIENCE = 5
LLRD = 0.9
MODEL = "microsoft/deberta-v3-base"
DATA_PATH = os.environ.get("ENTHYMEME_DATA", str(REPO_ROOT / "data" / "sample" / "enthymeme_annotations_sample.csv"))
TEST_PATH = os.environ.get("ENTHYMEME_TEST_DATA", str(REPO_ROOT / "data" / "sample" / "test_set_sample.csv"))
NAMES = ["none", "implicit"]


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load_train():
    df = pd.read_csv(DATA_PATH)
    df["majority_label"] = df["majority_label"].replace({"premise": "implicit", "conclusion": "implicit"})
    df = df[df["majority_label"].isin(["implicit", "none"])].copy()
    df["label"] = (df["majority_label"] == "implicit").astype(int)
    print(f"Train: {len(df)}  none={(df['label']==0).sum()}  implicit={df['label'].sum()}")
    return df


def load_test():
    df = pd.read_csv(TEST_PATH)
    df["majority_vote"] = df["majority_vote"].replace({"premise": "implicit", "conclusion": "implicit"})
    df = df[df["majority_vote"].isin(["implicit", "none"])].copy()
    df["label"] = (df["majority_vote"] == "implicit").astype(int)
    print(f"Test:  {len(df)}  none={(df['label']==0).sum()}  implicit={df['label'].sum()}")
    return df


# ── Lexical baselines ─────────────────────────────────────────────────────────

def make_pipeline(name):
    tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=50000)
    if name == "svm":
        clf = CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=2000, C=1.0))
    else:
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, solver="lbfgs")
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def run_lexical(train_df, test_df, name):
    pipe = make_pipeline(name)
    pipe.fit(train_df["tweet_text"].values, train_df["label"].values)
    preds = pipe.predict(test_df["tweet_text"].values)
    labels = test_df["label"].values
    return labels, preds


# ── DeBERTa ───────────────────────────────────────────────────────────────────

class InMemoryEarlyStopping(TrainerCallback):
    def __init__(self, patience=5):
        self.patience = patience
        self.best_score = None
        self.best_weights = None
        self.wait = 0

    def on_evaluate(self, args, state, control, model, metrics, **kwargs):
        score = metrics.get("eval_f1")
        if score is None:
            return
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_weights = copy.deepcopy(model.state_dict())
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                control.should_training_stop = True

    def restore_best(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class DebertaTextOnly(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        self.classifier = nn.Linear(self.pooler.output_dim, 2)
        self.dropout = StableDropout(getattr(config, "cls_dropout", config.hidden_dropout_prob))
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.deberta(input_ids, attention_mask=attention_mask)
        pooled = self.dropout(self.pooler(out[0]))
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            weight = torch.tensor(self.config.class_weights, dtype=torch.float, device=logits.device)
            loss = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.1)(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


class LLRDTrainer(Trainer):
    def __init__(self, llrd_factor=0.9, base_lr=2e-5, **kwargs):
        self.llrd_factor = llrd_factor
        self.base_lr = base_lr
        super().__init__(**kwargs)

    def create_optimizer(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        n = model.config.num_hidden_layers
        groups = []
        emb_lr = self.base_lr * (self.llrd_factor ** n)
        groups += [
            {"params": [p for nm, p in model.deberta.embeddings.named_parameters() if not any(nd in nm for nd in no_decay)],
             "lr": emb_lr, "weight_decay": 0.01},
            {"params": [p for nm, p in model.deberta.embeddings.named_parameters() if any(nd in nm for nd in no_decay)],
             "lr": emb_lr, "weight_decay": 0.0},
        ]
        for i, layer in enumerate(model.deberta.encoder.layer):
            lr_i = self.base_lr * (self.llrd_factor ** (n - i - 1))
            groups += [
                {"params": [p for nm, p in layer.named_parameters() if not any(nd in nm for nd in no_decay)],
                 "lr": lr_i, "weight_decay": 0.01},
                {"params": [p for nm, p in layer.named_parameters() if any(nd in nm for nd in no_decay)],
                 "lr": lr_i, "weight_decay": 0.0},
            ]
        groups.append({"params": [p for nm, p in model.named_parameters() if "deberta" not in nm],
                        "lr": self.base_lr, "weight_decay": 0.01})
        self.optimizer = AdamW(groups, lr=self.base_lr)
        return self.optimizer


def make_dataset(df, tokenizer):
    enc = tokenizer(df["tweet_text"].tolist(), max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="np")
    ds = Dataset.from_dict({
        "input_ids": enc["input_ids"].tolist(),
        "attention_mask": enc["attention_mask"].tolist(),
        "labels": df["label"].values.astype(np.int64).tolist(),
    })
    ds.set_format("torch")
    return ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds, average="macro", zero_division=0)}


def run_deberta(train_df, test_df, tokenizer, class_weights):
    set_seed(SEED)
    tr_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df["label"], random_state=SEED)
    tr_df = tr_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"  DeBERTa train={len(tr_df)}  val(early-stop)={len(val_df)}")

    config = DebertaV2Config.from_pretrained(MODEL)
    config.num_labels = 2
    config.cls_dropout = 0.2
    config.class_weights = class_weights

    model = DebertaTextOnly.from_pretrained(MODEL, config=config, ignore_mismatched_sizes=True)
    train_ds = make_dataset(tr_df, tokenizer)
    val_ds = make_dataset(val_df, tokenizer)
    test_ds = make_dataset(test_df, tokenizer)

    args = TrainingArguments(
        output_dir=str(REPO_ROOT / "results" / "baseline_test"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        learning_rate=LR,
        warmup_ratio=WARMUP,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
        logging_steps=9999,
        dataloader_num_workers=0,
    )

    early_stop = InMemoryEarlyStopping(patience=PATIENCE)
    trainer = LLRDTrainer(
        llrd_factor=LLRD, base_lr=LR,
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=tokenizer, compute_metrics=compute_metrics,
        callbacks=[early_stop],
    )
    trainer.train()
    early_stop.restore_best(model)
    print(f"  Early-stop best val Macro F1: {early_stop.best_score:.4f}")

    preds_out = trainer.predict(test_ds)
    preds = np.argmax(preds_out.predictions, axis=-1)
    labels = test_df["label"].values
    return labels, preds


# ── Reporting ─────────────────────────────────────────────────────────────────

def metrics(labels, preds):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, labels=[0, 1], zero_division=0)
    return acc, f1, p, r, f


def print_table(results):
    sep = "=" * 108
    print(f"\n{sep}")
    print("  TASK 1A — BINARY RESULTS  (none vs implicit)  |  held-out test set")
    print(sep)
    hdr = (f"  {'Model':<26} {'Acc':>6} {'Mac-F1':>8} "
           f"{'None-P':>8} {'None-R':>8} {'None-F1':>8} "
           f"{'Impl-P':>8} {'Impl-R':>8} {'Impl-F1':>8}")
    print(hdr)
    print("  " + "-" * 104)
    for name, (acc, f1, p, r, f) in results.items():
        print(f"  {name:<26} {acc:>6.4f} {f1:>8.4f} "
              f"{p[0]:>8.4f} {r[0]:>8.4f} {f[0]:>8.4f} "
              f"{p[1]:>8.4f} {r[1]:>8.4f} {f[1]:>8.4f}")
    print(sep)


def main():
    set_seed(SEED)
    train_df = load_train()
    test_df = load_test()
    results = {}

    print("\n── TF-IDF + LinearSVC ──")
    labels, preds = run_lexical(train_df, test_df, "svm")
    print(classification_report(labels, preds, target_names=NAMES, digits=4, zero_division=0))
    results["TF-IDF + LinearSVC"] = metrics(labels, preds)

    print("── TF-IDF + LogisticReg ──")
    labels, preds = run_lexical(train_df, test_df, "logreg")
    print(classification_report(labels, preds, target_names=NAMES, digits=4, zero_division=0))
    results["TF-IDF + LogisticReg"] = metrics(labels, preds)

    print("── DeBERTa-v3-base ──")
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL)
    counts = train_df["label"].value_counts().sort_index()
    cw = [len(train_df) / (2 * counts[i]) for i in range(2)]
    print(f"  Class weights: none={cw[0]:.3f}  implicit={cw[1]:.3f}")
    labels, preds = run_deberta(train_df, test_df, tokenizer, cw)
    print(classification_report(labels, preds, target_names=NAMES, digits=4, zero_division=0))
    results["DeBERTa-v3-base"] = metrics(labels, preds)

    print_table(results)


if __name__ == "__main__":
    main()
