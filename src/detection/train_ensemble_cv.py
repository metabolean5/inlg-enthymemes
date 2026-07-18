"""
5-fold stratified CV training of a 3-seed DeBERTa-v3 ensemble for enthymeme
detection (Section 4.1, Table 1). Each fold trains 3 models (seeds 42/123/456)
with layer-wise learning-rate decay (LLRD) and focal loss on the class
imbalance, averages their softmax probabilities, and tunes the decision
threshold on the fold's held-out split.

Usage:
    # base ensemble
    python train_ensemble_cv.py --checkpoint microsoft/deberta-v3-base --lr 2e-5

    # large ensemble (the deployed configuration, threshold ~0.53)
    python train_ensemble_cv.py \
        --checkpoint MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli \
        --lr 1e-5

Set ENTHYMEME_DATA to point at the full annotated dataset; the small CSV
sample in data/sample/ is only useful for smoke-testing the training loop.
"""
import argparse
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
from scipy.special import softmax
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
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
DATA_PATH = os.environ.get("ENTHYMEME_DATA", str(REPO_ROOT / "data" / "sample" / "enthymeme_annotations_sample.csv"))

SEEDS = [42, 123, 456]
FOLDS = 5
EPOCHS = 20
BATCH_SIZE = 16
GRAD_ACCUM = 2
MAX_LEN = 128
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 5
LLRD_FACTOR = 0.9


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["majority_label"] = df["majority_label"].replace({"premise": "implicit", "conclusion": "implicit"})
    df = df[df["majority_label"].isin(["implicit", "none"])].copy()
    df["label"] = (df["majority_label"] == "implicit").astype(int)
    print(f"Dataset: {len(df)} samples | implicit={df['label'].sum()} none={(df['label']==0).sum()}")
    return df


class InMemoryEarlyStopping(TrainerCallback):
    def __init__(self, patience=5, metric="eval_f1", greater_is_better=True):
        self.patience = patience
        self.metric = metric
        self.greater_is_better = greater_is_better
        self.best_score = None
        self.best_weights = None
        self.wait = 0

    def on_evaluate(self, args, state, control, model, metrics, **kwargs):
        score = metrics.get(self.metric)
        if score is None:
            return
        improved = (
            self.best_score is None
            or (self.greater_is_better and score > self.best_score)
            or (not self.greater_is_better and score < self.best_score)
        )
        if improved:
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


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, labels):
        ce = nn.CrossEntropyLoss(weight=self.weight, reduction="none", label_smoothing=self.label_smoothing)(logits, labels)
        probs = torch.softmax(logits, dim=-1)
        p_t = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * ce).mean()


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
            loss = FocalLoss(weight=weight, gamma=2.0, label_smoothing=0.05)(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


class LLRDTrainer(Trainer):
    """Trainer with layer-wise learning-rate decay: deeper layers get a smaller LR."""

    def __init__(self, llrd_factor=0.9, base_lr=2e-5, **kwargs):
        self.llrd_factor = llrd_factor
        self.base_lr = base_lr
        super().__init__(**kwargs)

    def create_optimizer(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        num_layers = model.config.num_hidden_layers

        groups = []
        emb_lr = self.base_lr * (self.llrd_factor ** num_layers)
        groups += [
            {"params": [p for n, p in model.deberta.embeddings.named_parameters() if not any(nd in n for nd in no_decay)],
             "lr": emb_lr, "weight_decay": 0.01},
            {"params": [p for n, p in model.deberta.embeddings.named_parameters() if any(nd in n for nd in no_decay)],
             "lr": emb_lr, "weight_decay": 0.0},
        ]
        for i, layer in enumerate(model.deberta.encoder.layer):
            layer_lr = self.base_lr * (self.llrd_factor ** (num_layers - i - 1))
            groups += [
                {"params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                 "lr": layer_lr, "weight_decay": 0.01},
                {"params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                 "lr": layer_lr, "weight_decay": 0.0},
            ]
        groups.append({"params": [p for n, p in model.named_parameters() if "deberta" not in n],
                        "lr": self.base_lr, "weight_decay": 0.01})
        self.optimizer = AdamW(groups, lr=self.base_lr)
        return self.optimizer


def make_dataset(df, tokenizer):
    enc = tokenizer(
        df["tweet_text"].tolist(),
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
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
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds, average="macro")}


def run_single(seed, train_df, val_df, tokenizer, class_weights, fold, checkpoint, lr, output_root):
    set_seed(seed)
    config = DebertaV2Config.from_pretrained(checkpoint)
    config.num_labels = 2
    config.cls_dropout = 0.2
    config.class_weights = class_weights

    model = DebertaTextOnly.from_pretrained(checkpoint, config=config, ignore_mismatched_sizes=True)
    train_ds = make_dataset(train_df, tokenizer)
    val_ds = make_dataset(val_df, tokenizer)

    args = TrainingArguments(
        output_dir=os.path.join(output_root, f"fold{fold}_seed{seed}"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        learning_rate=lr,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=seed,
        logging_steps=9999,
        dataloader_num_workers=0,
    )

    early_stop = InMemoryEarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    trainer = LLRDTrainer(
        llrd_factor=LLRD_FACTOR,
        base_lr=lr,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stop],
    )
    trainer.train()
    early_stop.restore_best(model)
    preds_out = trainer.predict(val_ds)
    return preds_out.predictions  # logits


def tune_threshold(logits, labels):
    probs = softmax(logits, axis=-1)[:, 1]
    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.arange(0.25, 0.76, 0.01):
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(labels, preds, average="macro")
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    return best_thresh, best_f1


def main():
    parser = argparse.ArgumentParser(description="3-seed DeBERTa-v3 ensemble, 5-fold CV, with threshold tuning")
    parser.add_argument("--checkpoint", default="microsoft/deberta-v3-base")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "ensemble_cv"))
    args = parser.parse_args()

    df = load_data()
    tokenizer = DebertaV2Tokenizer.from_pretrained(args.checkpoint)

    counts = df["label"].value_counts().sort_index()
    total = len(df)
    class_weights = [total / (2 * counts[i]) for i in range(2)]
    print(f"Class weights: none={class_weights[0]:.3f}, implicit={class_weights[1]:.3f}")
    print(f"Checkpoint={args.checkpoint} | lr={args.lr:.1e} | seeds={SEEDS} | LLRD={LLRD_FACTOR}")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    all_preds, all_labels = [], []
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"]), 1):
        print(f"\nFold {fold}/{FOLDS}  |  train={len(train_idx)}  val={len(val_idx)}")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        labels = val_df["label"].values

        seed_logits = []
        for seed in SEEDS:
            print(f"  Seed {seed}...")
            logits = run_single(seed, train_df, val_df, tokenizer, class_weights, fold, args.checkpoint, args.lr, args.output_dir)
            seed_logits.append(logits)

        avg_logits = np.mean(seed_logits, axis=0)
        thresh, _ = tune_threshold(avg_logits, labels)
        probs = softmax(avg_logits, axis=-1)[:, 1]
        preds = (probs >= thresh).astype(int)

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")
        fold_results.append({"fold": fold, "acc": acc, "f1": f1, "threshold": thresh})
        all_preds.extend(preds)
        all_labels.extend(labels)
        print(f"  Threshold: {thresh:.2f} | Fold {fold} -> Accuracy: {acc:.4f}  Macro F1: {f1:.4f}")

    print(f"\n{'='*60}")
    print("PER-FOLD RESULTS:")
    print(f"{'Fold':<6} {'Accuracy':<12} {'Macro F1':<10} {'Threshold'}")
    print("-" * 40)
    for r in fold_results:
        print(f"{r['fold']:<6} {r['acc']:<12.4f} {r['f1']:<10.4f} {r['threshold']:.2f}")

    accs = [r["acc"] for r in fold_results]
    f1s = [r["f1"] for r in fold_results]
    print(f"\nMean   {np.mean(accs):<12.4f} {np.mean(f1s):.4f}")
    print(f"Std    {np.std(accs):<12.4f} {np.std(f1s):.4f}")
    print("\nAGGREGATED CLASSIFICATION REPORT:")
    print(classification_report(all_labels, all_preds, target_names=["none", "implicit"], digits=4))


if __name__ == "__main__":
    main()
