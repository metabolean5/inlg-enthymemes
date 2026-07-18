"""
Check-worthiness scoring with ClaimBuster (Section 5.3, Table 3, "Full pipeline" row).

1. Runs ClaimBuster on every annotator reconstruction in the gold (rewritten)
   test set -> derives a gold claim/no-claim label per tweet (claim if ANY
   reconstruction is check-worthy).
2. Runs ClaimBuster on the (rewritten) BART pipeline generations.
3. Merges on tweet id (tweets the pipeline never generated for default to
   no-claim) and reports precision/recall/F1 for the claim class.

Usage:
    python score_reconstructions.py \
        --gold-csv results/generation_eval/test_set_rewritten.csv \
        --bart-csv results/generation_eval/roscoe_bart_rewritten.csv
"""
import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "whispAI/ClaimBuster-DeBERTaV2"
BATCH_SIZE = 16
RECON_COLS = [f"recon{i}_rewritten" for i in range(1, 6)]


def load_model():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Device: {device}")
    print(f"Label map: {model.config.id2label}")
    return tokenizer, model, device


def predict_all(texts, tokenizer, model, device):
    all_preds = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    return all_preds


def label_to_binary(label_id, id2label):
    label_str = id2label[label_id].lower()
    # NFS (non-factual statement) -> not a claim; UFS/CFS -> claim
    return 0 if ("non" in label_str or "nfs" in label_str) else 1


def main():
    parser = argparse.ArgumentParser(description="ClaimBuster check-worthiness scoring for gold vs. system reconstructions")
    parser.add_argument("--gold-csv", default=str(REPO_ROOT / "results" / "generation_eval" / "test_set_rewritten.csv"))
    parser.add_argument("--bart-csv", default=str(REPO_ROOT / "results" / "generation_eval" / "roscoe_bart_rewritten.csv"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "claim_detection"))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer, model, device = load_model()
    id2label = model.config.id2label

    gold_df = pd.read_csv(args.gold_csv)
    bart_df = pd.read_csv(args.bart_csv)
    print(f"\nGold rows: {len(gold_df)}, BART rows: {len(bart_df)}")

    # --- Gold claim labels on ALL tweets ---
    print("\nRunning ClaimBuster on ALL gold rewritten reconstructions...")
    recon_records = []
    for _, row in gold_df.iterrows():
        for col in RECON_COLS:
            text = row.get(col)
            if isinstance(text, str) and text.strip():
                recon_records.append({"id": row["id"], "col": col, "text": text.strip()})

    recon_texts = [r["text"] for r in recon_records]
    print(f"  Total recon texts to classify: {len(recon_texts)}")
    recon_preds = predict_all(recon_texts, tokenizer, model, device)
    for rec, pred in zip(recon_records, recon_preds):
        rec["pred_class"] = pred
        rec["pred_label"] = id2label[pred]
        rec["is_claim"] = label_to_binary(pred, id2label)
    recon_df = pd.DataFrame(recon_records)

    gold_claim_per_tweet = (
        recon_df.groupby("id")["is_claim"].max().reset_index().rename(columns={"is_claim": "gold_claim"})
    )
    none_ids = gold_df[gold_df["majority_vote"] == "none"]["id"]
    none_rows = pd.DataFrame({"id": none_ids, "gold_claim": 0})
    gold_claim_per_tweet = pd.concat([gold_claim_per_tweet, none_rows], ignore_index=True)

    n_gold_claims = int(gold_claim_per_tweet["gold_claim"].sum())
    print(f"\n  -> {n_gold_claims} tweets with claim / {len(gold_claim_per_tweet)} total")

    # --- System (BART) predictions ---
    print("\nRunning ClaimBuster on BART rewritten generations...")
    bart_texts = bart_df["bart_rewritten"].tolist()
    bart_preds_raw = predict_all(bart_texts, tokenizer, model, device)
    bart_df = bart_df.copy()
    bart_df["bart_pred_class"] = bart_preds_raw
    bart_df["bart_pred_label"] = [id2label[p] for p in bart_preds_raw]
    bart_df["bart_is_claim"] = [label_to_binary(p, id2label) for p in bart_preds_raw]

    # --- Merge and score over ALL tweets ---
    all_ids = pd.DataFrame({"id": gold_df["id"]})
    bart_preds = bart_df[["id", "bart_is_claim", "bart_pred_label", "bart_rewritten", "implicit_content"]].copy()
    merged = all_ids.merge(bart_preds, on="id", how="left")
    merged["bart_is_claim"] = merged["bart_is_claim"].fillna(0).astype(int)
    merged["bart_pred_label"] = merged["bart_pred_label"].fillna("no prediction")
    merged = merged.merge(gold_claim_per_tweet, on="id", how="left")
    merged["gold_claim"] = merged["gold_claim"].fillna(0).astype(int)

    y_true, y_pred = merged["gold_claim"].tolist(), merged["bart_is_claim"].tolist()
    print(f"\nEvaluating on all {len(merged)} tweets (unmatched BART rows -> pred=0)")
    print(f"Gold: {merged['gold_claim'].sum()} claim / {(merged['gold_claim']==0).sum()} no-claim")
    print(f"BART: {merged['bart_is_claim'].sum()} claim / {(merged['bart_is_claim']==0).sum()} no-claim")

    print("\n=== Claim Detection Results ===")
    print(classification_report(y_true, y_pred, target_names=["no-claim", "claim"], zero_division=0))
    print(f"Precision (claim): {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall    (claim): {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1        (claim): {f1_score(y_true, y_pred, zero_division=0):.4f}")

    detected = int(merged[(merged["gold_claim"] == 1) & (merged["bart_is_claim"] == 1)].shape[0])
    print(f"\nOf {n_gold_claims} gold claims: {detected} detected by BART, {n_gold_claims - detected} missed")

    out = merged[["id", "bart_rewritten", "bart_pred_label", "bart_is_claim", "gold_claim", "implicit_content"]].copy()
    out.to_csv(os.path.join(args.output_dir, "claim_detection_results.csv"), index=False)
    recon_df.to_csv(os.path.join(args.output_dir, "gold_recon_predictions.csv"), index=False)
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
