"""
Qwen2.5-7B-Instruct detect+generate baseline (Section 5.3, Table 3, "Qwen
detect+generate" row). For each tweet: prompts Qwen to decide whether there is
an implicit argument component and, if so, generate it; then scores the
generation with ClaimBuster and reports F1 against the gold claim labels
produced by score_reconstructions.py (gold_recon_predictions.csv).
"""
import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
CLAIM_MODEL = "whispAI/ClaimBuster-DeBERTaV2"

SYSTEM_PROMPT = (
    "You are an argument analysis assistant. Given a tweet, decide if it contains "
    "an implicit argument component (an unstated premise or conclusion that the argument "
    "depends on, also called an enthymeme). "
    "If yes, generate the implicit component as a short declarative proposition. "
    "If no, output exactly: no enthymeme"
)

FEW_SHOT = [
    {"tweet": "These are the same people trying to force you to get a vaccine to be a member of society last year. "
              "Once again, I will never support them regardless of what my own beliefs are in regards to abortion.",
     "output": "Forcing people to take vaccines undermines personal freedom and bodily autonomy."},
    {"tweet": "New study indicates natural immunity offers greater protection from COVID-19 than vaccines.",
     "output": "no enthymeme"},
    {"tweet": "If you believe in bodily autonomy but mandate vaccines, you are a hypocrite.",
     "output": "Mandating vaccines violates bodily autonomy."},
    {"tweet": "Inflation is at a 40-year high and gas prices are rising.",
     "output": "no enthymeme"},
]


def build_messages(tweet):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT:
        messages.append({"role": "user", "content": ex["tweet"]})
        messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({"role": "user", "content": tweet})
    return messages


def qwen_generate(tweet, tokenizer, model, device):
    text = tokenizer.apply_chat_template(build_messages(tweet), tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=80, do_sample=False, num_beams=1, pad_token_id=tokenizer.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def predict_claim(texts, tokenizer, model, device, batch_size=16):
    id2label = model.config.id2label
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        for p in logits.argmax(dim=-1).cpu().numpy().tolist():
            label = id2label[p].lower()
            preds.append(0 if ("non" in label or "nfs" in label) else 1)
    return preds


def main():
    parser = argparse.ArgumentParser(description="Qwen detect+generate baseline, scored with ClaimBuster")
    parser.add_argument("--gold-csv", default=str(REPO_ROOT / "results" / "generation_eval" / "test_set_rewritten.csv"))
    parser.add_argument("--gold-recon-preds", default=str(REPO_ROOT / "results" / "claim_detection" / "gold_recon_predictions.csv"),
                         help="output of score_reconstructions.py")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "claim_detection"))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gold_df = pd.read_csv(args.gold_csv)
    gold_recon_df = pd.read_csv(args.gold_recon_preds)

    gold_claim_per_tweet = (
        gold_recon_df.groupby("id")["is_claim"].max().reset_index().rename(columns={"is_claim": "gold_claim"})
    )
    none_ids = gold_df[gold_df["majority_vote"] == "none"]["id"]
    none_rows = pd.DataFrame({"id": none_ids, "gold_claim": 0})
    gold_claim_per_tweet = pd.concat([gold_claim_per_tweet, none_rows], ignore_index=True)
    gold_claim_map = dict(zip(gold_claim_per_tweet["id"], gold_claim_per_tweet["gold_claim"]))

    tweets = gold_df[["id", "tweet_text", "majority_vote"]].copy()

    print(f"Loading {QWEN_MODEL}...")
    qwen_tok = AutoTokenizer.from_pretrained(QWEN_MODEL)
    qwen_model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL, dtype=torch.bfloat16, device_map=str(device))
    qwen_model.eval()

    print(f"\nRunning Qwen on {len(tweets)} tweets...")
    qwen_outputs = [qwen_generate(row["tweet_text"], qwen_tok, qwen_model, device) for _, row in tweets.iterrows()]
    tweets["qwen_output"] = qwen_outputs
    tweets["qwen_has_enthymeme"] = tweets["qwen_output"].apply(lambda x: 0 if "no enthymeme" in x.lower() else 1)

    del qwen_model, qwen_tok
    torch.cuda.empty_cache()
    print("Qwen unloaded.")

    print(f"Loading {CLAIM_MODEL}...")
    claim_tok = AutoTokenizer.from_pretrained(CLAIM_MODEL)
    claim_model = AutoModelForSequenceClassification.from_pretrained(CLAIM_MODEL).to(device)
    claim_model.eval()

    enthymeme_rows = tweets[tweets["qwen_has_enthymeme"] == 1].copy()
    print(f"\nRunning ClaimBuster on {len(enthymeme_rows)} Qwen generations...")
    enthymeme_rows["qwen_is_claim"] = predict_claim(enthymeme_rows["qwen_output"].tolist(), claim_tok, claim_model, device)

    tweets = tweets.merge(enthymeme_rows[["id", "qwen_is_claim"]], on="id", how="left")
    tweets["qwen_is_claim"] = tweets["qwen_is_claim"].fillna(0).astype(int)
    tweets["gold_claim"] = tweets["id"].map(gold_claim_map).fillna(0).astype(int)

    y_true, y_pred = tweets["gold_claim"].tolist(), tweets["qwen_is_claim"].tolist()
    print(f"\nGold: {sum(y_true)} claim / {len(y_true)-sum(y_true)} no-claim")
    print(f"Pred: {sum(y_pred)} claim / {len(y_pred)-sum(y_pred)} no-claim")

    print("\n=== Qwen Pipeline Claim Detection ===")
    print(classification_report(y_true, y_pred, target_names=["no-claim", "claim"], zero_division=0))
    print(f"Precision (claim): {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall    (claim): {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1        (claim): {f1_score(y_true, y_pred, zero_division=0):.4f}")

    tweets.to_csv(os.path.join(args.output_dir, "qwen_pipeline_results.csv"), index=False)
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
