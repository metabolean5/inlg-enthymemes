"""
Runs the enthymeme detection + generation pipeline over an entire test-set CSV
and writes per-tweet predictions (probabilities, predicted role, generated
implicit content) to disk. This is what produced inlg/test_results.csv, the
basis for the downstream claim-detection evaluation in claim_detection/.

Usage:
    python run_test_set.py --models-root /path/to/models \
        --test-csv data/sample/test_set_sample.csv \
        --output results/test_results.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
)

from pipeline import (
    BART_DIR_NAME,
    BART_MAX_NEW_TOKENS,
    BART_MAX_SOURCE,
    DEBERTA_MAX_LEN,
    DETECTOR_SEED_DIRS,
    DETECTOR_THRESHOLD,
    LABEL_CLF_SEED_DIRS,
    LABEL_MAP,
    SENTINEL_TOKEN,
)


def run_one_model_all_examples(model_dir, texts, max_len, device):
    """Load one classifier, run all texts, unload. Returns (N, num_classes) numpy array."""
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    all_probs = []
    with torch.no_grad():
        for text in texts:
            enc = tok(text, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt").to(device)
            logits = mdl(**enc).logits
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy()[0])
    del mdl
    torch.cuda.empty_cache()
    return np.array(all_probs)


def ensemble_all(dirs, texts, max_len, device, label=""):
    seed_probs = []
    for i, d in enumerate(dirs):
        print(f"  [{label} seed {i+1}/{len(dirs)}] {Path(d).name}")
        seed_probs.append(run_one_model_all_examples(d, texts, max_len, device))
    return np.mean(seed_probs, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Run the full pipeline over a test-set CSV")
    parser.add_argument("--models-root", required=True)
    parser.add_argument("--test-csv", required=True, help="CSV with at least 'id' and 'tweet_text' columns")
    parser.add_argument("--output", default="test_results.csv")
    parser.add_argument("--cache", default=None, help="Optional .npz path to cache classifier probabilities across runs")
    args = parser.parse_args()

    models_root = Path(args.models_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading test set...")
    df = pd.read_csv(args.test_csv)
    texts = df["tweet_text"].tolist()
    ids = df["id"].tolist()
    print(f"{len(df)} examples\n")

    detector_dirs = [models_root / d for d in DETECTOR_SEED_DIRS]
    label_clf_dirs = [models_root / d for d in LABEL_CLF_SEED_DIRS]

    cache_path = Path(args.cache) if args.cache else None
    if cache_path and cache_path.exists():
        print(f"Loading cached probabilities from {cache_path}")
        cache = np.load(cache_path)
        det_probs, clf_probs = cache["det_probs"], cache["clf_probs"]
    else:
        print("Running detector ensemble (one model at a time)...")
        det_probs = ensemble_all(detector_dirs, texts, DEBERTA_MAX_LEN, device, label="detector")
        print("\nRunning label classifier ensemble (one model at a time)...")
        clf_probs = ensemble_all(label_clf_dirs, texts, DEBERTA_MAX_LEN, device, label="label_clf")
        if cache_path:
            np.savez(cache_path, det_probs=det_probs, clf_probs=clf_probs)
            print(f"Probabilities cached to {cache_path}")

    print("\nLoading BART generator...")
    bart_tok = BartTokenizer.from_pretrained(models_root / BART_DIR_NAME)
    bart_mdl = BartForConditionalGeneration.from_pretrained(models_root / BART_DIR_NAME).to(device).eval()

    rows = []
    print("\nGenerating implicit content for detected enthymemes...")
    for i, (eid, text) in enumerate(zip(ids, texts)):
        prob_enthymeme = float(det_probs[i, 1])
        is_enthymeme = prob_enthymeme >= DETECTOR_THRESHOLD
        pred_label_id = int(np.argmax(clf_probs[i]))
        pred_label = LABEL_MAP[pred_label_id]

        implicit_content = ""
        if is_enthymeme:
            input_text = f"task: generate_implicit | label: {pred_label} | tweet: {text}"
            enc = bart_tok(input_text, max_length=BART_MAX_SOURCE, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                out_ids = bart_mdl.generate(**enc, num_beams=1, max_new_tokens=BART_MAX_NEW_TOKENS)
            generation = bart_tok.decode(out_ids[0], skip_special_tokens=True)
            if generation.strip() != SENTINEL_TOKEN:
                implicit_content = generation

        rows.append({
            "id": eid,
            "sentence": text,
            "prob_enthymeme": round(prob_enthymeme, 4),
            "is_enthymeme": is_enthymeme,
            "implicit_label": pred_label if is_enthymeme else "",
            "prob_none": round(float(clf_probs[i, 0]), 4),
            "prob_premise": round(float(clf_probs[i, 1]), 4),
            "prob_conclusion": round(float(clf_probs[i, 2]), 4),
            "implicit_content": implicit_content,
        })

        if is_enthymeme:
            preview = implicit_content[:70] + "..." if len(implicit_content) > 70 else implicit_content or "[sentinel]"
            print(f"  [{i+1}/{len(df)}] id={eid}  label={pred_label}  gen={preview}")

    fieldnames = ["id", "sentence", "prob_enthymeme", "is_enthymeme", "implicit_label",
                  "prob_none", "prob_premise", "prob_conclusion", "implicit_content"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_enthymemes = sum(r["is_enthymeme"] for r in rows)
    n_generated = sum(1 for r in rows if r["implicit_content"])
    print(f"\nDone. {n_enthymemes}/{len(rows)} enthymemes detected, {n_generated} implicit contents generated.")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
