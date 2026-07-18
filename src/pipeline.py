"""
End-to-end inference pipeline: enthymeme detection -> implicit-role
classification -> BART generation (Section 4, Figure 1).

Detection:  DeBERTa-v3-large ensemble (3 seeds), threshold-tuned
Label clf:  DeBERTa-v3-large label-classifier ensemble (3 seeds), predicts
            premise / conclusion / none
Generation: BART-base with a <NO_IMPLICIT> sentinel token

Requires fine-tuned model checkpoints, which are not distributed in this
repository (see README "Models" section) — point --models-root at your own
checkpoints, laid out as:
    <models-root>/<detector-seed-dirs>/...
    <models-root>/<label-clf-seed-dirs>/...
    <models-root>/<bart-dir>/...

Usage:
    python pipeline.py --text "Your sentence here." --models-root /path/to/models
    python pipeline.py --file sentences.txt --models-root /path/to/models
    python pipeline.py --interactive --models-root /path/to/models
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
)

DETECTOR_SEED_DIRS = ["final_deberta_large_seed42", "final_deberta_large_seed123", "final_deberta_large_seed456"]
DETECTOR_THRESHOLD = 0.53

LABEL_CLF_SEED_DIRS = [
    "final_deberta_large_label_classifier_seed42",
    "final_deberta_large_label_classifier_seed123",
    "final_deberta_large_label_classifier_seed456",
]
LABEL_MAP = {0: "none", 1: "premise", 2: "conclusion"}

BART_DIR_NAME = "final_bart_base_with_sentinel_seed42"
SENTINEL_TOKEN = "<NO_IMPLICIT>"
BART_MAX_SOURCE = 128
BART_MAX_NEW_TOKENS = 48
DEBERTA_MAX_LEN = 128


def load_detector(models_root, device):
    tokenizers, models = [], []
    for name in DETECTOR_SEED_DIRS:
        d = models_root / name
        tokenizers.append(AutoTokenizer.from_pretrained(d))
        models.append(AutoModelForSequenceClassification.from_pretrained(d).to(device).eval())
    return tokenizers, models


def load_label_clf(models_root, device):
    tokenizers, models = [], []
    for name in LABEL_CLF_SEED_DIRS:
        d = models_root / name
        tokenizers.append(AutoTokenizer.from_pretrained(d))
        models.append(AutoModelForSequenceClassification.from_pretrained(d).to(device).eval())
    return tokenizers, models


def load_generator(models_root, device):
    d = models_root / BART_DIR_NAME
    tok = BartTokenizer.from_pretrained(d)
    mdl = BartForConditionalGeneration.from_pretrained(d).to(device).eval()
    return tok, mdl


def predict_proba_ensemble(text, tokenizers, models, max_len, device):
    """Average softmax probabilities across ensemble members."""
    probs_list = []
    with torch.no_grad():
        for tok, mdl in zip(tokenizers, models):
            enc = tok(text, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt").to(device)
            logits = mdl(**enc).logits
            probs_list.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.mean(probs_list, axis=0)[0]


def detect_enthymeme(text, tokenizers, models, device):
    probs = predict_proba_ensemble(text, tokenizers, models, DEBERTA_MAX_LEN, device)
    enthymeme_prob = float(probs[1])
    return enthymeme_prob >= DETECTOR_THRESHOLD, enthymeme_prob


def classify_label(text, tokenizers, models, device):
    probs = predict_proba_ensemble(text, tokenizers, models, DEBERTA_MAX_LEN, device)
    pred_id = int(np.argmax(probs))
    return LABEL_MAP[pred_id], {LABEL_MAP[i]: float(probs[i]) for i in range(len(probs))}


def generate_implicit(text, label, tok, mdl, device):
    input_text = f"task: generate_implicit | label: {label} | tweet: {text}"
    enc = tok(input_text, max_length=BART_MAX_SOURCE, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = mdl.generate(**enc, num_beams=1, max_new_tokens=BART_MAX_NEW_TOKENS)
    return tok.decode(output_ids[0], skip_special_tokens=True)


def run_pipeline(text, detector_toks, detector_mdls, clf_toks, clf_mdls, bart_tok, bart_mdl, device):
    text = text.strip()
    if not text:
        return None

    is_enthymeme, enthymeme_prob = detect_enthymeme(text, detector_toks, detector_mdls, device)
    result = {
        "text": text,
        "is_enthymeme": is_enthymeme,
        "enthymeme_prob": round(enthymeme_prob, 4),
        "implicit_label": None,
        "label_probs": None,
        "implicit_content": None,
    }

    if is_enthymeme:
        label, label_probs = classify_label(text, clf_toks, clf_mdls, device)
        implicit = generate_implicit(text, label, bart_tok, bart_mdl, device)
        is_sentinel = implicit.strip() == SENTINEL_TOKEN
        result["implicit_label"] = label
        result["label_probs"] = {k: round(v, 4) for k, v in label_probs.items()}
        result["implicit_content"] = None if is_sentinel else implicit

    return result


def print_result(result):
    print(f"\nText: {result['text']}")
    print(f"Enthymeme detected: {result['is_enthymeme']}  (p={result['enthymeme_prob']:.3f})")
    if result["is_enthymeme"]:
        lp = result["label_probs"]
        print(f"Implicit role: {result['implicit_label']}  "
              f"(premise={lp['premise']:.3f}, conclusion={lp['conclusion']:.3f}, none={lp['none']:.3f})")
        print(f"Implicit content: {result['implicit_content'] or '[none generated]'}")


def main():
    parser = argparse.ArgumentParser(description="Enthymeme detection + generation pipeline")
    parser.add_argument("--models-root", required=True, help="Directory containing the fine-tuned checkpoints (see module docstring)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Single sentence to analyse")
    group.add_argument("--file", type=str, help="File with one sentence per line")
    group.add_argument("--interactive", action="store_true", help="Interactive prompt loop")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    models_root = Path(args.models_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading detector ensemble...")
    detector_toks, detector_mdls = load_detector(models_root, device)
    print("Loading label classifier ensemble...")
    clf_toks, clf_mdls = load_label_clf(models_root, device)
    print("Loading BART generator...")
    bart_tok, bart_mdl = load_generator(models_root, device)
    print("Models loaded.\n")

    results = []

    if args.text:
        r = run_pipeline(args.text, detector_toks, detector_mdls, clf_toks, clf_mdls, bart_tok, bart_mdl, device)
        print_result(r)
        results.append(r)
    elif args.file:
        with open(args.file) as f:
            sentences = [line.strip() for line in f if line.strip()]
        for sent in sentences:
            r = run_pipeline(sent, detector_toks, detector_mdls, clf_toks, clf_mdls, bart_tok, bart_mdl, device)
            print_result(r)
            results.append(r)
    elif args.interactive:
        print("Enter a sentence (or 'quit' to exit):")
        while True:
            try:
                text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            r = run_pipeline(text, detector_toks, detector_mdls, clf_toks, clf_mdls, bart_tok, bart_mdl, device)
            print_result(r)
            results.append(r)

    if args.output and results:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
