"""
Rewrite annotator gold reconstructions as declarative claim-like propositions.

For each non-empty recon{n}_text in the test set, uses Qwen2.5-7B-Instruct to
convert conditional/If-then phrasing into a direct claim while preserving the
same verbs and entity names (needed because ClaimBuster rarely scores
conditionals as check-worthy; see Section 5.3 / Appendix on claim detection).

Output: <output-dir>/test_set_rewritten.csv
"""
import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
RECON_COLS = [f"recon{n}_text" for n in range(1, 6)]

SYSTEM = (
    "You rewrite argumentative propositions. "
    "The input is an annotator's reconstruction of an implicit component of an enthymeme — "
    "it may be phrased as a conditional ('If X, then Y'), a hypothetical, or a verbose description. "
    "Your task: rewrite it as a direct, concise declarative claim (a proposition). "
    "Rules: (1) do NOT use 'If ... then ...' structure; "
    "(2) keep the same verbs and entity/noun phrases as the original; "
    "(3) output ONLY the rewritten proposition as a single sentence, no preamble, no explanation."
)

FEWSHOT = [
    ("If you fail once, people will not believe you again",
     "Failing once causes people to permanently distrust you"),
    ("If someone forces medical procedures on others, they forfeit the right to support",
     "Those who force medical procedures on others forfeit the right to receive support"),
    ("If the government censors data, it is trying to hide the truth",
     "Governments that censor data are hiding the truth"),
]


def build_rewriter(model_id, device):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map=str(device))
    model.eval()

    def rewrite(text: str) -> str:
        messages = [{"role": "system", "content": SYSTEM}]
        for orig, rewritten in FEWSHOT:
            messages.append({"role": "user", "content": f"Original: {orig}"})
            messages.append({"role": "assistant", "content": rewritten})
        messages.append({"role": "user", "content": f"Original: {text}"})
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=64, do_sample=False, num_beams=1, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    return rewrite


def main():
    parser = argparse.ArgumentParser(description="Rewrite conditional gold reconstructions into declarative claims")
    parser.add_argument("--test-set", default=str(REPO_ROOT / "data" / "sample" / "test_set_sample.csv"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "generation_eval"))
    parser.add_argument("--model-id", default=MODEL_ID)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, "test_set_rewritten.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[rewrite] Device: {device}")
    print(f"[rewrite] Loading {args.model_id} ...")
    rewrite = build_rewriter(args.model_id, device)
    print("[rewrite] Model ready")

    df = pd.read_csv(args.test_set)
    print(f"[rewrite] Loaded {len(df)} rows from {args.test_set}")

    ann_label_cols = [f"ann{n}_label" for n in range(1, 6)]
    total = sum(df[c].notna().sum() for c in RECON_COLS if c in df.columns)
    done = 0

    for n in range(1, 6):
        col_orig = f"recon{n}_text"
        col_new = f"recon{n}_rewritten"
        if col_orig not in df.columns:
            continue
        df[col_new] = ""
        for idx, row in df.iterrows():
            orig = str(row[col_orig]).strip() if pd.notna(row[col_orig]) else ""
            if not orig:
                continue
            done += 1
            print(f"[rewrite] [{done}/{total}] ann{n} row {row['id']}: {orig[:60]}...")
            df.at[idx, col_new] = rewrite(orig)

    out_cols = ["id", "tweet_text", "majority_vote"]
    for n in range(1, 6):
        lbl, orig, rw = f"ann{n}_label", f"recon{n}_text", f"recon{n}_rewritten"
        if lbl in df.columns:
            out_cols.append(lbl)
        if orig in df.columns:
            out_cols.append(orig)
        if rw in df.columns:
            out_cols.append(rw)

    df[out_cols].to_csv(out_file, index=False)
    print(f"\n[rewrite] Done. Saved {len(df)} rows -> {out_file}")


if __name__ == "__main__":
    main()
