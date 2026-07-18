"""
ROSCOE-SS evaluation of BART pipeline generations vs. rewritten gold reconstructions
(Section 5.2, Table 2).

Steps:
  1. Load test_set_rewritten.csv  -> per-tweet gold references (recon{n}_rewritten)
  2. Load test_results.csv        -> BART implicit_content generations (from run_test_set.py)
  3. Match on id; keep only rows where BOTH sides are non-empty
  4. Rewrite BART generations with Qwen2.5-7B (same no-if-then prompt as annotator rewrites)
  5. Compute ROSCOE-SS per tweet: mean over annotators k of roscoe_ss(bart_rewritten, recon_k_rewritten)
  6. Report mean over all matched tweets; save per-row CSV
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from roscoe_utils import RoscoeEncoder, roscoe_ss

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

QWEN_ID = "Qwen/Qwen2.5-7B-Instruct"
RECON_RW_COLS = [f"recon{n}_rewritten" for n in range(1, 6)]

SYSTEM = (
    "You rewrite argumentative propositions. "
    "The input is a generated reconstruction of an implicit component of an enthymeme — "
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


def has_ref(row):
    return any(pd.notna(row[c]) and str(row[c]).strip() for c in RECON_RW_COLS)


def main():
    parser = argparse.ArgumentParser(description="ROSCOE-SS: BART generations vs. rewritten gold references")
    parser.add_argument("--rewritten-refs", default=str(REPO_ROOT / "results" / "generation_eval" / "test_set_rewritten.csv"))
    parser.add_argument("--bart-results", default=str(REPO_ROOT / "data" / "sample" / "pipeline_output_sample.csv"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "generation_eval"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device: {device}")

    refs_df = pd.read_csv(args.rewritten_refs)
    bart_df = pd.read_csv(args.bart_results)

    bart_gen = bart_df[bart_df["implicit_content"].notna() & (bart_df["implicit_content"].str.strip() != "")][["id", "implicit_content"]].copy()
    print(f"[eval] BART rows with generation : {len(bart_gen)}")

    merged = refs_df[["id", "majority_vote"] + RECON_RW_COLS].merge(bart_gen, on="id", how="inner")
    matched = merged[merged.apply(has_ref, axis=1)].copy().reset_index(drop=True)
    print(f"[eval] Matched (BART gen + annotation): {len(matched)}")

    print("\n[eval] Loading Qwen2.5-7B-Instruct for BART generation rewriting...")
    q_tok = AutoTokenizer.from_pretrained(QWEN_ID)
    q_model = AutoModelForCausalLM.from_pretrained(QWEN_ID, dtype=torch.float16, device_map=str(device))
    q_model.eval()

    def rewrite(text: str) -> str:
        messages = [{"role": "system", "content": SYSTEM}]
        for orig, rw in FEWSHOT:
            messages.append({"role": "user", "content": f"Original: {orig}"})
            messages.append({"role": "assistant", "content": rw})
        messages.append({"role": "user", "content": f"Original: {text}"})
        prompt = q_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = q_tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = q_model.generate(**enc, max_new_tokens=64, do_sample=False, num_beams=1, pad_token_id=q_tok.eos_token_id)
        return q_tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    bart_rewritten = []
    for i, (_, row) in enumerate(matched.iterrows()):
        gen = str(row["implicit_content"]).strip()
        print(f"[eval] rewriting [{i+1}/{len(matched)}] id={row['id']}: {gen[:60]}...")
        bart_rewritten.append(rewrite(gen))
    matched["bart_rewritten"] = bart_rewritten

    del q_model, q_tok
    torch.cuda.empty_cache()
    print("\n[eval] Qwen unloaded; GPU memory freed")

    print("[eval] Loading ROSCOE model...")
    encoder = RoscoeEncoder(device=device)
    print("[eval] ROSCOE model ready")

    hyps = matched["bart_rewritten"].tolist()
    all_ref_texts, ref_idx, pos = [], [], 0
    for _, row in matched.iterrows():
        refs = [str(row[c]).strip() for c in RECON_RW_COLS if pd.notna(row[c]) and str(row[c]).strip()]
        ref_idx.append(list(range(pos, pos + len(refs))))
        all_ref_texts.extend(refs)
        pos += len(refs)

    print(f"[eval] Embedding {len(hyps)} hypotheses and {len(all_ref_texts)} references...")
    hyp_embs = encoder.encode(hyps)
    ref_embs = encoder.encode(all_ref_texts)

    scores = [float(np.mean([roscoe_ss(hyp_embs[i], ref_embs[j]) for j in ref_idx[i]])) for i in range(len(hyps))]
    matched["roscoe_ss"] = scores

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, "roscoe_bart_rewritten.csv")
    out_cols = ["id", "majority_vote", "implicit_content", "bart_rewritten", *RECON_RW_COLS, "roscoe_ss"]
    matched[out_cols].to_csv(out_csv, index=False)

    sep = "=" * 60
    print(f"\n{sep}\n  ROSCOE-SS  |  BART pipeline vs. rewritten gold refs\n{sep}")
    print(f"  Matched tweets (BART gen + annotation) : {len(scores)}")
    print(f"  ROSCOE-SS  mean   : {np.mean(scores):.4f}")
    print(f"  ROSCOE-SS  median : {np.median(scores):.4f}")
    print(f"  ROSCOE-SS  std    : {np.std(scores):.4f}")
    print(sep)
    print(f"\n[eval] Per-row results saved -> {out_csv}")


if __name__ == "__main__":
    main()
