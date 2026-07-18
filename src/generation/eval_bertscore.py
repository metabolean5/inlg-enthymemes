"""
BERTScore cross-check for the generation-quality evaluation (Section 5.2):
same three comparisons as the ROSCOE-SS scripts (BART pipeline, IAA ceiling,
random-permutation floor), reported side by side to show that ROSCOE-SS
discriminates genuine reconstruction from topical noise more sharply than a
token-level metric.

Model: roberta-large (the standard BERTScore English model).
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from bert_score import score as bert_score_fn

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

RECON_RW_COLS = [f"recon{n}_rewritten" for n in range(1, 6)]
N_PERMS = 20
SEED = 42
MODEL = "roberta-large"


def get_rewrites(row):
    return [str(row[c]).strip() for c in RECON_RW_COLS if pd.notna(row[c]) and str(row[c]).strip()]


def bs_f1(hyps, refs_list, device):
    flat_hyps, flat_refs, idx = [], [], []
    for i, (h, rs) in enumerate(zip(hyps, refs_list)):
        for r in rs:
            flat_hyps.append(h)
            flat_refs.append(r)
            idx.append(i)
    _, _, F1 = bert_score_fn(flat_hyps, flat_refs, model_type=MODEL, device=device, verbose=False, batch_size=64)
    f1_np = F1.numpy()
    return [float(np.mean(f1_np[[j for j, x in enumerate(idx) if x == i]])) for i in range(len(hyps))]


def main():
    parser = argparse.ArgumentParser(description="BERTScore cross-check vs. ROSCOE-SS")
    parser.add_argument("--bart-scored", default=str(REPO_ROOT / "results" / "generation_eval" / "roscoe_bart_rewritten.csv"))
    parser.add_argument("--rewritten-refs", default=str(REPO_ROOT / "results" / "generation_eval" / "test_set_rewritten.csv"))
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"[bs] BERTScore model: {MODEL}\n[bs] Device: {args.device}\n")

    bart_df = pd.read_csv(args.bart_scored)
    refs_df = pd.read_csv(args.rewritten_refs)

    print("-- 1. BART pipeline --------------------------------------")
    hyps_bart = bart_df["bart_rewritten"].tolist()
    refs_bart = [[str(row[c]).strip() for c in RECON_RW_COLS if pd.notna(row[c]) and str(row[c]).strip()]
                 for _, row in bart_df.iterrows()]
    scores_bart = bs_f1(hyps_bart, refs_bart, args.device)
    print(f"  n={len(scores_bart)}  mean F1={np.mean(scores_bart):.4f}\n")

    print("-- 2. IAA ceiling ------------------------------------------")
    eligible = refs_df[refs_df.apply(lambda r: len(get_rewrites(r)) >= 2, axis=1)].copy().reset_index(drop=True)
    iaa_flat_hyps, iaa_flat_refs, iaa_tweet_idx = [], [], []
    for tweet_i, (_, row) in enumerate(eligible.iterrows()):
        rws = get_rewrites(row)
        for h_idx in range(len(rws)):
            iaa_flat_hyps.append(rws[h_idx])
            iaa_flat_refs.append([rws[r_idx] for r_idx in range(len(rws)) if r_idx != h_idx])
            iaa_tweet_idx.append(tweet_i)
    iaa_pair_scores = bs_f1(iaa_flat_hyps, iaa_flat_refs, args.device)
    tweet_scores_iaa = {}
    for score, t_idx in zip(iaa_pair_scores, iaa_tweet_idx):
        tweet_scores_iaa.setdefault(t_idx, []).append(score)
    scores_iaa = [float(np.mean(v)) for v in tweet_scores_iaa.values()]
    print(f"  n={len(scores_iaa)} tweets  mean F1={np.mean(scores_iaa):.4f}\n")

    print("-- 3. Random permutation baseline --------------------------")
    ref_pool = [(row["id"], get_rewrites(row)) for _, row in refs_df.iterrows() if get_rewrites(row)]
    rng = random.Random(SEED)
    perm_means = []
    for perm in range(N_PERMS):
        wrong_refs = []
        for _, row in bart_df.iterrows():
            candidates = [(tid, rws) for tid, rws in ref_pool if tid != row["id"]]
            _, rws = rng.choice(candidates)
            wrong_refs.append(rws)
        perm_scores = bs_f1(hyps_bart, wrong_refs, args.device)
        m = float(np.mean(perm_scores))
        perm_means.append(m)
        print(f"  perm {perm+1:2d}/{N_PERMS}  mean F1 = {m:.4f}")

    sep = "=" * 68
    print(f"\n{sep}\n  BERTScore F1 ({MODEL})  vs.  ROSCOE-SS\n{sep}")
    print(f"  {'System':<35} {'BERTScore F1':>12}")
    print(f"  {'IAA — human ceiling':<35} {np.mean(scores_iaa):>12.4f}")
    print(f"  {'BART pipeline':<35} {np.mean(scores_bart):>12.4f}")
    print(f"  {'Random floor (mean over 20 perms)':<35} {np.mean(perm_means):>12.4f}")
    print(sep)


if __name__ == "__main__":
    main()
