"""
Random permutation baseline for ROSCOE-SS (Section 5.2, Table 2, "random floor" row).

For each matched BART tweet, replaces the correct rewritten gold refs with
rewritten refs from a randomly chosen *different* tweet that has annotations.
Repeats N_PERMS times and averages -> the score the metric gives when the
semantic content is completely wrong but the surface form is identical
(proposition, same domain). If this floor is close to the BART score, the
metric is collapsing to topic overlap; if it is clearly lower, it discriminates.
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from roscoe_utils import RoscoeEncoder, roscoe_ss

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

RECON_RW_COLS = [f"recon{n}_rewritten" for n in range(1, 6)]
N_PERMS = 20
SEED = 42


def get_rewrites(row):
    return [str(row[c]).strip() for c in RECON_RW_COLS if pd.notna(row[c]) and str(row[c]).strip()]


def main():
    parser = argparse.ArgumentParser(description="ROSCOE-SS random-permutation floor")
    parser.add_argument("--bart-scored", default=str(REPO_ROOT / "results" / "generation_eval" / "roscoe_bart_rewritten.csv"),
                         help="output of eval_bart_roscoe.py (26 matched rows in the paper)")
    parser.add_argument("--rewritten-refs", default=str(REPO_ROOT / "results" / "generation_eval" / "test_set_rewritten.csv"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "generation_eval"))
    parser.add_argument("--n-perms", type=int, default=N_PERMS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[rand] Device: {device}")

    bart_df = pd.read_csv(args.bart_scored)
    refs_df = pd.read_csv(args.rewritten_refs)

    ref_pool = [(row["id"], get_rewrites(row)) for _, row in refs_df.iterrows() if get_rewrites(row)]
    print(f"[rand] Reference pool: {len(ref_pool)} tweets with annotations")
    print(f"[rand] BART matched rows: {len(bart_df)}")

    print("[rand] Loading ROSCOE model...")
    encoder = RoscoeEncoder(device=device)
    print("[rand] ROSCOE model ready")

    hyps = bart_df["bart_rewritten"].tolist()
    print(f"[rand] Encoding {len(hyps)} BART hypotheses...")
    hyp_embs = encoder.encode(hyps, batch_size=128)

    rng = random.Random(SEED)
    perm_means = []

    for perm in range(args.n_perms):
        wrong_refs_per_hyp = []
        for _, row in bart_df.iterrows():
            candidates = [(tid, rws) for tid, rws in ref_pool if tid != row["id"]]
            _, wrong_rws = rng.choice(candidates)
            wrong_refs_per_hyp.append(wrong_rws)

        all_wrong, ref_idx, pos = [], [], 0
        for rws in wrong_refs_per_hyp:
            ref_idx.append(list(range(pos, pos + len(rws))))
            all_wrong.extend(rws)
            pos += len(rws)

        wrong_embs = encoder.encode(all_wrong, batch_size=128)
        scores = [float(np.mean([roscoe_ss(hyp_embs[i], wrong_embs[j]) for j in ref_idx[i]])) for i in range(len(hyps))]
        perm_mean = float(np.mean(scores))
        perm_means.append(perm_mean)
        print(f"[rand] perm {perm+1:2d}/{args.n_perms}  mean ROSCOE-SS = {perm_mean:.4f}")

    sep = "=" * 60
    print(f"\n{sep}\n  ROSCOE-SS Random Permutation Baseline  ({args.n_perms} runs)\n{sep}")
    print(f"  BART matched tweets      : {len(bart_df)}")
    print(f"  Mean over perms  mean    : {np.mean(perm_means):.4f}")
    print(f"  Mean over perms  std     : {np.std(perm_means):.4f}")
    print(sep)

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, "roscoe_random_baseline.csv")
    pd.DataFrame({"perm": range(1, args.n_perms + 1), "roscoe_ss_mean": perm_means}).to_csv(out_csv, index=False)
    print(f"\n[rand] Saved -> {out_csv}")


if __name__ == "__main__":
    main()
