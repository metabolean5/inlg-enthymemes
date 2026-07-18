"""
Inter-annotator agreement (IAA) via ROSCOE-SS on Qwen-rewritten gold reconstructions
(Section 5.2, Table 2, "human ceiling" row).

For each tweet with >= 2 rewritten reconstructions, computes all pairwise
ROSCOE-SS scores and averages within the tweet, then averages across tweets.
"""
import argparse
import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from roscoe_utils import RoscoeEncoder, roscoe_ss

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

RECON_RW_COLS = [f"recon{n}_rewritten" for n in range(1, 6)]


def get_rewrites(row):
    return [str(row[c]).strip() for c in RECON_RW_COLS if pd.notna(row[c]) and str(row[c]).strip()]


def main():
    parser = argparse.ArgumentParser(description="ROSCOE-SS inter-annotator agreement (human ceiling)")
    parser.add_argument("--rewritten-refs", default=str(REPO_ROOT / "results" / "generation_eval" / "test_set_rewritten.csv"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "generation_eval"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[iaa] Device: {device}")

    df = pd.read_csv(args.rewritten_refs)
    print(f"[iaa] Loaded {len(df)} rows from {args.rewritten_refs}")

    eligible = df[df.apply(lambda r: len(get_rewrites(r)) >= 2, axis=1)].copy().reset_index(drop=True)
    print(f"[iaa] Rows with >= 2 rewritten reconstructions: {len(eligible)}")

    print("[iaa] Loading ROSCOE model...")
    encoder = RoscoeEncoder(device=device)
    print("[iaa] ROSCOE model ready")

    all_texts, row_slices = [], []
    for _, row in eligible.iterrows():
        rewrites = get_rewrites(row)
        start = len(all_texts)
        all_texts.extend(rewrites)
        row_slices.append((start, len(all_texts), rewrites))

    print(f"[iaa] Encoding {len(all_texts)} texts...")
    all_embs = encoder.encode(all_texts, batch_size=128)

    records, all_row_scores = [], []
    for i, (_, row) in enumerate(eligible.iterrows()):
        start, end, rewrites = row_slices[i]
        embs = all_embs[start:end]
        pairs = list(itertools.combinations(range(len(embs)), 2))
        pair_scores = [roscoe_ss(embs[a], embs[b]) for a, b in pairs]
        row_mean = float(np.mean(pair_scores))
        all_row_scores.append(row_mean)
        records.append({
            "id": row["id"],
            "majority_vote": row["majority_vote"],
            "n_annotators": len(embs),
            "n_pairs": len(pairs),
            "roscoe_ss_mean": row_mean,
            "roscoe_ss_min": float(np.min(pair_scores)),
            "roscoe_ss_max": float(np.max(pair_scores)),
            "rewrites": " | ".join(rewrites),
        })

    out_df = pd.DataFrame(records)
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, "roscoe_iaa.csv")
    out_df.to_csv(out_csv, index=False)

    sep = "=" * 60
    print(f"\n{sep}\n  ROSCOE-SS IAA  |  Qwen-rewritten gold reconstructions\n{sep}")
    print(f"  Eligible tweets (>= 2 annotators)  : {len(all_row_scores)}")
    print(f"  Total annotator pairs evaluated    : {out_df['n_pairs'].sum()}")
    print(f"  ROSCOE-SS IAA  mean   : {np.mean(all_row_scores):.4f}")
    print(f"  ROSCOE-SS IAA  median : {np.median(all_row_scores):.4f}")
    print(sep)
    print(f"\n[iaa] Per-row results saved -> {out_csv}")


if __name__ == "__main__":
    main()
