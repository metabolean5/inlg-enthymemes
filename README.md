# Identifying Check-Worthy Claims by Generating the Implicit Content of Enthymemes

Code accompanying the paper *"Identifying Check-Worthy Claims by Generating the
Implicit Content of Enthymemes."* Fact-checking systems focus almost
exclusively on explicit claims, but a lot of misleading discourse works by
leaving its key claim unstated — as an *enthymeme*, an argument with a missing
premise or conclusion. This repo implements a three-stage pipeline that:

1. **Detects** whether a tweet contains enthymematic argumentation (DeBERTa-v3).
2. **Generates** the missing premise or conclusion (BART-base), with
   hyperparameters selected directly against the ROSCOE semantic-similarity
   metric rather than a token-overlap metric.
3. **Assesses check-worthiness** of the reconstructed content (ClaimBuster),
   after rewriting conditional ("if X then Y") reconstructions into direct
   declarative claims.

See the paper for the full motivation, evaluation, and error analysis.

## Repository layout

```
src/
  pipeline.py                     end-to-end inference: detect -> classify role -> generate
  run_test_set.py                 run the pipeline over a whole test-set CSV

  detection/
    hpo_cv.py                     Optuna HPO with 5-fold CV (Section 4.1.1, Appendix A)
    train_ensemble_cv.py          3-seed ensemble training + threshold tuning (Table 1)
    train_baselines_and_test.py   TF-IDF baselines + single DeBERTa, held-out test eval
    utils/                        HPO/CV helpers (metric computation, fold construction, logging)

  generation/
    rewrite_reconstructions.py    rewrite gold "if X then Y" reconstructions into claims (Qwen2.5-7B)
    eval_bart_roscoe.py           ROSCOE-SS: BART generations vs. gold references (Table 2)
    eval_iaa_roscoe.py            ROSCOE-SS inter-annotator agreement (human ceiling)
    eval_random_baseline_roscoe.py  ROSCOE-SS random-permutation floor
    eval_bertscore.py             BERTScore cross-check of the above three
    roscoe_utils.py                shared ROSCOE encoder

  claim_detection/
    score_reconstructions.py      ClaimBuster scoring of gold + system reconstructions (Table 3)
    qwen_baseline.py               Qwen detect+generate+ClaimBuster baseline (Table 3)

data/sample/                      small CSV samples illustrating each stage's I/O (see its README)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10+, a CUDA GPU is strongly recommended for anything
beyond the smoke tests below (DeBERTa-v3-large and BART fine-tuning, and the
Qwen2.5-7B baseline, all expect a GPU).

## Data

Only small **samples** are included, under `data/sample/` (see its README for
the schema of each file) — enough to smoke-test every script end to end, not
to reproduce the paper's numbers. The full annotated dataset (tweets about
vaccination and immigration, with per-annotator enthymeme reconstructions
grounded in Walton's argumentation schemes) is described in the companion
dataset paper (Pastor, 2026); contact the authors for access.

Point scripts at your own copy of the full data via the `ENTHYMEME_DATA` /
`ENTHYMEME_TEST_DATA` environment variables, or the equivalent `--*-csv`
arguments, e.g.:

```bash
ENTHYMEME_DATA=/path/to/merged_annotations.csv \
ENTHYMEME_TEST_DATA=/path/to/test_set.csv \
python src/detection/train_baselines_and_test.py
```

## Models

Fine-tuned checkpoints (the DeBERTa-v3 detector/label-classifier ensembles and
the BART generator) are **not** included in this repository — they're large
binary artifacts, not code. `src/pipeline.py` and `src/run_test_set.py` expect
a `--models-root` directory laid out as described in `pipeline.py`'s
docstring; train your own with the scripts in `src/detection/` and a BART
fine-tuning run using the `task: generate_implicit | label: <label> | tweet: <text>`
input format described in the paper (Section 4.2).

## Usage

**Detection** — HPO, then train the final ensemble:
```bash
python src/detection/hpo_cv.py --checkpoint microsoft/deberta-v3-base --n-trials 30
python src/detection/train_ensemble_cv.py --checkpoint microsoft/deberta-v3-base --lr 2e-5
python src/detection/train_baselines_and_test.py
```

**Generation evaluation** (requires a fine-tuned BART pipeline's output, e.g.
`run_test_set.py`'s CSV, and rewritten gold references):
```bash
python src/generation/rewrite_reconstructions.py --test-set data/sample/test_set_sample.csv
python src/generation/eval_bart_roscoe.py
python src/generation/eval_iaa_roscoe.py
python src/generation/eval_random_baseline_roscoe.py
python src/generation/eval_bertscore.py
```

**Claim detection**:
```bash
python src/claim_detection/score_reconstructions.py
python src/claim_detection/qwen_baseline.py
```

**End-to-end inference** on new text, once you have trained checkpoints:
```bash
python src/pipeline.py --models-root /path/to/models --text "Your sentence here."
```

## Citation

This paper is currently under review. A citation will be added here upon
publication.

## License

MIT — see [LICENSE](LICENSE).
