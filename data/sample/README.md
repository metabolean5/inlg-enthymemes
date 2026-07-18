# Sample data

These CSVs are small, illustrative excerpts of the datasets used in the paper —
**not** the full corpus. They exist so the code in `src/` can be smoke-tested
and so readers can see the exact schema each stage expects. For the full
annotated dataset, see the companion dataset paper (Pastor, 2026) or contact
the authors.

## `enthymeme_annotations_sample.csv` (20 rows)

Per-annotator enthymeme annotations on the development set (mirrors the full
`merged_annotations_v2.csv`). One row per tweet, with up to 5 annotators:

| column | meaning |
|---|---|
| `id` | tweet id |
| `tweet_text` | the tweet |
| `ann{1..5}_label` | that annotator's label: `none` / `premise` / `conclusion` |
| `ann{1..5}_implicit` | that annotator's reconstructed implicit text (empty if `none`) |
| `majority_label` | majority vote across annotators |

## `test_set_sample.csv` (15 rows)

Excerpt of the untouched, 148-tweet held-out test set (mirrors
`Eval/test_set_final.csv`), with the same per-annotator label/reconstruction
columns as above plus `recon{1..5}_text` (the raw reconstruction text) and
`majority_vote`.

## `pipeline_output_sample.csv` (15 rows)

Excerpt of the full pipeline's predictions on the test set (mirrors
`inlg/test_results.csv`): detection probability, predicted implicit role
(premise/conclusion/none), and the BART-generated implicit content.

## `error_analysis_sample.csv` (7 rows)

The exact tweets discussed as examples in the paper's error analysis
(Section 6) — detection failures, normative-vs-factual generation, and
overgeneration cases — with gold annotations, pipeline predictions, and the
Qwen baseline's outputs side by side.
