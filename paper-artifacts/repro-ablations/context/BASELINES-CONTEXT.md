# Baselines context

The "baseline" for each MMD metric is the train↔test MMD² computed on
the canonical splits, with no model in the loop. This is the natural
denominator for a ratio: `r_run = MMD²(gen, val) / MMD²(train, test)`.
A perfect generator would produce r ≈ 1; published anchors (DiGress,
HiGen) report values in the 1–14 range depending on dataset and metric.

## Files in this folder

- `mmd_baselines/spectre_sbm.json` — train↔test MMD² on the SPECTRE-SBM
  splits (128 train / 32 val / 40 test).
- `mmd_baselines/pyg_enzymes.json` — train↔test MMD² on PyG ENZYMES
  splits (480 train / 60 val / 60 test).

## Computation

Both files were produced 2026-05-06 by the `graph-denoising-mmd-baselines` Modal
entry point, which loads the canonical splits and calls
`compute_mmd_metrics(train_graphs, test_graphs)` from
`tmgg.evaluation.mmd_metrics`. Same kernel and bandwidth as the
`gen-val/*_mmd` keys in the runs.

Re-run if the splits change or the kernel implementation is updated.

## Values (verbatim from the JSONs)

**SBM** (`spectre_sbm.json`):

| metric | train↔test MMD² |
|--------|-----------------:|
| degree | 3.41e-04 |
| clustering | 3.31e-02 |
| orbit | 3.10e-02 |
| spectral | 2.82e-03 |

**ENZYMES** (`pyg_enzymes.json`):

| metric | train↔test MMD² |
|--------|-----------------:|
| degree | 3.00e-04 |
| clustering | 1.04e-02 |
| orbit | 1.73e-04 |
| spectral | 2.85e-03 |

## Why orbit's ENZYMES baseline is so small

The 1.73e-04 baseline is two orders of magnitude smaller than the
clustering or spectral baselines because the orbit MMD on this dataset
is dominated by very-rare orbit counts; the train↔test variability is
correspondingly low. This makes the ratio extremely sensitive to even
small absolute generation errors, which explains the ENZYMES orbit
gap (~10²–10³×) reported in the measurement docs.

## Cross-links

- `mmd-units-and-protocol.md` — unit semantics, kernel choice, sigma rationale.
- `ANCHORS.md` — DiGress paper and HiGen anchors and conversion math.
