# Published anchors

Two papers anchor our generator-quality measurements: DiGress (Vignac
et al., ICLR 2023, `arXiv:2209.14734v3`) and HiGen (Karami,
`arXiv:2305.19337`). Both report MMD² values; their conventions differ
on whether they publish ratios or raw values.

## DiGress paper

- Reports `r = MMD²(gen, test) / MMD²(train, test)` in Table 1.
- Has SBM, planar, and ego datasets. **No ENZYMES.**
- SPECTRE-SBM splits identical to ours (128 / 32 / 40), so we can
  compute their absolute MMD² as `r_paper × baseline_train_test`.

| metric | DiGress paper r | implied abs MMD² (= r × our SBM baseline) |
|--------|----------------:|------------------------------------------:|
| degree | 1.6 | 5.46e-04 |
| clustering | 1.5 | 4.97e-02 |
| orbit | 1.7 | 5.27e-02 |
| spectral | n/a | n/a |

## HiGen paper

- Reports raw MMD² values in Table 1 ("DiGress" rows).
- Both SBM and ENZYMES.
- We divide by our cached baseline to get HiGen-implied ratios in
  our pipeline's units.

**SBM** (HiGen Table 1):

| metric | HiGen abs MMD² | implied r_higen |
|--------|---------------:|----------------:|
| degree | 1.30e-03 | 3.81 |
| clustering | 4.97e-02 | 1.50 |
| orbit | 4.34e-02 | 1.40 |

**ENZYMES** (HiGen Table 1):

| metric | HiGen abs MMD² | implied r_higen |
|--------|---------------:|----------------:|
| degree | 4.00e-03 | 13.34 |
| clustering | 8.27e-02 | 7.95 |
| orbit | 2.00e-03 | 11.55 |

## Conversion

```
r_run     = MMD²(gen-val, val) / MMD²(train, test)        # our pipeline
r_paper   = published ratio (DiGress only)
abs_paper = r_paper × MMD²(train, test)                    # our baseline assumed compatible
abs_higen = published raw MMD² (HiGen)
r_higen   = abs_higen / MMD²(train, test)                  # our baseline
gap       = best_run_abs / min(abs_paper, abs_higen)       # how far we are from the tightest anchor
```

## Important caveats

1. **HiGen ≡ paper on SBM clustering** — both report 1.50 / 4.97e-02.
   Strong evidence the bandwidth-pair protocol is stable across
   implementations.
2. **HiGen ≠ paper on SBM degree** — HiGen 3.81, paper 1.6 (2.4× off).
   Either kernel-bandwidth or seed variance.
3. **DiGress paper has no ENZYMES anchor.** All ENZYMES gap analysis
   is against HiGen.
4. **Reproduction caveats.** Our `vignac_repro` (pre-fix) is *not*
   byte-equivalent to upstream DiGress: `dim_ffy=2048` (upstream 256),
   `amsgrad=true`, code is a tmgg port. The `_repro_exact` (post-fix)
   configs drop those tweaks. See
   `../snapshots/ablations-measurement-2026-05-06.md` "Reproduction caveats".

## Cross-links

- `BASELINES-CONTEXT.md` — how the train↔test baselines were computed.
- `mmd-units-and-protocol.md` — V-statistic vs U-statistic, kernel,
  sigma.
- `../snapshots/ablations-measurement-2026-05-06.md` — anchor-table
  with current run values.
