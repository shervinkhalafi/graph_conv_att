# Repro panel ablations — MMDs vs paper anchors

Snapshot: **2026-05-06 20:13 UTC**.

Single-page measurement summary across the SBM and ENZYMES `_repro`
panels: per-step MMD trajectories, last-cycle headline values,
ratios against the train↔test baseline, and ratio comparisons to
DiGress paper Table 1 and HiGen Table 1.

All MMD values are V-statistic squared MMD² (gen↔val for runs,
train↔test for baselines), per the GraphRNN/GRAN convention. Unit
semantics: see [`docs/eval/mmd-units-and-protocol.md`](mmd-units-and-protocol.md).
For the prior detailed analysis (caveats, GDPO-vs-Vignac config drift,
reproduction caveats), see [`2026-05-06-mmd-ratio-analysis.md`](2026-05-06-mmd-ratio-analysis.md).

## Anchors

- **DiGress paper** (Vignac et al., ICLR 2023): Table 1 reports
  `r = MMD²(gen, test) / MMD²(train, test)`. Has SBM, no ENZYMES.
- **HiGen paper** (Karami, `arXiv:2305.19337`): Table 1 reports
  *raw* MMD² values for "DiGress" rows on both SBM and ENZYMES. We
  divide by our cached train↔test baseline to get HiGen-implied ratios
  in our pipeline's units.
- **Our baselines** (`data/eval/mmd_baselines/{spectre_sbm,pyg_enzymes}.json`,
  computed by `graph-denoising-mmd-baselines` on Modal 2026-05-06):
  - SBM: `degree=3.41e-04, clustering=3.31e-02, orbit=3.10e-02, spectral=2.82e-03`
  - ENZYMES: `degree=3.00e-04, clustering=1.04e-02, orbit=1.73e-04, spectral=2.85e-03`

### Anchor conversion via our measured baseline

The DiGress paper publishes only ratios, but their evaluation
protocol uses SPECTRE-SBM splits identical to ours (128 train / 32
val / 40 test). So multiplying `r_paper` by our measured train↔test
baseline gives the absolute MMD² we'd expect their generator to
produce on our pipeline:

```
anchor_abs = anchor_ratio × baseline_train_test
```

For HiGen this conversion is definitional — HiGen publishes raw
MMD² and we computed `r_higen` by *dividing* by our baseline, so
the conversion just recovers HiGen's published number. For DiGress
it's a faithful absolute under the matching-protocol assumption.

**SBM** (baseline computed 2026-05-06 by `graph-denoising-mmd-baselines`):

| metric     | baseline_train↔test (ours) | DiGress paper r | DiGress paper abs MMD² (= r × baseline) | HiGen r | HiGen abs MMD² (published) |
|------------|---------------------------:|----------------:|----------------------------------------:|--------:|---------------------------:|
| degree     | 3.41e-04 | 1.6 | 5.46e-04 | 3.81 | 1.30e-03 |
| clustering | 3.31e-02 | 1.5 | 4.97e-02 | 1.50 | 4.97e-02 |
| orbit      | 3.10e-02 | 1.7 | 5.27e-02 | 1.40 | 4.34e-02 |
| spectral   | 2.82e-03 | n/a | n/a      | n/a  | n/a      |

**ENZYMES** (baseline computed 2026-05-06):

| metric     | baseline_train↔test (ours) | DiGress paper r | DiGress paper abs MMD² (= r × baseline) | HiGen r | HiGen abs MMD² (published) |
|------------|---------------------------:|----------------:|----------------------------------------:|--------:|---------------------------:|
| degree     | 3.00e-04 | n/a | n/a | 13.34 | 4.00e-03 |
| clustering | 1.04e-02 | n/a | n/a | 7.95  | 8.27e-02 |
| orbit      | 1.73e-04 | n/a | n/a | 11.55 | 2.00e-03 |
| spectral   | 2.85e-03 | n/a | n/a | n/a   | n/a      |

DiGress paper has no ENZYMES evaluation, so the paper-anchor column
is empty for that dataset.

## SBM panel

### Last-cycle MMD² (raw)

| variant | run_id | health | last step | degree | cluster | orbit | spectral |
|---------|--------|:------:|----------:|-------:|--------:|------:|---------:|
| `vignac_repro` (mask-bug)              | `12s2b4a7` | ✗ invalidated | 375k | 0.193 | 0.131 | 0.0953 | 0.211 |
| `vignac_spectral_repro` (mask-bug)     | `e5pd9drt` | ✗ invalidated | 75k | 0.225 | 0.130 | 0.0842 | 0.213 |
| `pearl_repro` (mask-bug)               | `s07qwx3b` | ✗ invalidated | 450k | 0.190 | 0.130 | 0.104 | 0.213 |
| `pearl_spectral_repro` (mask-bug)      | `jbraoj7o` | ✗ invalidated | 375k | 0.195 | 0.130 | 0.0766 | 0.208 |
| `pearl_gnnconv_norm_repro` (mask-bug)  | `rarihsee` | ✗ invalidated | 375k | 0.187 | 0.130 | 0.119 | 0.215 |
| `pearl_gnnconv_raw_repro` (mask-bug)   | `qao36vwu` / `g1g6xpx1` | ✗ blew up | — | 0.299 / 0.433 | 0.135 / 0.140 | 0.143 / 0.142 | 0.227 / 0.270 |
| **`vignac_repro_exact`** (post-fix)    | `cgfv3f85` (`2026-05-06-sbm-vignac-1`)            | ✓ running | 41.8k (1 cycle) | 0.187 | 0.135 | 0.0909 | 0.210 |
| **`pearl_repro_exact`** (post-fix)     | `k4iiw5sg` (`2026-05-06-sbm-pearl-1`)             | ✓ running | 44.0k (1 cycle) | 0.181 | 0.131 | 0.108 | 0.211 |
| **`pearl_spectral_repro_exact`** (post-fix) | `qukgm6zu` (`2026-05-06-sbm-pearl-spectral-1`) | ✓ running | 41.2k (1 cycle) | 0.210 | 0.132 | 0.130 | 0.211 |
| **`pearl_gnnconv_norm_repro_exact`** (post-fix) | `5qchu8c4` (`2026-05-06-sbm-pearl-gnnconv-1`) | ✓ running | 44.0k (1 cycle) | 0.188 | 0.131 | 0.137 | 0.223 |

### Ratios `r_run = run / baseline`

| variant | run_id | r_degree | r_cluster | r_orbit (last) | r_orbit (best) | r_spectral |
|---------|--------|---------:|----------:|---------------:|---------------:|-----------:|
| `vignac_repro` (mask-bug)              | `12s2b4a7` | 565 | 3.96 | 3.07 | **3.07** (375k) | 74.7 |
| `vignac_spectral_repro` (mask-bug)     | `e5pd9drt` | 660¹ | 3.93 | **2.72** | **2.72** (75k only) | 75.4 |
| `pearl_repro` (mask-bug)               | `s07qwx3b` | 555 | 3.94 | 3.35 | 2.72 (150k) | 75.7 |
| `pearl_spectral_repro` (mask-bug)      | `jbraoj7o` | 573 | 3.93 | **2.47** | **2.47** (375k) | 73.9 |
| `pearl_gnnconv_norm_repro` (mask-bug)  | `rarihsee` | 549 | 3.92 | 3.85 | 3.01 (150k) | 76.2 |
| `pearl_gnnconv_raw_repro` (1) (mask-bug) | `qao36vwu` | 877 | 4.09 | 4.60 | — | 80.4 |
| `pearl_gnnconv_raw_repro` (2) (mask-bug) | `g1g6xpx1` | 1269 | 4.23 | 4.58 | — | 95.7 |
| **`vignac_repro_exact`** (post-fix)            | `cgfv3f85` | 549² | 4.07 | **2.93** | **2.93** (cycle 1 only) | 74.5 |
| **`pearl_repro_exact`** (post-fix)             | `k4iiw5sg` | **531**² | 3.96 | 3.49 | 3.49 (cycle 1 only) | 74.9 |
| **`pearl_spectral_repro_exact`** (post-fix)    | `qukgm6zu` | 616² | 3.99 | 4.19 | 4.19 (cycle 1 only) | 74.9 |
| **`pearl_gnnconv_norm_repro_exact`** (post-fix) | `5qchu8c4` | 551² | 3.96 | 4.42 | 4.42 (cycle 1 only) | 79.0 |
| **HiGen reproduction** (Table 1 / our baseline) | | **3.81** | **1.50** | **1.40** | | n/a |
| **DiGress paper** Table 1 | | **1.6** | **1.5** | **1.7** | | n/a |

¹ `e5pd9drt` is at step 75k only — only one eval cycle so far. Degree
may still be in the high-bias early phase.

² Post-fix runs all at cycle 1 only (trainer step ~22k = first eval).
Same caveat as ¹: degree and orbit have only one observation each, may
move noticeably in subsequent cycles. Pre-fix (mask-bug-invalidated)
rows show the trajectories are mostly flat after cycle 1, so these are
indicative of the panel basin but not final.

### Per-step trajectory (CSV)

Pre-fix (mask-bug-invalidated) trajectories preserved for trend
context; post-fix rows below.

```csv
run_id,variant,step,degree_mmd,clustering_mmd,orbit_mmd,spectral_mmd
12s2b4a7,vignac_repro_maskbug,75000,0.1997,0.1301,0.1085,0.2155
12s2b4a7,vignac_repro_maskbug,150000,0.1832,0.1309,0.1155,0.2119
12s2b4a7,vignac_repro_maskbug,225000,0.1842,0.1302,0.1139,0.2158
12s2b4a7,vignac_repro_maskbug,300000,0.1848,0.1326,0.1424,0.2126
12s2b4a7,vignac_repro_maskbug,375000,0.1928,0.1309,0.0953,0.2106
e5pd9drt,vignac_spectral_repro_maskbug,75000,0.2251,0.1300,0.0842,0.2125
s07qwx3b,pearl_repro_maskbug,75000,0.1849,0.1305,0.1075,0.2158
s07qwx3b,pearl_repro_maskbug,150000,0.1882,0.1305,0.0842,0.2085
s07qwx3b,pearl_repro_maskbug,225000,0.1856,0.1339,0.0946,0.2115
s07qwx3b,pearl_repro_maskbug,300000,0.1878,0.1315,0.1001,0.2149
s07qwx3b,pearl_repro_maskbug,375000,0.1893,0.1317,0.0871,0.2167
s07qwx3b,pearl_repro_maskbug,450000,0.1896,0.1303,0.1038,0.2135
jbraoj7o,pearl_spectral_repro_maskbug,75000,0.2154,0.1301,0.1059,0.2138
jbraoj7o,pearl_spectral_repro_maskbug,150000,0.1867,0.1342,0.0989,0.2072
jbraoj7o,pearl_spectral_repro_maskbug,225000,0.1853,0.1300,0.1167,0.2131
jbraoj7o,pearl_spectral_repro_maskbug,300000,0.2112,0.1305,0.0907,0.2099
jbraoj7o,pearl_spectral_repro_maskbug,375000,0.1953,0.1300,0.0766,0.2083
rarihsee,pearl_gnnconv_norm_repro_maskbug,75000,0.1896,0.1297,0.1187,0.2187
rarihsee,pearl_gnnconv_norm_repro_maskbug,150000,0.1925,0.1287,0.0934,0.2186
rarihsee,pearl_gnnconv_norm_repro_maskbug,225000,0.1838,0.1318,0.1186,0.2170
rarihsee,pearl_gnnconv_norm_repro_maskbug,300000,0.1842,0.1334,0.0994,0.2126
rarihsee,pearl_gnnconv_norm_repro_maskbug,375000,0.1874,0.1299,0.1193,0.2149
cgfv3f85,vignac_repro_exact,22000,0.1871,0.1347,0.0909,0.2102
k4iiw5sg,pearl_repro_exact,22000,0.1813,0.1312,0.1084,0.2113
qukgm6zu,pearl_spectral_repro_exact,22000,0.2100,0.1320,0.1298,0.2113
5qchu8c4,pearl_gnnconv_norm_repro_exact,22000,0.1878,0.1309,0.1370,0.2229
```

## ENZYMES panel

### Last-cycle MMD² (raw)

| variant | run_id | health | last step | degree | cluster | orbit | spectral |
|---------|--------|:------:|----------:|-------:|--------:|------:|---------:|
| `vignac_repro` (mask-bug)             | `l1nk0622` | ✗ invalidated (finished) | 525k | 0.190 | 0.118 | 0.0896 | 0.208 |
| `pearl_repro` (mask-bug)              | `ge461v1o` | ✗ invalidated (killed) | 375k | 0.187 | 0.103 | 0.163 | 0.201 |
| `pearl_spectral_repro` (mask-bug)     | `4n28svrj` | ✗ invalidated (killed) | 450k | 0.188 | 0.115 | 0.176 | 0.187 |
| `pearl_gnnconv_norm_repro` (mask-bug) | `ly0d6lyi` | ✗ invalidated (failed/infra) | 225k | 0.184 | 0.110 | 0.148 | 0.203 |
| `pearl_gnnconv_norm_repro` (3, mask-bug) | `zyawhwrx` | ✗ invalidated (killed) | 75k | 0.181 | 0.104 | 0.076 | 0.187 |
| `pearl_gnnconv_raw_repro` (mask-bug)  | `dt0ux9zh` | ✗ blew up | 375k | 0.211 | 0.094 | 0.538 | 0.196 |
| **`vignac_repro_exact`** (post-fix)    | `8nhefhnl` (`2026-05-06-enzymes-vignac-1`)             | ✓ running | 178.7k (2 cycles) | 0.180 | 0.106 | 0.132 | 0.196 |
| **`pearl_repro_exact`** (post-fix)     | `7yi627fv` (`2026-05-06-enzymes-pearl-1`)              | ✓ running | 134.0k (1 cycle)  | 0.183 | 0.0967 | 0.151 | 0.190 |
| **`pearl_spectral_repro_exact`** (post-fix) | `ths6e1da` (`2026-05-06-enzymes-pearl-spectral-1`) | ✓ running | 113.6k (1 cycle)  | 0.188 | 0.0974 | 0.255 | 0.197 |
| **`pearl_gnnconv_norm_repro_exact`** (post-fix) | `xsmz6yql` (`2026-05-06-enzymes-pearl-gnnconv-1`) | ✓ running | 159.2k (2 cycles) | 0.218 | 0.102 | 0.531 | 0.203 |

### Ratios `r_run = run / baseline`

| variant | run_id | r_degree | r_cluster | r_orbit (last) | r_orbit (best) | r_spectral |
|---------|--------|---------:|----------:|---------------:|---------------:|-----------:|
| `vignac_repro` (mask-bug)             | `l1nk0622` | 633 | 11.30 | 518 | **311** (225k) | 72.9 |
| `pearl_repro` (mask-bug)              | `ge461v1o` | 624 | 9.91 | 941 | **487** (75k) | 70.6 |
| `pearl_spectral_repro` (mask-bug)     | `4n28svrj` | 627 | 11.08 | 1016 | **390** (75k) | 65.7 |
| `pearl_gnnconv_norm_repro` (mask-bug) | `ly0d6lyi` | 613 | 10.61 | 854 | 440 (75k) | 71.3 |
| `pearl_gnnconv_norm_repro` (3, mask-bug) | `zyawhwrx` | 603 | 10.00 | 440 | 440 (75k only) | 65.8 |
| `pearl_gnnconv_raw_repro` (mask-bug)  | `dt0ux9zh` | 705 | 9.02 | 3107 | — | 68.6 |
| **`vignac_repro_exact`** (post-fix)             | `8nhefhnl` | 601 | 10.23 | 765 | **700** (75k) | 68.8 |
| **`pearl_repro_exact`** (post-fix)              | `7yi627fv` | 609 | **9.30** | 875 | 875 (75k only) | 66.7 |
| **`pearl_spectral_repro_exact`** (post-fix)     | `ths6e1da` | 627 | 9.37 | 1474 | 1474 (75k only) | 69.2 |
| **`pearl_gnnconv_norm_repro_exact`** (post-fix) | `xsmz6yql` | 727 | 9.81 | 3068 | 2236 (75k) | 71.2 |
| **HiGen reproduction** (Table 1 / our baseline) | | **13.34** | **7.95** | **11.55** | | n/a |
| **DiGress paper** Table 1 | | n/a | n/a | n/a | | n/a |

### Per-step trajectory (CSV)

```csv
run_id,variant,step,degree_mmd,clustering_mmd,orbit_mmd,spectral_mmd
l1nk0622,vignac_repro_maskbug,75000,0.1842,0.1198,0.0743,0.1990
l1nk0622,vignac_repro_maskbug,150000,0.1797,0.1076,0.1152,0.1932
l1nk0622,vignac_repro_maskbug,225000,0.1899,0.1168,0.0538,0.2028
l1nk0622,vignac_repro_maskbug,300000,0.1822,0.1029,0.0946,0.2012
l1nk0622,vignac_repro_maskbug,375000,0.1868,0.1108,0.1188,0.2048
l1nk0622,vignac_repro_maskbug,450000,0.1857,0.1344,0.0818,0.2084
l1nk0622,vignac_repro_maskbug,525000,0.1901,0.1175,0.0896,0.2078
ge461v1o,pearl_repro_maskbug,75000,0.1830,0.1140,0.0890,0.1938
ge461v1o,pearl_repro_maskbug,150000,0.1815,0.1020,0.2070,0.1973
ge461v1o,pearl_repro_maskbug,225000,0.1855,0.1078,0.1139,0.2029
ge461v1o,pearl_repro_maskbug,300000,0.1857,0.1084,0.1980,0.2004
ge461v1o,pearl_repro_maskbug,375000,0.1873,0.1031,0.1629,0.2011
4n28svrj,pearl_spectral_repro_maskbug,75000,0.1818,0.1021,0.0674,0.1821
4n28svrj,pearl_spectral_repro_maskbug,150000,0.1801,0.1159,0.0741,0.1835
4n28svrj,pearl_spectral_repro_maskbug,225000,0.1794,0.1104,0.1858,0.1924
4n28svrj,pearl_spectral_repro_maskbug,300000,0.1874,0.1082,0.0968,0.1965
4n28svrj,pearl_spectral_repro_maskbug,375000,0.1846,0.1123,0.0850,0.1858
4n28svrj,pearl_spectral_repro_maskbug,450000,0.1881,0.1152,0.1757,0.1872
ly0d6lyi,pearl_gnnconv_norm_repro_maskbug,75000,0.1810,0.1040,0.0760,0.1874
ly0d6lyi,pearl_gnnconv_norm_repro_maskbug,150000,0.1901,0.1126,0.1359,0.1934
ly0d6lyi,pearl_gnnconv_norm_repro_maskbug,225000,0.1839,0.1103,0.1477,0.2031
zyawhwrx,pearl_gnnconv_norm_repro_maskbug,75000,0.1810,0.1040,0.0760,0.1874
8nhefhnl,vignac_repro_exact,75000,0.1817,0.1082,0.1211,0.1961
8nhefhnl,vignac_repro_exact,150000,0.1803,0.1064,0.1324,0.1960
7yi627fv,pearl_repro_exact,75000,0.1827,0.0967,0.1513,0.1900
ths6e1da,pearl_spectral_repro_exact,75000,0.1880,0.0974,0.2550,0.1974
xsmz6yql,pearl_gnnconv_norm_repro_exact,75000,0.2116,0.0981,0.3868,0.2004
xsmz6yql,pearl_gnnconv_norm_repro_exact,150000,0.2181,0.1017,0.5306,0.2029
```

## Quick reads

1. **Mask fix did not move the panel basin.** Post-fix `vignac_repro_exact`
   at trainer step ~22k sits at degree 0.187 / clustering 0.135 / orbit
   0.091 / spectral 0.210, essentially the same MMD basin as the
   mask-bug-invalidated runs at the same early step. Conclusion: the
   diagonal-mask divergence was a correctness defect (state-dict
   non-equivalence to upstream), not the source of the published-anchor
   gap. The anchor gap survives the fix and must be explained by
   something else — kernel/sigma, undertraining, baseline-protocol
   drift, or a different bug entirely.
2. **Trajectories are flat after step 75k** (pre-fix data, mask-bug
   invalidated but useful as trend evidence). Every healthy variant
   plateaus at panel-floor MMDs by the first eval cycle and barely
   moves over the next 5–7 cycles. Post-fix we have 1–2 cycles per
   variant — too few to confirm the plateau survives the fix, but the
   first-cycle values land in the same range.
3. **Orbit is volatile cycle-to-cycle.** Within a single run, orbit
   can swing ~3× (e.g. `ge461v1o` 0.089 → 0.207 → 0.114; `4n28svrj`
   0.067 → 0.186 → 0.097). Post-fix `xsmz6yql` already shows 0.387 →
   0.531 across two cycles. Degree, clustering, spectral are stable
   to ≲15% across cycles. Headline reports must use min-or-mean over
   multiple cycles, not last-cycle.
4. **Clustering is calibrated.** SBM post-fix at 3.96–4.07 vs HiGen
   1.50 / paper 1.5 — same 2.6× gap as pre-fix. ENZYMES post-fix at
   9.30–10.23 vs HiGen 7.95 — 1.2× off, slightly tighter than pre-fix
   (9.91–11.30). The pearl_repro_exact 9.30 is the panel best on
   ENZYMES clustering. **HiGen ≡ paper on SBM clustering** — strong
   evidence the protocol is bandwidth-stable across implementations.
5. **Orbit on ENZYMES regressed early in post-fix data.** Pre-fix
   l1nk0622 hit 0.0538 by step 225k (best cycle); post-fix runs
   sit at 0.121–0.531 at step 75k–150k. Likely a combination of
   (a) only 1–2 cycles measured so far so we may be catching unlucky
   cycles in the high-volatility regime, and (b) post-fix runs are
   still in the early-bias phase. Re-evaluate after step 200k.
6. **Degree is still broken everywhere.** SBM post-fix 531–616× vs
   paper 1.6 (~330×); ENZYMES post-fix 601–727× vs HiGen 13.34
   (~46×). Same magnitude as pre-fix. Even HiGen's own degree ratio
   (3.81) was 2.4× off DiGress paper, so we're well past their gap.
   The small absolute baseline (3e-4) amplifies any absolute generation
   error; combined with apparent undertraining of the block-degree
   distribution. Mask fix did not touch this.
7. **Spectral has no anchor**, just a uniform 65–80× across both
   datasets and all healthy variants, pre- and post-fix. Suggestively
   flat — possible systematic histogram-bin/range mismatch vs upstream
   `spectral_stats` (PICKUP doc Task 3 §3).
8. **`_raw_` is unviable on both datasets.** Pre-fix SBM r_degree
   877/1269; ENZYMES r_orbit 3107. The variant was killed terminally
   2026-05-06 09:35–09:59 UTC and has no post-fix successor.
9. **GDPO-vs-`_exact` drift is now measurable.** Pre-fix
   `vignac_repro` (12s2b4a7) at step 75k: degree 0.200 / cluster 0.130
   / orbit 0.109 / spectral 0.216. Post-fix `vignac_repro_exact`
   (cgfv3f85) at step ~22k: degree 0.187 / cluster 0.135 / orbit 0.091
   / spectral 0.210. Different steps confound a clean comparison; the
   gap will become readable once cgfv3f85 reaches step 75k+.

## Best-per-metric across all healthy runs and all eval cycles

`gap` = best run raw MMD² ÷ tightest published anchor abs MMD² (i.e.
the smallest anchor between DiGress paper and HiGen, where both exist).

### Pre-fix (mask-bug, INVALIDATED — kept for trend reference)

| dataset | metric | best raw MMD² | run / step | r_run | tightest anchor abs MMD² | gap |
|---------|--------|--------------:|------------|------:|-------------------------:|----:|
| SBM     | degree     | 0.183  | `s07qwx3b` / 150k | 537 | 5.46e-04 (paper) | **335×** |
| SBM     | clustering | 0.129  | several / various | 3.90 | 4.97e-02 (paper ≡ HiGen) | 2.6× |
| SBM     | orbit      | **0.0766** | `jbraoj7o` / 375k | **2.47** | 4.34e-02 (HiGen) | 1.8× |
| SBM     | spectral   | 0.207  | several / various | 73.5 | n/a | n/a |
| ENZYMES | degree     | 0.179  | `4n28svrj` / 225k | 597 | 4.00e-03 (HiGen) | **45×** |
| ENZYMES | clustering | 0.102  | `4n28svrj` / 75k  | 9.82 | 8.27e-02 (HiGen) | 1.2× |
| ENZYMES | orbit      | **0.0538** | `l1nk0622` / 225k | **311** | 2.00e-03 (HiGen) | **27×** |
| ENZYMES | spectral   | 0.182  | `4n28svrj` / 75k  | 63.9 | n/a | n/a |

### Post-fix (current authoritative reference, 1–2 cycles per variant)

| dataset | metric | best raw MMD² | run / step | r_run | tightest anchor abs MMD² | gap |
|---------|--------|--------------:|------------|------:|-------------------------:|----:|
| SBM     | degree     | **0.1813** | `k4iiw5sg` / ~22k | 531 | 5.46e-04 (paper) | **332×** |
| SBM     | clustering | 0.1312     | `k4iiw5sg` / ~22k | 3.96 | 4.97e-02 (paper ≡ HiGen) | 2.6× |
| SBM     | orbit      | **0.0909** | `cgfv3f85` / ~22k | **2.93** | 4.34e-02 (HiGen) | 2.1× |
| SBM     | spectral   | 0.2102     | `cgfv3f85` / ~22k | 74.5 | n/a | n/a |
| ENZYMES | degree     | 0.1803     | `8nhefhnl` / 150k | 601 | 4.00e-03 (HiGen) | **45×** |
| ENZYMES | clustering | **0.0967** | `7yi627fv` / 75k  | **9.30** | 8.27e-02 (HiGen) | 1.2× |
| ENZYMES | orbit      | 0.1211     | `8nhefhnl` / 75k  | 700 | 2.00e-03 (HiGen) | **61×** |
| ENZYMES | spectral   | 0.1900     | `7yi627fv` / 75k  | 66.7 | n/a | n/a |

Reads from the post-fix gap column: SBM clustering and orbit are
within ~2–3× of an anchor — plausibly closeable. SBM degree is 332×
off; ENZYMES degree 45× and orbit 61×. Degree is the dominant
pathology on both datasets, and the mask-bug fix did not change the
order of magnitude. **Caveat: post-fix runs have 1–2 eval cycles
each so far; pre-fix trajectories show orbit can swing 3× per
cycle, so post-fix bests will tighten as more cycles land — but
degree, clustering, and spectral are stable cycle-to-cycle so those
gaps are unlikely to move much.**

## Data sources

- Run state and per-step MMDs: W&B via `mcp__wandb__get_run_history_tool`,
  pulled 2026-05-06 11:56 UTC. Project per variant
  (`your-wandb-entity/discrete-{sbm,enzymes}-<variant>-repro`).
- Baselines: `data/eval/mmd_baselines/{spectre_sbm,pyg_enzymes}.json`,
  computed by `graph-denoising-mmd-baselines` 2026-05-06.
- Anchors: DiGress (`arXiv:2209.14734v3` Table 1), HiGen
  (`arXiv:2305.19337` Table 1).

## See also

- [`internal notes (not included)`](internal notes (not included)) — index of running and finished runs.
- [`internal notes (not included)`](2026-05-06-mmd-ratio-analysis.md)
  — prior detailed reproduction-caveat write-up; same anchors,
  pre-finish snapshot.
- [`docs/eval/mmd-units-and-protocol.md`](mmd-units-and-protocol.md)
  — V-statistic vs U-statistic, kernel choice, sigma rationale.
- [`internal notes (not included)`](../../internal notes (not included))
  — open work on baseline computation and the kernel/sigma audit.
