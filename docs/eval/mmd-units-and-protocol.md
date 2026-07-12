# MMD vs MMD² — what our pipeline emits and how it compares to DiGress / HiGen

## TL;DR

The function `compute_mmd` in `src/tmgg/evaluation/mmd_metrics.py`
returns a **squared MMD value** — `k11 + k22 - 2·k12`, the V-statistic
biased estimator. Every W&B key matching `gen-val/{degree,clustering,
spectral,orbit}_mmd`, every column named `*_mmd` in our exported
parquets, and every entry in `MMDResults` / `GraphEvaluator` results
holds this MMD² value, despite the unsuffixed name. The literature
calls this quantity "MMD" — it is the same convention used by
GraphRNN, GRAN, SPECTRE, DiGress, and HiGen.

**Practical implications:**

- HiGen Table 1's raw values (e.g. SBM degree=0.0013) are MMD². They are directly comparable to our `gen-val/degree_mmd` numbers — no transformation.
- DiGress Table 1's ratios (e.g. SBM degree r=1.6) are computed as `MMD²(gen, test) / MMD²(train, test)`. To match, divide our `gen-val/degree_mmd` by `MMD²(train, test)` for the same dataset and kernel — **do not square again**.
- Don't `sqrt` our values to "convert to MMD distances" before comparing to the literature. The literature reports the same MMD² we do, just under the unsuffixed name "MMD".

## What the code returns, in symbols

The biased V-statistic estimator of the squared Maximum Mean
Discrepancy is

$$\widehat{\text{MMD}}^2 = \frac{1}{n^2}\sum_{i,j} k(x_i,x_j) + \frac{1}{m^2}\sum_{i,j} k(y_i,y_j) - \frac{2}{nm}\sum_{i,j} k(x_i,y_j)$$

where the within-set sums include the diagonal `i==j` terms (this is
what makes it biased; the unbiased U-statistic excludes them and would
shift our reported numbers by O(1/n)).

`compute_mmd` (`src/tmgg/evaluation/mmd_metrics.py:385`) computes
exactly that and returns it without taking a square root. The function
clamps to `≥ 0` to absorb numerical noise; the underlying quantity is
non-negative by construction.

The same return-value shape applies to `compute_orbit_mmd`
(`src/tmgg/evaluation/graph_evaluator.py:173`) and
`compute_mmd_metrics` (`src/tmgg/evaluation/mmd_metrics.py:463`).

## What HiGen and DiGress do

**Both inherit the GraphRNN / GRAN evaluation protocol** (You et al.
ICML 2018; Liao et al. NeurIPS 2019). That protocol defines the four
graph-statistics MMDs (degree, clustering, orbit, spectral) using
`gaussian_tv` kernels with specific bandwidths, biased V-statistic,
and the convention of calling `k11 + k22 - 2·k12` "MMD" without the
square. SPECTRE (Martinkus et al. NeurIPS 2022), DiGress (Vignac et al.
ICLR 2023), and HiGen (Karami arXiv:2305.19337, May 2023) all use this
protocol verbatim.

| Source | What it reports | Comparable to our `gen-val/*_mmd` how? |
|--------|-----------------|----------------------------------------|
| **DiGress upstream code** (`src/analysis/dist_helper.py:139`) | `compute_mmd` returns `disc(s1,s1) + disc(s2,s2) - 2·disc(s1,s2)` — bit-exactly our formula. | Identity. We ported this. |
| **DiGress paper** Table 1 (Vignac et al. 2023, `arXiv:2209.14734v3`) | Appendix F.1: "we do not report raw numbers but ratios computed as follows: `r = MMD(generated, test)² / MMD(training, test)²`". The numerator and denominator are each one call to `compute_mmd` — i.e. each is already MMD² in our units; the squared notation in the paper is just labelling the function output, not an additional square operation. | Divide our `gen-val/*_mmd` by `MMD²(train, test)` computed offline on the same dataset + kernel. The result is dimensionless. |
| **HiGen paper** Table 1 (Karami 2023, `arXiv:2305.19337`) | Reports raw values for "DiGress" rows, e.g. SBM degree=0.0013, ENZYMES degree=0.004. These are `compute_mmd` outputs, not square-roots. | Identity (modulo kernel/sigma audit — see `internal notes (not included)` Task 3). |
| **HiGen code** (`github.com/Karami-m/HiGen_main`) | Reads metrics from GraphRNN/GRAN code path; same `disc + disc - 2·disc` shape. We have not run this code to byte-match, but the protocol is explicit in the README. | Identity, with the audit caveat. |

DiGress's `4n28svrj`-style ratio reporting masks an absolute-magnitude
benchmark that HiGen exposes. The two papers are therefore
complementary anchors: the paper-direct anchor is HiGen for raw
numbers; the paper-direct anchor is DiGress only via the ratio.

## How to compare, concretely

1. **Anchor on HiGen Table 1 raw numbers.** No transform; check the value matches sigmas and bin counts in `mmd_metrics.py`. See `internal notes (not included)` for the dataset-by-dataset list.
2. **Anchor on DiGress Table 1 ratios.** Compute `MMD²(train, test)` once per dataset (cached in `data/eval/mmd_baselines/<dataset>.json` per `src/tmgg/evaluation/mmd_baselines.py`). Divide each run's W&B-logged `gen-val/*_mmd` by the cached baseline value for the matching metric. Report the ratio.
3. **Don't sqrt.** Squaring and unsquaring shifts the absolute magnitude by a factor of (the value)^0.5 ≈ 0.04 for typical SBM clustering MMD², which is what makes "is this 0.0498 or 0.223?" so easy to get wrong. Stick to MMD² throughout.
4. **Don't double-square.** When you read DiGress's appendix `r = MMD(...)² / MMD(...)²`, both numerator and denominator are already squared by the function. The notation does not require you to square the function output again.

## Anti-patterns we have hit before

- Reporting our `gen-val/degree_mmd` (an MMD²) alongside DiGress paper r=1.6 (a ratio of MMD²) without dividing — comparing absolute squared distance to a dimensionless ratio. See `internal notes (not included)` §2.3 for the recovery from this mistake.
- Calling `compute_mmd`'s output "MMD" in docstrings — this fed the ambiguity. Fixed in this round; current docstrings say "MMD²" or "Biased MMD² value". If you find a stale "Returns the MMD" anywhere, fix it.
- Caching `mmd` (the sqrt) and `mmd_squared` separately in baseline files (`src/tmgg/evaluation/mmd_baselines.py`). This was done to defuse the ambiguity, not because both quantities are useful — readers should default to `mmd_squared`. The `mmd` field is only there for tooling that has already committed to the (imprecise) literature-style "MMD" name.

## Where this matters operationally

- **Cross-paper comparison:** anchor decisions in `anchors.yaml` and methodology §2.3 hinge on knowing both we and HiGen report MMD².
- **Ratio computation:** the entire `compute_ratios` flow in `mmd_baselines.py` and the analysis side of `scripts/sbm_repro_report.py` operates on MMD² inputs. Feeding sqrt-MMD would silently change every reported ratio by sqrt() vs nothing.
- **Reviewer-facing tables:** any cross-variant or cross-dataset comparison table — like `internal analysis exports (not included)` — must label its values "MMD² (V-statistic, gaussian_tv kernel)" or equivalent. "MMD" alone invites the reader to the same trap we keep falling into.

## See also

- `src/tmgg/evaluation/mmd_metrics.py` — the canonical implementation.
- `src/tmgg/evaluation/mmd_baselines.py` — schema + ratio helper; already explicit about MMD² vs MMD.
- `internal notes (not included)` §2.3 — narrative around the anchor unit-mismatch (HiGen vs DiGress).
- `internal notes (not included)` — open work on baseline computation and the kernel/sigma audit (Tasks 1–3).
- DiGress paper Appendix F.1 (`arXiv:2209.14734v3`) — verbatim quote on the ratio convention.
- HiGen paper Table 1 (`arXiv:2305.19337`) — raw MMD² anchors for SBM and ENZYMES.
