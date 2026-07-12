# Rationale — `TestNormalizeEigenvaluesFlag` in `test_digress_spectral_projections.py`

## Context

`SpectralProjectionLayer` historically rescaled eigenvalues by
$\bar\lambda = \lambda / \max|\lambda|$ (clamped at $10^{-6}$ from
below) inside its `forward`. We made the rescale optional via a new
constructor argument `normalize_eigenvalues: bool = True`, plumbed
through `projection_config["spectral_normalize_eigenvalues"]` →
`GraphTransformer` → `XEyTransformerLayer` → `NodeEdgeBlock`.

Defaults are `True` everywhere so existing experiments keep producing
bit-identical outputs. Mirror of the GNN-side rationale, applied to
the eigenvalue-rescale gate.

## Assumed starting state

- `tmgg.models.layers.spectral_projection.SpectralProjectionLayer`
  accepts `normalize_eigenvalues: bool = True`.
- The rescale formula in `forward` is exactly
  $\Lambda_{\text{used}} = \Lambda / \max|\Lambda|$ when the flag is
  True, with the max-magnitude clamped at `1e-6`.
- `_PROJ_KEYS` in `transformer_model.py` includes
  `"spectral_normalize_eigenvalues"`.

## Invariants the tests pin

1. **`test_layer_identity_off_on_normalized_input`** — Algebraic
   equivalence: a layer with `normalize_eigenvalues=False` fed
   $\bar\Lambda$ produces the *same* output as a layer with
   `normalize_eigenvalues=True` fed $\Lambda$, given identical
   weights. The eigenvalues are drawn with magnitude floored at 0.1
   so the `max(...).clamp(min=1e-6)` is non-degenerate (the floor
   matters: a draw with a near-zero max would hit the clamp and the
   rescale would no longer match the test's hand-computed
   $\bar\Lambda$).

2. **`test_layer_off_uses_raw_eigenvalues`** — Negative test: with
   $\max|\Lambda| \neq 1$ (we scale by $\times 3$), the two layers
   must produce different outputs given the same input; otherwise the
   gating is disconnected.

3. **`test_default_preserved_end_to_end`** — Bit-equality between an
   omitted flag and an explicit `True`. Two `GraphTransformer`s
   constructed under identical seeds with
   `projection_config={..., "spectral_normalize_eigenvalues": True}`
   and `projection_config={... no flag}` must produce identical
   output tensors.

4. **`test_wiring_smoke_normalize_off`** — End-to-end propagation.
   With `spectral_normalize_eigenvalues: False` set on the
   model-level `projection_config`, every Q/K/V
   `SpectralProjectionLayer` in every transformer layer carries
   `normalize_eigenvalues=False`. Forward must still run; we feed a
   real adjacency through the model so the eigen layer produces
   bounded eigenvalues, avoiding the dense-graph overflow
   condition the docstring warns about.

5. **`test_hidden_dims_fallback`** — Legacy `hidden_dims`-based
   config path picks up the new key.

## Why these are the right pins

Same failure-mode reasoning as the GNN-side rationale: the dangerous
modes are silent — flag accepted but not threaded, key absent from
`_PROJ_KEYS`, typo. Tests 1 + 2 force observable behavioural
difference, test 3 catches default flips, test 4 walks the full
plumbing, test 5 catches the legacy-config gap.
