# Rationale — `TestNormalizeAdjFlag` in `test_digress_gnn_projections.py`

## Context

`BareGraphConvolutionLayer` (used by `GraphTransformer` when any of
`use_gnn_q/k/v` is True) historically applied symmetric normalisation
$\tilde A = D^{-1/2} A D^{-1/2}$ unconditionally inside its `forward`.
We made the normalisation optional via a new constructor argument
`normalize_adjacency: bool = True`, plumbed through
`projection_config["gnn_normalize_adj"]` →
`GraphTransformer` → `XEyTransformerLayer` → `NodeEdgeBlock`.

The defaults must be `True` everywhere so existing experiments keep
producing bit-identical outputs.

## Assumed starting state

- `tmgg.models.layers.gcn.BareGraphConvolutionLayer` accepts
  `normalize_adjacency: bool = True`.
- `tmgg.models.layers.graph_ops.sym_normalize_adjacency` returns
  $D^{-1/2} A D^{-1/2}$, leaving zero-degree rows zeroed.
- `_PROJ_KEYS` in `transformer_model.py` includes
  `"gnn_normalize_adj"` so the legacy `hidden_dims` fallback works.

## Invariants the tests pin

1. **`test_layer_identity_off_on_normalized_input`** — Algebraic
   equivalence: a layer with `normalize_adjacency=False` fed
   $\tilde A$ produces the *same* output as a layer with
   `normalize_adjacency=True` fed $A$, given identical weights. This
   is the cleanest correctness pin: it asserts the new branch
   *replaces* the normalisation rather than introducing a different
   operator. Self-loops are added before normalisation so no row gets
   zero-degree-masked, which would create a vacuous equality.

2. **`test_layer_off_uses_raw_adjacency`** — Negative test: ensures
   the flag is not silently dead. With the same `A` fed to both
   layers, outputs must differ; otherwise the gating in `forward` is
   disconnected.

3. **`test_default_preserved_end_to_end`** — Bit-equality between an
   omitted flag and an explicit `True`. Two `GraphTransformer`s
   constructed under identical seeds with `projection_config={...,
   "gnn_normalize_adj": True}` and `projection_config={... no flag}`
   must produce identical output tensors. Catches accidental default
   flips during refactors.

4. **`test_wiring_smoke_normalize_off`** — End-to-end propagation.
   With `gnn_normalize_adj: False` set on the model-level
   `projection_config`, every Q/K/V `BareGraphConvolutionLayer` in
   every transformer layer carries `normalize_adjacency=False`. Walks
   the full plumbing path. Reaches into `model.transformer.tf_layers`
   to confirm the attribute, since a forward pass alone does not
   distinguish silent fall-through.

5. **`test_hidden_dims_fallback`** — The legacy `hidden_dims`-based
   config path also picks up the new key. Some older Stage-1 configs
   place projection knobs in `hidden_dims` rather than a dedicated
   `projection_config`; `_PROJ_KEYS` must list the new key for that
   path to work.

## Why these are the right pins

The risky failure modes are all silent:

- The flag could be accepted at the `GraphTransformer` constructor
  but not threaded into `XEyTransformerLayer` or `NodeEdgeBlock`,
  defaulting to `True` on the inner layer regardless. Test 4 catches
  this by asserting on the leaf layer attribute.
- The `_PROJ_KEYS` set could miss the new key, making the
  `hidden_dims` legacy path silently ignore it. Test 5 catches this.
- A typo in the key name in either the dict-extraction logic or the
  layer-pass-through could leave the default permanently `True`.
  Test 1 + test 2 together force the actual `forward` behaviour to
  observably differ when the flag is set.
