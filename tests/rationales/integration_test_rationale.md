# Integration Test Rationale

This document explains the purpose, design, and expected behavior of the
integration tests for experiment runners and config composition.

## Purpose

The integration tests serve as a pre-deployment validation layer. They verify
that all experiment configurations and CLI runners work correctly before
launching expensive GPU jobs on Modal. A failing test here costs seconds;
a failing Modal job costs dollars and hours of debugging.

## Test Categories

### Runner Integration Tests (`test_runner_integration.py`)

These tests invoke each CLI runner via subprocess with minimal training
parameters (2 steps, 8 samples, CPU execution).

**What they catch:**
- Import errors in runner modules
- Hydra decorator configuration issues
- Missing config files or invalid defaults
- Path resolution failures
- Data module instantiation errors
- Model construction failures

**Design rationale:**
- Subprocess isolation prevents Hydra state conflicts between tests
- 120-second timeout catches infinite loops or hangs
- Exit code 0 assertion catches Python exceptions
- Minimal data/steps ensures tests complete quickly (~10-30s each)

### Config Composition Tests (`test_config_composition.py`)

These tests load configs via Hydra's programmatic API and verify that all
components can be instantiated and executed.

**What they catch:**
- Missing interpolation references (e.g., `${undefined_key}`)
- Invalid _target_ paths for instantiation
- Incompatible config compositions
- Model architecture mismatches (wrong input dimensions)
- Forward pass failures (NaN, shape errors)

**Design rationale:**
- `hydra.compose()` tests config composition without subprocess overhead
- `hydra.utils.instantiate()` validates that configs produce real objects
- Forward pass with sample data catches shape/dimension errors
- Trainer instantiation validates Lightning configuration

## Test Markers

- `@pytest.mark.integration` - All integration tests
- `@pytest.mark.slow` - Full subprocess runner tests (~10-30s each)
- `@pytest.mark.config` - Config composition tests (~1-2s each)
- `@pytest.mark.runner` - CLI runner tests

## Assumptions

1. **CPU execution**: Tests run on CPU regardless of GPU availability
2. **Minimal data**: 4-8 samples, batch size 2, no workers
3. **Brief training**: 2 steps maximum, no checkpointing
4. **No logging**: W&B and TensorBoard disabled
5. **Temporary directories**: All outputs go to pytest's `tmp_path`

## Invariants Verified

1. **Exit code 0**: Runner subprocess completes successfully
2. **No exceptions**: stderr contains no Python tracebacks
3. **Output created**: Experiment directory exists after run
4. **Valid objects**: instantiate() returns LightningModule/DataModule
5. **Finite output**: Forward pass produces no NaN/Inf values
6. **Shape preservation**: Model output matches input dimensions

## Running the Tests

```bash
# All integration tests
uv run pytest -m integration -v

# Config tests only (faster)
uv run pytest -m config -v

# Runner tests only (slower, more comprehensive)
uv run pytest -m "integration and slow" -v

# Skip slow tests for quick feedback
uv run pytest -m "integration and not slow" -v

# Single runner test
uv run pytest tests/test_runner_integration.py::TestExperimentRunners::test_runner_executes_brief_training[tmgg-spectral-arch-base_config_spectral_arch] -v
```

## Adding New Runners

When adding a new experiment runner:

1. Add the runner to `pyproject.toml` under `[project.scripts]`
2. Add the runner to `EXPERIMENT_RUNNERS` or `STAGE_RUNNERS` in
   `test_runner_integration.py`
3. Add the base config to `BASE_CONFIGS` in `test_config_composition.py`
4. Run the integration tests to verify

## Adding New Configs

When adding a new config file:

1. Ensure it has a `_target_` for instantiation
2. Test locally: `uv run tmgg-<runner> trainer.max_steps=2`
3. Add to appropriate test parametrization if not auto-discovered
4. Run config composition tests
