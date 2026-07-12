"""Profiling helpers — wrap training and eval entrypoints with torch.profiler.

The two runners (``run_train_profile``, ``run_eval_profile``) are kept
small and importable so they can be invoked either:

  * directly from a host shell (CPU-only, slow but useful for dev loops)
  * inside a Modal container via ``tmgg.modal._profile_functions``

Both produce two artefacts under ``output_dir``:

  * ``trace.json`` — Chrome-trace format; load in chrome://tracing or
    perfetto.dev for flame-graph inspection.
  * ``summary.txt`` — ``key_averages().table()`` ranked by total CPU
    time and CUDA time; the high-leverage human-readable view.
"""
