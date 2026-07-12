# Debugging the Modal app (graph-denoising)

Practical playbook for when a run crashes on Modal. The Modal CLI is
**live-streaming only**: once a container exits, logs are no longer
retrievable from the command line (they remain in the web dashboard at
modal.com, which Claude cannot access). All debugging happens by
reproducing the crash under a live stream and capturing it to a file.

## Project surface

- **App name:** `graph-denoising` (defined in `src/tmgg/modal/app.py`).
- **Entrypoint module:** `tmgg.modal._functions` — holds `modal.App(...)`
  and all `@app.function` decorators. Deploying = `modal deploy -m
  tmgg.modal._functions`.
- **Deploy task:** `mise run modal-deploy` (tees logs
  to `/tmp/tmgg-modal-logs/modal-deploy-<ts>.log`).
- **Launch wrapper:** `run-upstream-digress-sbm-modal.zsh` invokes
  `uv run tmgg-modal run tmgg-discrete-gen ...`. DEBUG wrappers shadow
  it with specific knobs disabled (`skip-orbit`, `skip-sbm`, `no-viz`).
- **Env knobs in the launcher:** `DEPLOY_FIRST=1`, `DETACH=1`,
  `DRY_RUN=0`, `GPU_TIER=standard`.

## Modal CLI limits (read this first)

| Command | Behaviour |
|---|---|
| `modal app list` | Deployed/running/recently-stopped apps with IDs. |
| `modal app logs <id-or-name>` | **Streams while active.** Hangs silently if no live tasks. No `--since`, no `--tail`. |
| `modal container list` | Currently-running containers only. |
| `modal container logs <id>` | Streams for a live container only. |
| `modal app history <id>` | Deployment versions, not runtime logs. |
| `modal run <ref>` | Runs AND streams stdout/stderr locally until the container exits. |

**Implication:** to debug you must capture the crash live. You cannot
post-mortem a finished run from the CLI.

## Two ways to get live output

### A. Foreground run (preferred for debugging)

Set `DETACH=0` so `modal run` stays attached and streams the crash
directly to your terminal. Tee to a file for inspection.

```bash
cd .
DETACH=0 DEPLOY_FIRST=1 \
  ./DEBUG-run-upstream-digress-sbm-modal-no-viz.zsh \
  2>&1 | tee /tmp/tmgg-modal-debug-$(date +%Y%m%d-%H%M%S).log
```

The traceback appears in the tee'd file. No second terminal needed.

### B. Detached run + tail in parallel

If you need to keep the default `--detach` behaviour (e.g. a long run
you want to survive disconnect), stream from a second shell right
after launching.

```bash
# shell 1: launch
./DEBUG-run-upstream-digress-sbm-modal-skip-orbit.zsh

# shell 2: attach to the live app stream as soon as tasks appear.
# Start this within ~10s of launch — otherwise you may catch only the
# tail of the run.
LOG=/tmp/tmgg-modal-stream-$(date +%Y%m%d-%H%M%S).log
modal app logs graph-denoising 2>&1 | tee "$LOG" &
LOGPID=$!

# poll the log
tail -f "$LOG"  # or periodic: wc -l "$LOG"; tail -80 "$LOG"

# when done
kill $LOGPID
```

Name-based addressing (`graph-denoising`) is more stable than the random
app ID; the ID changes on redeploy.

### Background-task pattern (from within Claude Code)

When running non-interactively, launch the streamer with
`run_in_background: true` on Bash and poll its output file, rather than
blocking on `modal app logs`. The command hangs forever if no tasks are
live, so always use `timeout` or background + kill.

## Deploy, then launch

The wrappers run `mise run modal-deploy` first because a stale image
will happily run old code and mask the fix you just made. If you are
iterating on source inside `src/tmgg/modal/` or anything in the Modal
image, you **must** redeploy. If you're iterating on pure
Hydra/training config, you can skip by exporting `DEPLOY_FIRST=0`.

The deploy step itself logs to `/tmp/tmgg-modal-logs/modal-deploy-*.log`.
Image build errors land there, not in the run log.

## Quick diagnosis recipes

| Symptom | First check |
|---|---|
| Hang at launch, no output | `modal app list` — is the deploy new? `modal container list` — did any task actually start? |
| Crash inside orbit/MMD metric | Re-run with `DEBUG-...-skip-orbit.zsh`. If it then runs, the fault is in `orca` / orbit_mmd path. |
| Crash inside SBM eval | `DEBUG-...-skip-sbm.zsh` isolates the SBM-specific metric. |
| Crash inside visualisation | `DEBUG-...-no-viz.zsh` keeps metrics on, disables viz only. Isolates matplotlib / figure writers. |
| Image build fails | Look in the latest `/tmp/tmgg-modal-logs/modal-deploy-*.log`, not the run log. |

## What the DEBUG wrappers disable

Each wrapper forwards to `run-upstream-digress-sbm-modal.zsh` with one
override. Read them directly — they're three lines of `exec` plus the
override — and combine by appending more overrides on the CLI.

## Budget / safety

GPU runs cost real money and wall-clock time. Before launching from
Claude Code:

1. Prefer the shortest reproduction (small `data.num_graphs`, low
   `trainer.max_steps`, cheapest `GPU_TIER`).
2. Confirm the launch with the user if the run is expected to take
   more than a few minutes or if `GPU_TIER` is anything above
   `standard`.
3. `DRY_RUN=1` ./run-upstream-digress-sbm-modal.zsh prints the full
   command without launching — use this to sanity-check overrides.

## What Claude Code CAN and CANNOT do

- **Can:** deploy, launch, stream logs live, capture to file, read the
  file, grep tracebacks, propose a fix, edit code, redeploy, re-run.
- **Cannot:** view past-run logs (only in the Modal web dashboard),
  attach an interactive debugger inside a container, SSH into a
  running task (`modal shell` opens a *new* sandbox, not the failing
  task's container).

To substitute for a debugger: add `logging` / `print` / an explicit
`raise RuntimeError(f"...context...")` at the suspected site, redeploy,
re-run. Fast-fail behaviour is already the project default (see
`CLAUDE.md`), so unexpected values tend to crash loudly near the fault.
