"""Per-run experiment directories and the results index.

Each run gets a self-contained dir ``runs/<name>_<YYYY_MM_DD-HHmm>/`` holding the
resolved config, a NOTES.md (why/results/observations/conclusions), a meta.json
(git commit, command, time), a checkpoints/ dir and a tb/ dir for TensorBoard.
``runs/README.md`` is a curated one-line-per-run results index.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

NOTES_TEMPLATE = """# {name}

## Why
{why}

## Setup
- config: `config.yaml` (snapshot in this dir)
- method: {method}
- dataset: {dataset}

## Results
_fill after the run (mAP / per-class AP / val loss)_

## Observations
_what stood out: failure modes, training behaviour_

## Conclusions
_keep or drop this idea, and why_
"""

RESULTS_HEADER = (
    "# Experiment results\n\n"
    "| Run | Accuracy | Dataset | Method |\n"
    "|---|---|---|---|\n"
)


def _git(args):
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, cwd=Path.cwd()
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def create_run_dir(cfg, device=None):
    """Create and populate a fresh run directory; return its Path."""
    name = cfg.experiment.name
    ts = datetime.now().strftime("%Y_%m_%d-%H%M")
    run_dir = Path(cfg.paths.runs_dir) / f"{name}_{ts}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "tb").mkdir(parents=True, exist_ok=True)

    cfg.save(run_dir / "config.yaml")

    commit = _git(["rev-parse", "HEAD"])
    meta = {
        "git_commit": commit,
        "git_dirty": bool(_git(["status", "--porcelain"])) if commit else None,
        "command": " ".join(sys.argv),
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "device": str(device) if device is not None else None,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    (run_dir / "NOTES.md").write_text(NOTES_TEMPLATE.format(
        name=name,
        why=cfg.experiment.get("why", ""),
        method=cfg.experiment.get("method", ""),
        dataset=cfg.data.root,
    ))
    return run_dir


def append_results_row(runs_dir, run_name, accuracy, dataset, method):
    """Append one row to runs/README.md, creating the header if missing."""
    index = Path(runs_dir) / "README.md"
    if not index.exists():
        index.write_text(RESULTS_HEADER)
    with index.open("a") as f:
        f.write(f"| {run_name} | {accuracy} | {dataset} | {method} |\n")
