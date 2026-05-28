# Experiment workflow

How to add, run, and document an experiment. Follow this every time so every run is
reproducible and the results index stays useful. (Summary lives in the root
`CLAUDE.md`; this is the detailed version.)

## 1. Create the experiment config

Copy an existing override file and edit it:

```
cp configs/exp/baseline.yaml configs/exp/<name>.yaml
```

Each file inherits the defaults and overrides only what changes:

```yaml
_base_: ../base.yaml

experiment:
  name: <name>                 # used in the run-dir name; keep it short, kebab/snake
  why: "One line: what question this run answers."
  method: "Backbone / neck / head in brief"   # shown in runs/README.md

model:
  neck:
    smooth_conv: gaconv        # only the keys you change
```

Rules:
- `name` must be filesystem-safe (letters, digits, `_`, `-`).
- Keep `why` to one sentence; keep `method` succinct (it goes in the index table).
- Only override keys that differ from `base.yaml`. Don't copy the whole base.

## 2. Run it

```
python -m obb_detector.train --config configs/exp/<name>.yaml
```

This creates `runs/<name>_<YYYY_MM_DD-HHmm>/` containing:

| File | Contents |
|---|---|
| `config.yaml` | fully-resolved config (re-run with this for exact reproduction) |
| `meta.json` | git commit + dirty flag, full command, start time, device |
| `NOTES.md` | why / setup / results / observations / conclusions |
| `checkpoints/` | `best.pth` (best val), `last.pth` (latest epoch) |
| `tb/` | TensorBoard logs (`tensorboard --logdir runs/<run>/tb`) |

On completion a row is appended to `runs/README.md`.

## 3. Document the outcome

After the run, edit `runs/<run>/NOTES.md`:
- **Results**: the headline metric (mAP50 / per-class AP, or val loss before DOTA),
  param/FLOP counts if relevant.
- **Observations**: failure modes, training stability, what surprised you.
- **Conclusions**: keep or drop the idea, and the next thing to try.

Then make sure the `runs/README.md` row is accurate (the auto-appended accuracy is a
placeholder — replace with the real metric once known). Keep that table succinct:
`run | accuracy | dataset | method`, no filler.

## 4. Reproduce or resume

Re-run an old experiment from its snapshot:

```
python -m obb_detector.train --config runs/<run>/config.yaml
```

Check `meta.json` for the git commit it was run at if results don't match.

## Conventions

- One idea per experiment file; ablations are sibling files (`gaconv_neck.yaml`,
  `gaconv_neck_theta_only.yaml`, ...), not edits to a shared file.
- `runs/` is gitignored except `runs/README.md` — checkpoints/logs stay local; the
  index is the shared record.
- New reusable code goes in `common/` or `obb_detector/`; register swappable
  components in `common/registry.py` so configs can select them by name.
