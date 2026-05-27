# Roadmap

Open work on the primary OBB detector. These are deliberate design/tuning choices,
not bugs.

## Tuning / training

- **Anchor assignment**: currently a naive best-IoU > 0.5 rule. Upgrade to an
  adaptive scheme (e.g. ATSS) for better positive/negative balance.
- **Loss weighting**: classification, regression, and objectness are summed with
  equal weight. Add per-term weighting / balancing.
- **Anchor hyperparameters**: per-level scales, ratios, and angles are a starting
  point and should be tuned against baseline results.

## Possible architectural directions (post-baseline)

To be decided after a baseline mAP is established and failure modes are analysed:

1. An orientation-aware neck (e.g. strip convolutions in the FPN fusion nodes).
2. A decoupled head (rotation-invariant classification + rotation-sensitive
   regression branches).
3. Improved angle regression (e.g. GWD/KLD losses, which add no parameters).
