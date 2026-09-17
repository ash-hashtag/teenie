# Bounded arithmetic accuracy experiments

For the separate 99-example context experiment, see
[context learning](../docs/context-learning.md). Its known-task accuracy is
99.25%, but first unseen floor-division accuracy is only 9.52%; these must not
be confused with the single-pair numeric results below.

Inputs `a,b` in `[-10,10]`; all ADD/SUB/MUL operations; 1,323 examples with a
seeded 1,058/265 train/validation split. Validation labels are never used for
gradient updates, but are used for model/checkpoint selection. There is no third,
untouched test set; do not present these as independent test results or proof of
algorithmic reasoning. Seeds change both initialization and split.

## Results

| Run | Best validation | Selected epoch | Full grid (includes training) |
|---|---:|---:|---:|
| User's original embedding checkpoint, seed 0 | 113/265 = 42.64% | 1000 | — |
| Numeric 5k, seed 0 | 262/265 = 98.87% | 4500 | — |
| Numeric 15k, seed 0 | 263/265 = 99.25% | 8500 | 1321/1323 = 99.85% |
| Numeric 15k, seed 1 | 264/265 = 99.62% | 6000 | 1319/1323 = 99.70% |

The seed-0 15k configuration has two remaining full-grid errors: `(-10,10,MUL)`
and `(10,-10,MUL)` predict -98 instead of -100. Both are held out, and -100 never
occurs as a target in that seed's training set. The seed-1 checkpoint has one
validation error: `(-10,-10,MUL)` predicts 98 instead of 100. Full-grid evaluation
also includes training examples and must not be called generalization accuracy.

Addition and subtraction are 100% correct across the full grid for both selected
15k models. Multiplication still has boundary errors. No guarantee applies to
other seeds, ranges or operations. These are supervised all-operation runs, not
addition-only transfer experiments.

## Reproduce

```bash
uv run teenie --device cuda --representation numeric --ops add,sub,mul \
  --hidden 64 --max-steps 4 --no-ponder --batch-size 2048 \
  --epochs 15000 --lr-schedule cosine --log-every 500 --seed 0 \
  --save checkpoints/numeric-long-seed0.pt > experiments/numeric-long-seed0.jsonl
uv run teenie --load checkpoints/numeric-long-seed0.best.pt --eval-only
```

Repeat with `--seed 1` and separate seed-1 filenames. The 5k exploratory run used
`--epochs 5000 --log-every 250`, otherwise the same settings. Learning rate is
0.001 with cosine decay to 0.00001; AdamW weight decay is 0.0001. Best weights are
selected at logging intervals, with earlier epochs retained on ties. Models use
four fixed recurrent steps; adaptive halting is deliberately disabled.

## Artifacts

- `original-baseline.json`: user's original checkpoint reevaluated with per-op
  validation metrics; original `checkpoints/model.pt` was never overwritten.
- `numeric-seed0.jsonl`: 5k-epoch production run.
- `numeric-long-seed0.jsonl`, `numeric-long-seed1.jsonl`: full 15k run logs.
- `numeric-summary.json`: best checkpoints reloaded on CPU, exact validation and
  full-grid metrics, remaining validation inputs/targets/raw predictions.
- `accuracy_probe.py`, `accuracy-probe.jsonl`: preliminary CUDA-only comparison
  of a fixed-depth classifier and a numeric prototype. Run with
  `uv run python experiments/accuracy_probe.py`. The classifier received 1,000
  full-batch updates, numeric variants 5,000, so this is exploratory evidence,
  not an equal-budget ablation. The prototype is not the production model.
- `checkpoints/` (gitignored): locally saved final and best weights. Old embedding
  checkpoint format remains supported; no optimizer/RNG resume is claimed.

Numeric mode learns linear operand projections, a tanh recurrent cell and scalar
output. It does not compute reference arithmetic inside the model or augment
training using validation answers. Predictions are rescaled, rounded and bounded;
accuracy requires exact integer equality.
