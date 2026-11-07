# Teenie docs site

Astro + TypeScript, static output. Pages: Overview, Journey, Model (diagram + replay of
real predictions), Learning (curves, per-relation heatmap, checkpoint replay), Benchmarks.

```bash
pnpm install
pnpm dev      # http://localhost:4321
pnpm build    # astro check + static build into dist/
```

All numbers come from the trained model. To refresh them after retraining (run from the repo root):

```bash
uv run teenie-bits --device cuda --updates 3000 --log-every 100 --save checkpoints/bits.pt \
  --snapshots checkpoints/bits-snapshots --log experiments/bits-seed0-train.jsonl
uv run teenie-bits-eval --load checkpoints/bits.pt --device cuda --episodes 64 --seed 60000 \
  --output experiments/bits-seed0-eval.json
uv run python -m teenie.bits.export   # writes site/src/data/*.json and site/public/data/*.json
```

`astro check` needs TypeScript 6 (pinned); `pnpm-workspace.yaml` pins `@emnapi/runtime` to a
stable release because the 2.0 beta crashes the checker.
