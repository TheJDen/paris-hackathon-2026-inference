# profiles/

Profiling artifacts captured while iterating on the engine.

## Naming convention (chrome traces — for Perfetto comparison)

Every torch.profiler artifact set is named so teammates can identify it
without opening it:

```
torch_<tag>_b<batch_size>_n<captured_new_tokens>_<sha>_<ts>.{json.gz,txt,summary.json}
```

Pieces:

- `tag` — phase + scenario, e.g. `phase1_microbatch`, `phase2_paged_kv`,
  `phase2_tp8`, `phase2_ep_inside_tp`. Set on the server CLI:
  `--profile-tag phase2_paged_kv`.
- `b<batch_size>` — number of sequences in the captured batch.
- `n<captured_new_tokens>` — generation tokens captured under
  `torch.profiler.profile` (NOT the same as the request's `max_tokens`;
  the engine bounds the captured slice to keep traces small).
- `sha` — first 7 chars of the git SHA the engine was built from. So
  any two profiles with different SHAs are A/B-comparable across commits.
- `ts` — capture timestamp in UTC, `YYYYMMDD-HHMMSS`.

Example:
```
torch_phase1_microbatch_b16_n32_8e3ea09_20260409-123238.json.gz
torch_phase1_microbatch_b16_n32_8e3ea09_20260409-123238.txt
torch_phase1_microbatch_b16_n32_8e3ea09_20260409-123238.summary.json
```

The three files share a stem so they sort together in `ls`.

## Three artifacts per capture

| File | Use it for |
|---|---|
| `*.json.gz` | Chrome trace — open in `chrome://tracing` or [Perfetto](https://ui.perfetto.dev). The visual timeline of CPU + CUDA work, with `time_region`/`record_function` ranges named exactly like the CLI region table. |
| `*.txt` | Sorted-by-self-CUDA-time op table with a header listing tag, sha, batch size, captured tokens, etc. The fastest "what's hot right now" read without opening a UI. |
| `*.summary.json` | Structured top-30 kernels — consumed by `bench/refresh_status.py` to embed the top hotspots in `STATUS.md`. Tiny, gitignore-friendly, committed alongside `STATUS.md`. |

## Comparing two profiles in Perfetto

1. Drop both `.json.gz` files into [ui.perfetto.dev](https://ui.perfetto.dev).
2. Use the **command bar** ("/") to filter by region name (e.g. `engine.model.generate`).
3. Sort by `self_cuda_time_total` in the slice details panel.
4. The filename tells you which is which without renaming.

## Throughput sweep artifacts

`throughput_<sha>_<ts>.json` — produced by `bench/quick_throughput`.
Per-level tok/s, partial weighted score, configuration. Consumed by
`bench/refresh_status.py` to render the throughput table in
`STATUS.md`.

## What's committed vs gitignored

**Committed** (so teammates can grab them from GitHub without ssh):
- This README
- All `torch_*.json.gz` chrome traces (~20-30 MB each, drop into Perfetto)
- All `*.summary.json` (small, drives STATUS.md across commits)
- All `throughput_*.json` (per-run sweep results)

**Gitignored**:
- Other large captures (nsys `.nsys-rep`, ncu `.ncu-rep`)
- Anything else under `profiles/`

If repo size becomes a problem (50+ pushes × 25 MB ≈ 1.25 GB), prune
old traces with `git filter-repo` or move them to git-lfs.

## How to grab a profile from GitHub

From the GitHub UI: navigate to the file under `profiles/`, hit
**Download**. The chrome trace lands locally as `.json.gz`. Drop the
.gz directly into [Perfetto](https://ui.perfetto.dev) — it understands
gzip natively.
