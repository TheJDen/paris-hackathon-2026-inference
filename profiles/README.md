# profiles/

Profiling artifacts captured while iterating on the engine.

## Naming convention

```
{tool}_{phase}_{scenario}_{git-sha}_{timestamp}.{ext}
```

- **tool**: `regions` (CLI region timer dump), `torch` (torch.profiler), `nsys`, `ncu`, `helion`
- **phase**: `phase0`, `phase1`, ..., `phase5`
- **scenario**: short description — `c8`, `c64`, `moe_swap_v2`, `ep_vs_replicated`
- **git-sha**: first 7 chars of `git rev-parse HEAD`
- **timestamp**: `YYYYMMDD-HHMMSS`
- **ext**: `txt` (region tables), `json.gz` (chrome traces), `nsys-rep`, `ncu-rep`, `json` (helion autotune)

Examples:
```
regions_phase1_c8_a1b2c3d_20260409-153012.txt
torch_phase3_moe_swap_v2_e4f5a6b_20260411-091200.json.gz
nsys_phase4_ep_vs_replicated_8d9e0f1_20260412-160045.nsys-rep
```

## What to capture when

Default loop is **CLI region tables only** — they're cheap and answer
~90% of "what got faster / slower" questions:

```bash
curl -s localhost:8000/metrics/regions > profiles/regions_phaseX_<scenario>_<sha>_$(date +%Y%m%d-%H%M%S).txt
```

Reach for `torch.profiler` (`--profile-torch --profile-window=10:20 --profile-tag <name>`)
**only** when the region table has identified a region of interest and
you need to see *inside* it.

Reach for `nsys` / `ncu` only when torch.profiler can't show you what
you need (NCCL timelines, CUDA-graph internals, kernel-level
occupancy/stalls).

The artifact bodies in this directory are **gitignored** — only this
README is tracked. Keep the convention so a `before / after` diff is
trivial: matching scenario tag + adjacent timestamps.
