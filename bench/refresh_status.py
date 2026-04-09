"""Pre-push status refresh.

One command. Run it before every push to main.

It does:
  1. Fires a quick throughput sweep against the running engine (so the
     numbers are always fresh).
  2. Fetches /metrics + /metrics/regions for the live engine snapshot.
  3. Picks up the freshest torch.profiler artifact in profiles/ and
     extracts the top kernels by self CUDA time.
  4. Rewrites STATUS.md with: commit, implementation summary, throughput
     numbers, region table, top kernel hotspots, profile artifact paths.

The implementation summary is read from `bench/.status_intro.md` (a
short, manually-maintained file). Update it when the engine architecture
changes — that's the only piece a human owns. Everything else is regenerated.

The torch profile is captured by the engine itself in one-shot mode. To
arm it, launch the server with `--profile-torch-after-batches N` so it
captures batch N+1. The refresh script does NOT bounce the server.

Usage (server is already up):
    python -m bench.refresh_status --base-url http://localhost:8765
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import shlex
import subprocess
import sys
import urllib.error
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATUS_PATH = os.path.join(REPO_ROOT, "STATUS.md")
PROFILES_DIR = os.path.join(REPO_ROOT, "profiles")
INTRO_PATH = os.path.join(REPO_ROOT, "bench", ".status_intro.md")


# --------------------------------------------------------------------------- #
# small fetch helpers
# --------------------------------------------------------------------------- #


def _http_get(url: str, timeout: int = 10) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.URLError:
        return None


def fetch_metrics(base_url: str) -> dict | None:
    body = _http_get(f"{base_url.rstrip('/')}/metrics")
    if not body:
        return None
    try:
        return json.loads(body)
    except Exception:
        return None


def fetch_regions(base_url: str) -> str | None:
    return _http_get(f"{base_url.rstrip('/')}/metrics/regions")


# --------------------------------------------------------------------------- #
# git
# --------------------------------------------------------------------------- #


def git_info() -> dict:
    def _git(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return ""

    return {
        "sha": _git("rev-parse", "--short", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "subject": _git("log", "-1", "--pretty=%s"),
        "body": _git("log", "-1", "--pretty=%b"),
        "author_date": _git("log", "-1", "--pretty=%ad", "--date=short"),
    }


# --------------------------------------------------------------------------- #
# profile artifact picker
# --------------------------------------------------------------------------- #


def latest_profile() -> dict | None:
    """Find the most recent torch profile summary in profiles/."""
    summary_glob = os.path.join(PROFILES_DIR, "torch_*.summary.json")
    matches = sorted(glob.glob(summary_glob), key=os.path.getmtime, reverse=True)
    if not matches:
        return None
    summary_path = matches[0]
    with open(summary_path) as f:
        data = json.load(f)
    base = summary_path[: -len(".summary.json")]
    return {
        "summary_json": summary_path,
        "summary_txt": base + ".txt" if os.path.exists(base + ".txt") else None,
        "chrome_trace": base + ".json.gz" if os.path.exists(base + ".json.gz") else None,
        "top_kernels": data.get("top_kernels", []),
        "meta": data.get("meta", {}),
        "captured_at": _dt.datetime.fromtimestamp(os.path.getmtime(summary_path)).isoformat(timespec="seconds"),
    }


# --------------------------------------------------------------------------- #
# latest throughput artifact
# --------------------------------------------------------------------------- #


def latest_throughput() -> dict | None:
    matches = sorted(
        glob.glob(os.path.join(PROFILES_DIR, "throughput_*.json")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not matches:
        return None
    path = matches[0]
    with open(path) as f:
        data = json.load(f)
    data["__path__"] = os.path.relpath(path, REPO_ROOT)
    data["__captured_at__"] = _dt.datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds")
    return data


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


PHASE_1_FLOOR = {16: 350.0, 32: 694.0, 64: 525.0, "weighted": 8380.0}


def _read_intro() -> str:
    if not os.path.exists(INTRO_PATH):
        return (
            "_(no `bench/.status_intro.md` found — write a 2-4 sentence "
            "description of the current engine architecture there and re-run.)_"
        )
    with open(INTRO_PATH) as f:
        return f.read().strip()


def _format_throughput_table(t: dict | None) -> str:
    if not t:
        return "_(no throughput artifact in `profiles/` — run `python -m bench.quick_throughput` first)_"

    rows = t.get("results", [])
    weights = {1: 1, 2: 1, 4: 2, 8: 2, 16: 4, 32: 4, 64: 8}
    lines = [
        "| concurrency | tok/s | wall_s | reqs ok | weight | vs Phase 1 floor |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        c = r.get("concurrency")
        if c is None:
            continue
        tps = float(r.get("throughput_tok_per_sec", 0.0))
        wall = float(r.get("wall_time_sec", 0.0))
        ok = r.get("successful_requests", 0)
        tot = ok + r.get("failed_requests", 0)
        w = weights.get(c, 0)
        floor = PHASE_1_FLOOR.get(c)
        delta = ""
        if floor:
            ratio = tps / floor
            arrow = "↑" if ratio > 1.05 else ("↓" if ratio < 0.95 else "·")
            delta = f"{ratio*100:.0f}% {arrow}"
        lines.append(f"| {c} | {tps:.0f} | {wall:.1f} | {ok}/{tot} | {w}× | {delta} |")

    score = t.get("partial_weighted_score", 0.0)
    weight = t.get("weight_covered", 0)
    cfg = t.get("config", {})
    lines.append(
        f"\n**Partial weighted score** ({weight}/22 weight covered): "
        f"**{score:.0f}**  (Phase 1 floor: {PHASE_1_FLOOR['weighted']:.0f})"
    )
    if cfg:
        lines.append(
            f"\n_config: c={cfg.get('concurrency')} reqs/level={cfg.get('num_requests_per_level')} "
            f"isl={cfg.get('input_tokens')} osl={cfg.get('output_tokens')}_"
        )
    return "\n".join(lines)


def _format_regions(text: str | None) -> str:
    if not text:
        return "_(server `/metrics/regions` not reachable)_"
    return f"```\n{text.strip()}\n```"


def _format_top_kernels(prof: dict | None, n: int = 12) -> str:
    if not prof:
        return (
            "_(no torch profile in `profiles/` yet — launch the server with "
            "`--profile-torch-after-batches 3` so it one-shot-captures batch 4)_"
        )
    rows = prof.get("top_kernels", [])[:n]
    if not rows:
        return "_(profile artifact had no kernels)_"
    lines = [
        f"_captured: {prof.get('captured_at', '?')}_",
        "",
        "| # | kernel | self CUDA (ms) | self CPU (ms) | calls |",
        "|---:|---|---:|---:|---:|",
    ]
    for i, k in enumerate(rows, 1):
        cuda_ms = k["self_cuda_us"] / 1000.0
        cpu_ms = k["self_cpu_us"] / 1000.0
        # Truncate long names
        name = k["name"]
        if len(name) > 60:
            name = name[:57] + "..."
        lines.append(f"| {i} | `{name}` | {cuda_ms:.1f} | {cpu_ms:.1f} | {k['count']} |")
    if prof.get("chrome_trace"):
        lines.append(f"\n_chrome trace: `{os.path.relpath(prof['chrome_trace'], REPO_ROOT)}`_")
    if prof.get("summary_txt"):
        lines.append(f"_full op table: `{os.path.relpath(prof['summary_txt'], REPO_ROOT)}`_")
    if prof.get("meta"):
        meta = prof["meta"]
        lines.append(f"\n_capture meta: batch_size={meta.get('batch_size')} "
                     f"input_padded_len={meta.get('input_padded_len')} "
                     f"max_new_tokens={meta.get('max_new_tokens')}_")
    return "\n".join(lines)


def _format_metrics(m: dict | None) -> str:
    if not m:
        return "_(server `/metrics` not reachable)_"
    keys_in_order = [
        "tok_per_s_recent",
        "batch_tok_per_s_p50",
        "batch_tok_per_s_p99",
        "avg_batch_size",
        "avg_batch_fill",
        "max_batch",
        "running",
        "waiting",
        "requests_total",
        "requests_failed",
        "uptime_s",
    ]
    rows = []
    for k in keys_in_order:
        if k in m:
            rows.append(f"  - `{k}`: {m[k]}")
    return "\n".join(rows) if rows else "_(metrics empty)_"


def render_status(
    *,
    git: dict,
    intro: str,
    throughput: dict | None,
    regions: str | None,
    metrics_snap: dict | None,
    profile: dict | None,
) -> str:
    now = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    parts = []
    parts.append("# STATUS")
    parts.append("")
    parts.append(f"_auto-generated by `bench/refresh_status.py` — refresh before every push_")
    parts.append("")
    parts.append(f"**commit:** `{git.get('sha', '?')}` on `{git.get('branch', '?')}` ({git.get('author_date', '')})  ")
    parts.append(f"**subject:** {git.get('subject', '')}  ")
    parts.append(f"**refreshed:** {now}")
    parts.append("")
    parts.append("## Implementation")
    parts.append("")
    parts.append(intro)
    parts.append("")
    parts.append("## Throughput (the number we're scored on)")
    parts.append("")
    parts.append(_format_throughput_table(throughput))
    parts.append("")
    if throughput and throughput.get("__path__"):
        parts.append(f"_artifact: `{throughput['__path__']}`_")
        parts.append("")
    parts.append("## Live engine `/metrics`")
    parts.append("")
    parts.append(_format_metrics(metrics_snap))
    parts.append("")
    parts.append("## CLI region table")
    parts.append("")
    parts.append(_format_regions(regions))
    parts.append("")
    parts.append("## Top kernel hotspots (torch.profiler)")
    parts.append("")
    parts.append(_format_top_kernels(profile))
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("Iteration loop is documented in `HACKING.md`. Phase plan in `docs/parallelism.md`.")
    return "\n".join(parts) + "\n"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refresh STATUS.md from a running engine")
    p.add_argument("--base-url", default="http://localhost:8765")
    p.add_argument(
        "--skip-throughput",
        action="store_true",
        help="don't run quick_throughput, just regenerate STATUS.md from existing artifacts",
    )
    p.add_argument(
        "--quick-throughput-args",
        default="",
        help="extra args passed through to bench.quick_throughput",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_throughput:
        # Run quick_throughput as a subprocess so its output streams live.
        sha = git_info().get("sha", "unknown")
        ts = _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        out = os.path.join(PROFILES_DIR, f"throughput_{sha}_{ts}.json")
        os.makedirs(PROFILES_DIR, exist_ok=True)
        cmd = [
            sys.executable, "-m", "bench.quick_throughput",
            "--base-url", args.base_url,
            "--output", out,
        ]
        if args.quick_throughput_args:
            cmd.extend(shlex.split(args.quick_throughput_args))
        print(f"[refresh] running: {' '.join(cmd)}")
        rc = subprocess.call(cmd, cwd=REPO_ROOT)
        if rc != 0:
            print(f"[refresh] quick_throughput exited {rc} — STATUS may be stale", file=sys.stderr)

    git = git_info()
    intro = _read_intro()
    throughput = latest_throughput()
    metrics_snap = fetch_metrics(args.base_url)
    regions = fetch_regions(args.base_url)
    profile = latest_profile()

    body = render_status(
        git=git,
        intro=intro,
        throughput=throughput,
        regions=regions,
        metrics_snap=metrics_snap,
        profile=profile,
    )
    with open(STATUS_PATH, "w") as f:
        f.write(body)

    print(f"\n[refresh] wrote {STATUS_PATH}")
    print()
    print(body)


if __name__ == "__main__":
    main()
