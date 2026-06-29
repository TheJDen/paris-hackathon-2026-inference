"""cloudbench CLI.

    python -m cloudbench.cli list
    python -m cloudbench.cli run --provider local    --bench decode_smoke
    python -m cloudbench.cli run --provider sesterce --bench single_h200_decode
    python -m cloudbench.cli run --provider nebius   --bench e2e_8xh200
    python -m cloudbench.cli run --provider modal    --bench decode_smoke
"""

from __future__ import annotations

import argparse
import sys

from . import ledger
from .benchmark_registry import get_benchmark, load_benchmarks
from .config import build_run_config, load_env
from .providers import PROVIDERS, reap_record
from .runner import run as run_benchmark


def _cmd_list(_args: argparse.Namespace) -> int:
    benches = load_benchmarks()
    if not benches:
        print("No benchmarks defined in configs/benchmarks.yaml")
        return 0
    name_w = max(len(n) for n in benches)
    print(f"{'BENCH'.ljust(name_w)}  {'GPU':<10} {'N':>2}  {'TIMEOUT':>7}  DESCRIPTION")
    print("-" * 80)
    for name in sorted(benches):
        b = benches[name]
        gpu = f"{b.gpu}" if b.needs_gpu else "none"
        print(f"{name.ljust(name_w)}  {gpu:<10} {b.ngpus:>2}  "
              f"{str(b.timeout_minutes) + 'm':>7}  {b.description}")
    print()
    print(f"Providers: {', '.join(PROVIDERS)}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    env = load_env()
    if args.provider not in PROVIDERS:
        print(f"error: unknown provider {args.provider!r}. "
              f"Choose: {', '.join(PROVIDERS)}", file=sys.stderr)
        return 2
    try:
        spec = get_benchmark(args.bench)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    cfg = build_run_config(
        provider=args.provider,
        spec=spec,
        bench_cmd=args.bench_cmd,
        env=env,
    )
    return run_benchmark(cfg)


def _cmd_reap(args: argparse.Namespace) -> int:
    """Destroy every cloud resource still listed in the active ledger.

    This is the backstop for orphans left by a hard kill (SIGKILL, power loss,
    closed laptop) where in-process cleanup never ran.
    """
    env = load_env()
    records = ledger.list_records()
    if not records:
        print("Ledger is clean — no active resources to reap.")
        return 0

    targets = [r for r in records if args.include_kept or not r.get("keep")]
    kept = [r for r in records if r.get("keep") and not args.include_kept]
    for r in kept:
        print(f"[reap] skipping KEEP_INSTANCE resource "
              f"{r.get('provider')}:{r.get('resource_name')} (use --include-kept)")

    if not targets:
        print("Nothing to reap (only kept resources present).")
        return 0

    failures = 0
    for rec in targets:
        name = rec.get("resource_name")
        print(f"[reap] {rec.get('provider')}:{name} ...")
        try:
            ok = reap_record(rec, env, dry_run=args.dry_run)
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"[reap] error: {exc}")
        if not ok:
            failures += 1
            print(f"[reap] FAILED to destroy {name} — destroy it manually!")
    if failures:
        print(f"[reap] {failures} resource(s) could not be destroyed.")
        return 3
    if args.dry_run:
        print("[reap] dry-run complete — nothing destroyed.")
    else:
        print("[reap] done — ledger clean.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cloudbench",
        description="Provider-independent benchmark harness "
                    "(BENCH = what, PROVIDER = where).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list available benchmarks and providers")
    p_list.set_defaults(func=_cmd_list)

    p_run = sub.add_parser("run", help="run a benchmark on a provider")
    p_run.add_argument("--provider", required=True,
                       help=f"where to run: {', '.join(PROVIDERS)}")
    p_run.add_argument("--bench", required=True, help="which benchmark to run")
    p_run.add_argument("--bench-cmd", default=None,
                       help="override the benchmark command")
    p_run.set_defaults(func=_cmd_run)

    p_reap = sub.add_parser(
        "reap", help="destroy any cloud resources left in the active ledger")
    p_reap.add_argument("--dry-run", action="store_true",
                        help="list what would be destroyed without doing it")
    p_reap.add_argument("--include-kept", action="store_true",
                        help="also destroy resources left alive via KEEP_INSTANCE=1")
    p_reap.set_defaults(func=_cmd_reap)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
