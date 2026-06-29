"""Runner — orchestrates a single benchmark run and writes the results tree.

Drives the provider lifecycle with guaranteed cleanup (try/finally) and
produces, under results/cloud/<timestamp>-<provider>-<bench>/:

    metadata.json           run metadata (see build_metadata)
    stdout.log
    stderr.log
    benchmark_result.json   if the benchmark produced one
    provider.json           resource IDs, instance type, region, GPU count
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
from pathlib import Path
from typing import Optional

from .config import REPO_ROOT, RunConfig
from .providers import ExecResult, ResourceInfo, get_provider


# --------------------------------------------------------------------------- #
# Git metadata
# --------------------------------------------------------------------------- #

def _git(*args: str) -> Optional[str]:
    try:
        proc = subprocess.run(["git", *args], cwd=str(REPO_ROOT),
                              capture_output=True, text=True, timeout=15)
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return None


def git_info() -> dict:
    commit = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    # Dirty diff summary (stat only — keep it small).
    diff_stat = _git("diff", "--stat")
    dirty = bool(_git("status", "--porcelain"))
    return {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
        "diff_summary": (diff_stat or "").splitlines()[-1].strip()
        if diff_stat else "",
        "diff_stat": diff_stat or "",
    }


# --------------------------------------------------------------------------- #
# Output writing
# --------------------------------------------------------------------------- #

def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def build_metadata(
    cfg: RunConfig,
    git: dict,
    resource: ResourceInfo,
    result: ExecResult,
    start: _dt.datetime,
    end: _dt.datetime,
    cleanup_ok: bool,
) -> dict:
    return {
        "provider": cfg.provider,
        "bench": cfg.spec.name,
        "description": cfg.spec.description,
        "command": cfg.command,
        "base_command": cfg.base_command,
        "command_overridden": cfg.base_command != cfg.spec.command,
        "profile": cfg.profile,
        "profile_args": cfg.profile_args,
        "artifacts": result.artifacts,
        "git_commit": git.get("commit"),
        "git_branch": git.get("branch"),
        "git_dirty": git.get("dirty"),
        "git_diff_summary": git.get("diff_summary"),
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "duration_seconds": round((end - start).total_seconds(), 2),
        "gpu": cfg.spec.gpu,
        "ngpus": cfg.spec.ngpus,
        "instance_id": resource.instance_id,
        "instance_type": resource.instance_type,
        "region": resource.region,
        "resource_name": cfg.resource_name,
        "exit_code": result.exit_code,
        "cleanup_succeeded": cleanup_ok,
        "keep_instance": cfg.keep_instance,
        "dry_run": cfg.dry_run,
        "sync_mode": cfg.sync_mode,
        "registry_image": cfg.registry_image,
        "results_dir": str(cfg.run_dir),
    }


def write_results(
    cfg: RunConfig,
    git: dict,
    resource: ResourceInfo,
    result: ExecResult,
    start: _dt.datetime,
    end: _dt.datetime,
    cleanup_ok: bool,
) -> dict:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)

    (cfg.run_dir / "stdout.log").write_text(result.stdout or "")
    (cfg.run_dir / "stderr.log").write_text(result.stderr or "")

    if result.benchmark_result:
        # Store verbatim; pretty-print if it parses as JSON.
        try:
            parsed = json.loads(result.benchmark_result)
            _write_json(cfg.run_dir / "benchmark_result.json", parsed)
        except (json.JSONDecodeError, TypeError):
            (cfg.run_dir / "benchmark_result.json").write_text(result.benchmark_result)

    _write_json(cfg.run_dir / "provider.json", resource.to_dict())

    metadata = build_metadata(cfg, git, resource, result, start, end, cleanup_ok)
    _write_json(cfg.run_dir / "metadata.json", metadata)
    return metadata


# --------------------------------------------------------------------------- #
# Signal handling — turn SIGTERM/SIGHUP into a normal exception so the
# try/finally cleanup runs, and shield cleanup from a second interrupt.
# --------------------------------------------------------------------------- #

import signal as _signal  # noqa: E402


def _install_term_handlers() -> dict:
    """Make SIGTERM/SIGHUP raise KeyboardInterrupt so `finally` cleanup runs.

    Returns the previous handlers so they can be restored. Best-effort: only
    works in the main thread (signal.signal raises ValueError otherwise).
    """
    def _handler(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    previous = {}
    for sig_name in ("SIGTERM", "SIGHUP"):
        sig = getattr(_signal, sig_name, None)
        if sig is None:
            continue
        try:
            previous[sig] = _signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass
    return previous


def _restore_handlers(previous: dict) -> None:
    for sig, handler in previous.items():
        try:
            _signal.signal(sig, handler)
        except (ValueError, OSError):
            pass


class _shield_cleanup:
    """Context manager that ignores SIGINT/SIGTERM/SIGHUP for its duration so a
    panicked second Ctrl-C cannot abort an in-flight destroy and leak a GPU."""

    def __enter__(self):
        self._saved = {}
        for sig_name in ("SIGINT", "SIGTERM", "SIGHUP"):
            sig = getattr(_signal, sig_name, None)
            if sig is None:
                continue
            try:
                self._saved[sig] = _signal.signal(sig, _signal.SIG_IGN)
            except (ValueError, OSError):
                pass
        return self

    def __exit__(self, *exc):
        _restore_handlers(self._saved)
        return False


def warn_if_stale_ledger() -> None:
    """Warn loudly if the ledger lists resources that may still be alive."""
    from . import ledger
    records = ledger.list_records()
    if not records:
        return
    print("!" * 70)
    print(f"WARNING: {len(records)} resource(s) in the active ledger may still "
          f"be running (possible orphans from a previous hard kill):")
    for rec in records:
        print(f"  - {rec.get('provider')}:{rec.get('resource_name')} "
              f"(id={rec.get('instance_id')}, kept={rec.get('keep')})")
    print("Run `make reap` (or `python -m cloudbench.cli reap`) to destroy them.")
    print("!" * 70)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run(cfg: RunConfig) -> int:
    """Execute one benchmark run. Returns the benchmark process exit code."""
    git = git_info()
    provider = get_provider(cfg.provider, cfg)
    if not cfg.dry_run:
        warn_if_stale_ledger()
        # Create the output dir up front so providers can stream artifacts in.
        cfg.run_dir.mkdir(parents=True, exist_ok=True)

    if cfg.profile == "ncu" and cfg.spec.ngpus > 1:
        print("[cloudbench] WARNING: ncu wraps a single process; on a multi-GPU "
              "command (e.g. torchrun) it usually won't profile the workers. "
              "Use PROFILE=nsys for multi-GPU runs.")

    print("=" * 70)
    print(f"cloudbench: {cfg.provider} / {cfg.spec.name}")
    print(f"  command : {cfg.command}")
    print(f"  gpu     : {cfg.spec.ngpus}x{cfg.spec.gpu}")
    if cfg.profile != "none":
        print(f"  profile : {cfg.profile}")
    print(f"  resource: {cfg.resource_name}")
    print(f"  output  : {cfg.run_dir}")
    if cfg.dry_run:
        print("  MODE    : DRY_RUN (no resources will be created)")
    print("=" * 70)

    start = _dt.datetime.now()
    result = ExecResult(exit_code=1, stderr="run did not complete")
    cleanup_ok = True
    interrupted = False

    previous_handlers = _install_term_handlers()
    try:
        try:
            provider.provision()
            result = provider.execute()
        except KeyboardInterrupt:
            interrupted = True
            print("\n[cloudbench] interrupted — tearing down resources "
                  "(do NOT force-kill; let cleanup finish)...")
            result = ExecResult(
                exit_code=130,
                stdout=result.stdout if result else "",
                stderr="[cloudbench] run interrupted by signal/Ctrl-C",
            )
        except Exception as exc:  # noqa: BLE001 - capture so cleanup still runs
            import traceback
            result = ExecResult(
                exit_code=1,
                stdout=result.stdout if result else "",
                stderr=f"[cloudbench] run failed: {exc}\n{traceback.format_exc()}",
            )
            print(result.stderr)
        finally:
            # Shield destroy from a panicked second Ctrl-C / further signals.
            with _shield_cleanup():
                try:
                    cleanup_ok = provider.cleanup()
                except Exception as exc:  # noqa: BLE001
                    cleanup_ok = False
                    print(f"[cloudbench] cleanup raised: {exc}")
    finally:
        _restore_handlers(previous_handlers)

    end = _dt.datetime.now()
    metadata = write_results(cfg, git, provider.resource, result, start, end, cleanup_ok)

    print("-" * 70)
    print(f"exit code        : {result.exit_code}")
    print(f"duration         : {metadata['duration_seconds']}s")
    print(f"cleanup succeeded: {cleanup_ok}")
    print(f"results          : {cfg.run_dir}")
    if result.artifacts:
        print(f"artifacts        : {len(result.artifacts)} file(s) in "
              f"{cfg.artifacts_dir}/")
    if cfg.keep_instance and not cfg.dry_run:
        hint = provider.ssh_instructions()
        if hint:
            print("KEEP_INSTANCE=1 — connect with:\n    " + hint)
    print("=" * 70)

    if not cleanup_ok:
        print("!" * 70)
        print("CLEANUP DID NOT SUCCEED — a cloud resource may still be running.")
        print(f"Destroy it now: `make reap`  (resource: {cfg.resource_name})")
        print("!" * 70)

    # Non-zero if benchmark failed OR cleanup failed (so CI notices leaks).
    if not cleanup_ok:
        return 3
    if interrupted:
        return 130
    return result.exit_code
