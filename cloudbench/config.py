"""Configuration: env loading, dataclasses, naming, and run-level options.

Everything provider-independent lives here so the runner and the provider
adapters share a single source of truth.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

# Repo root = parent of the cloudbench package directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "cloud"
CONFIGS_DIR = REPO_ROOT / "configs"
BENCHMARKS_YAML = CONFIGS_DIR / "benchmarks.yaml"
ENV_FILE = REPO_ROOT / ".env"                       # SECRETS only (gitignored)
PROVIDERS_CONF = CONFIGS_DIR / "providers.conf"     # NON-secret config (gitignored)


# --------------------------------------------------------------------------- #
# .env loading (no python-dotenv dependency)
# --------------------------------------------------------------------------- #

def _parse_dotenv(path: Path) -> dict:
    """Parse a KEY=value file (shared format for .env and providers.conf)."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if val and val[0] in "\"'":
            # Quoted: take the quoted span, ignore any trailing comment.
            quote = val[0]
            end = val.find(quote, 1)
            val = val[1:end] if end != -1 else val[1:]
        else:
            # Unquoted: strip an inline ' #...' comment, then trailing space.
            hash_idx = val.find(" #")
            if hash_idx != -1:
                val = val[:hash_idx]
            val = val.rstrip()
        values[key] = val
    return values


def load_env(env_file: Path = ENV_FILE, providers_conf: Path = PROVIDERS_CONF) -> dict:
    """Merge config from three layers, lowest precedence first:

        1. configs/providers.conf  — NON-secret settings (regions, image ids, ...)
        2. .env                    — SECRETS only (API keys/tokens)
        3. the real environment    — inline overrides (KEEP_INSTANCE=1 make ...)

    Splitting non-secret config out of .env keeps your API key sitting alone,
    and lets the non-secret file be populated/checked separately.
    """
    values: dict[str, str] = {}
    values.update(_parse_dotenv(providers_conf))  # non-secret base
    values.update(_parse_dotenv(env_file))        # secrets overlay
    for key, val in os.environ.items():           # inline overrides win
        values[key] = val
    return values


def env_get(env: dict, key: str, default: Optional[str] = None) -> Optional[str]:
    val = env.get(key, default)
    if val is None:
        return None
    val = str(val).strip()
    return val if val != "" else default


def env_flag(env: dict, key: str) -> bool:
    """A flag is truthy if set to 1/true/yes/on (case-insensitive)."""
    val = env_get(env, key)
    return bool(val) and val.lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------- #
# Benchmark spec
# --------------------------------------------------------------------------- #

@dataclass
class BenchmarkSpec:
    name: str
    command: str
    gpu: str = "none"
    ngpus: int = 0
    timeout_minutes: int = 30
    description: str = ""

    @property
    def needs_gpu(self) -> bool:
        return self.ngpus > 0 and self.gpu.lower() not in ("", "none")

    @property
    def timeout_seconds(self) -> int:
        return int(self.timeout_minutes) * 60


# --------------------------------------------------------------------------- #
# Run configuration (the resolved "what + where + how" of a single run)
# --------------------------------------------------------------------------- #

@dataclass
class RunConfig:
    provider: str
    spec: BenchmarkSpec
    command: str  # may differ from spec.command if overridden via --bench-cmd
    env: dict = field(default_factory=dict)

    # Output
    timestamp: str = ""
    run_dir: Path = field(default_factory=Path)

    # Lifecycle flags
    keep_instance: bool = False
    dry_run: bool = False
    sync_mode: str = "rsync"  # "rsync" | "docker"
    registry_image: Optional[str] = None
    docker_install: bool = False  # best-effort install docker on the instance

    # Profiling
    profile: str = "none"           # "none" | "nsys" | "ncu"
    profile_args: Optional[str] = None
    artifacts_dir: str = "artifacts"  # dir (relative to workdir) collected after a run
    base_command: str = ""          # the command before profiler wrapping

    # Misc
    user: str = "user"

    @property
    def resource_name(self) -> str:
        # paris-bench-${USER}-${timestamp}-${bench}
        return f"paris-bench-{self.user}-{self.timestamp}-{self.spec.name}"


def make_timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def current_user(env: dict) -> str:
    raw = env_get(env, "USER") or env_get(env, "USERNAME") or "user"
    # Sanitize so it's safe inside a resource name.
    return re.sub(r"[^a-zA-Z0-9_-]", "-", raw).lower()[:32] or "user"


PROFILE_MODES = ("none", "nsys", "ncu")


def wrap_command_for_profile(
    command: str,
    profile: str,
    profile_args: Optional[str],
    artifacts_dir: str,
    resource_name: str,
) -> str:
    """Prefix the benchmark command with nsys/ncu so its report lands in the
    artifacts dir (which the harness downloads). Returns command unchanged for
    profile='none'. Chrome/Perfetto traces are NOT produced here — your code
    emits those; the harness just collects them from the artifacts dir.
    """
    if profile in (None, "", "none"):
        return command
    if profile not in PROFILE_MODES:
        raise ValueError(f"PROFILE must be one of {PROFILE_MODES}, got {profile!r}")

    out_base = f"{artifacts_dir}/{resource_name}"
    if profile == "nsys":
        # -> artifacts/<name>_nsys.nsys-rep
        prefix = (f"nsys profile --force-overwrite true -o {out_base}_nsys "
                  f"--trace=cuda,nvtx,cublas,cudnn,osrt")
    else:  # ncu  -> artifacts/<name>_ncu.ncu-rep
        prefix = f"ncu --set full -f -o {out_base}_ncu"
    if profile_args:
        prefix += " " + profile_args
    return f"{prefix} {command}"


def build_run_config(
    provider: str,
    spec: BenchmarkSpec,
    bench_cmd: Optional[str] = None,
    env: Optional[dict] = None,
) -> RunConfig:
    """Resolve all run options from env + arguments into a RunConfig."""
    env = env if env is not None else load_env()
    timestamp = make_timestamp()
    user = current_user(env)

    base_command = bench_cmd or env_get(env, "BENCH_CMD") or spec.command

    run_dir = RESULTS_DIR / f"{timestamp}-{provider}-{spec.name}"

    profile = (env_get(env, "PROFILE") or "none").lower()
    profile_args = env_get(env, "PROFILE_ARGS")
    artifacts_dir = env_get(env, "ARTIFACTS_DIR") or "artifacts"
    resource_name = f"paris-bench-{user}-{timestamp}-{spec.name}"
    command = wrap_command_for_profile(
        base_command, profile, profile_args, artifacts_dir, resource_name)

    sync_mode = (env_get(env, "SYNC_MODE") or "rsync").lower()
    if sync_mode not in ("rsync", "docker"):
        raise ValueError(f"SYNC_MODE must be 'rsync' or 'docker', got {sync_mode!r}")

    registry_image = env_get(env, "REGISTRY_IMAGE")
    # A registry image implies docker sync.
    if registry_image and sync_mode != "docker":
        sync_mode = "docker"

    return RunConfig(
        provider=provider,
        spec=spec,
        command=command,
        env=env,
        timestamp=timestamp,
        run_dir=run_dir,
        keep_instance=env_flag(env, "KEEP_INSTANCE"),
        dry_run=env_flag(env, "DRY_RUN"),
        sync_mode=sync_mode,
        registry_image=registry_image,
        docker_install=env_flag(env, "DOCKER_INSTALL"),
        profile=profile,
        profile_args=profile_args,
        artifacts_dir=artifacts_dir,
        base_command=base_command,
        user=user,
    )
