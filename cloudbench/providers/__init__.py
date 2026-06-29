"""Provider adapters and the lifecycle interface the runner drives.

Lifecycle (driven by ``runner.py``):

    provider = get_provider(name, cfg)
    try:
        provider.provision()           # create instance / pull image / sync code
        result = provider.execute()    # run the benchmark, capture output
    finally:
        cleanup_ok = provider.cleanup()  # destroy resource unless KEEP_INSTANCE

Cloud providers share :class:`CloudProvider`, which implements the
sync + remote-run + result-download workflow once; concrete adapters only
implement create/wait/ssh-info/destroy.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..config import RunConfig
from . import _remote
from ._remote import SSHInfo


# --------------------------------------------------------------------------- #
# Result + resource data carried back to the runner
# --------------------------------------------------------------------------- #

@dataclass
class ExecResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    benchmark_result: Optional[str] = None  # raw JSON text, if produced
    artifacts: list = field(default_factory=list)  # relative paths collected


@dataclass
class ResourceInfo:
    provider: str
    instance_id: Optional[str] = None
    instance_type: Optional[str] = None
    region: Optional[str] = None
    gpu: Optional[str] = None
    ngpus: int = 0
    ssh: Optional[dict] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "instance_id": self.instance_id,
            "instance_type": self.instance_type,
            "region": self.region,
            "gpu": self.gpu,
            "ngpus": self.ngpus,
            "ssh": self.ssh,
            "extra": self.extra,
        }


# --------------------------------------------------------------------------- #
# Base provider
# --------------------------------------------------------------------------- #

class Provider:
    """Abstract provider. Subclasses implement provision/execute/cleanup."""

    name = "base"

    # Default rsync excludes so we never ship the world.
    SYNC_EXCLUDES = [
        ".git", ".venv", "venv", "__pycache__", "*.pyc",
        "results", "node_modules", ".pytest_cache", "*.egg-info",
    ]

    # Where source is synced / mounted on the remote host.
    REMOTE_WORKDIR = "/root/bench"

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.resource = ResourceInfo(
            provider=self.name,
            gpu=cfg.spec.gpu,
            ngpus=cfg.spec.ngpus,
        )

    # -- lifecycle ------------------------------------------------------- #
    def provision(self) -> None:
        raise NotImplementedError

    def execute(self) -> ExecResult:
        raise NotImplementedError

    def cleanup(self) -> bool:
        """Return True if cleanup succeeded (or nothing to clean)."""
        return True

    # -- helpers --------------------------------------------------------- #
    def _log(self, msg: str) -> None:
        print(f"[{self.name}] {msg}", flush=True)

    def ssh_instructions(self) -> Optional[str]:
        """Human-readable SSH hint, used when KEEP_INSTANCE=1."""
        return None


class CloudProvider(Provider):
    """Shared workflow for SSH-reachable cloud instances.

    Concrete adapters implement:
        create_instance() -> sets self.resource fields
        wait_ready()
        get_ssh_info() -> SSHInfo
        destroy_instance() -> bool
    """

    name = "cloud"

    # How many times to retry a destroy before giving up.
    DESTROY_ATTEMPTS = 5

    def __init__(self, cfg: RunConfig):
        super().__init__(cfg)
        self._ssh: Optional[SSHInfo] = None
        self._workdir: Optional[str] = None  # resolved under the SSH user's home
        # Optimistic ownership: as soon as we *intend* to create, we own it.
        # This guarantees cleanup attempts a destroy even if create_instance()
        # is interrupted mid-call (the dangerous window for orphaned GPUs).
        self._created = False

    # Single source of truth for the instance id (subclasses read/write this).
    @property
    def instance_id(self) -> Optional[str]:
        return self.resource.instance_id

    @instance_id.setter
    def instance_id(self, value: Optional[str]) -> None:
        self.resource.instance_id = value

    # -- to implement in subclasses -------------------------------------- #
    def create_instance(self) -> None:
        raise NotImplementedError

    def wait_ready(self) -> None:
        raise NotImplementedError

    def get_ssh_info(self) -> SSHInfo:
        raise NotImplementedError

    def destroy_instance(self) -> bool:
        raise NotImplementedError

    def find_instance_id_by_name(self, name: str) -> Optional[str]:
        """Best-effort lookup so we can destroy a resource whose id we never
        captured (e.g. create was interrupted). Returns None if unsupported."""
        return None

    # -- ledger ---------------------------------------------------------- #
    def _ledger_record(self) -> dict:
        from .. import ledger  # noqa: F401 (import for side-effect availability)
        return {
            "resource_name": self.cfg.resource_name,
            "provider": self.name,
            "instance_id": self.resource.instance_id,
            "instance_type": self.resource.instance_type,
            "region": self.resource.region,
            "gpu": self.cfg.spec.gpu,
            "ngpus": self.cfg.spec.ngpus,
            "bench": self.cfg.spec.name,
            "run_dir": str(self.cfg.run_dir),
            "keep": self.cfg.keep_instance,
        }

    def _write_ledger(self) -> None:
        from .. import ledger
        try:
            ledger.write_record(self._ledger_record())
        except Exception as exc:  # noqa: BLE001 - ledger must never break a run
            self._log(f"warning: could not write ledger record: {exc}")

    def _clear_ledger(self) -> None:
        from .. import ledger
        try:
            ledger.remove_record(self.cfg.resource_name)
        except Exception as exc:  # noqa: BLE001
            self._log(f"warning: could not clear ledger record: {exc}")

    # -- shared lifecycle ------------------------------------------------ #
    def provision(self) -> None:
        if self.cfg.dry_run:
            self._log(f"DRY_RUN: would create resource '{self.cfg.resource_name}' "
                      f"({self.cfg.spec.ngpus}x{self.cfg.spec.gpu})")
            return
        # Claim ownership + drop a ledger breadcrumb BEFORE the create call,
        # so a hard kill mid-create still leaves a trace for `reap`.
        self._created = True
        self._write_ledger()
        self._log(f"Creating resource '{self.cfg.resource_name}' ...")
        self.create_instance()
        self._write_ledger()  # update with the captured instance_id
        self._log("Waiting for instance to become ready ...")
        self.wait_ready()
        self._ssh = self.get_ssh_info()
        self.resource.ssh = self._ssh.to_public_dict()
        self._log(f"Instance ready: {self._ssh.host}")
        self._wait_for_ssh()

    def _wait_for_ssh(self, timeout: int = 300) -> None:
        """A provider's RUNNING state does not mean sshd is accepting yet
        (cloud-init / GPU driver init can lag). Poll a trivial remote command
        until SSH actually works, so the first rsync doesn't hit 'connection
        refused'."""
        import time
        self._log("Waiting for SSH to accept connections ...")
        deadline = time.monotonic() + timeout
        last = ""
        while time.monotonic() < deadline:
            try:
                proc = _remote.run_ssh(self._ssh, "true", timeout=20)
                if proc.returncode == 0:
                    self._log("SSH is up.")
                    return
                last = (proc.stderr or "").strip().splitlines()[-1:] or [""]
                last = last[0]
            except Exception as exc:  # noqa: BLE001
                last = str(exc)
            time.sleep(10)
        raise TimeoutError(f"SSH not reachable after {timeout}s (last: {last})")

    def run_remote_command(self, command: str, timeout: Optional[int] = None):
        """Run a command on the provisioned instance (exposed for adapters/tests)."""
        assert self._ssh is not None, "instance not provisioned"
        return _remote.run_ssh(self._ssh, command, timeout=timeout)

    def _resolve_workdir(self) -> str:
        """Workdir under the SSH user's home, so it works whether we log in as
        root (Sesterce) or a non-root user like ubuntu (Nebius)."""
        if getattr(self, "_workdir", None):
            return self._workdir
        home = "/root"
        try:
            proc = self.run_remote_command("echo $HOME", timeout=60)
            if proc.returncode == 0 and proc.stdout.strip():
                home = proc.stdout.strip().splitlines()[-1].strip()
        except Exception:  # noqa: BLE001 - fall back to /root
            pass
        self._workdir = f"{home.rstrip('/')}/cloudbench"
        return self._workdir

    def execute(self) -> ExecResult:
        if self.cfg.dry_run:
            return self._dry_run_execute()

        assert self._ssh is not None, "instance not provisioned"
        workdir = self._resolve_workdir()

        if self.cfg.sync_mode == "docker":
            return self._execute_docker(workdir)
        return self._execute_rsync(workdir)

    def cleanup(self) -> bool:
        if self.cfg.dry_run:
            self._log("DRY_RUN: would destroy resource")
            return True
        if not self._created:
            self._log("No resource created by this harness; nothing to clean up.")
            return True
        if self.cfg.keep_instance:
            self._log("KEEP_INSTANCE=1 set: leaving instance running.")
            self._log("It stays in the ledger; `make reap` will destroy it later.")
            hint = self.ssh_instructions()
            if hint:
                self._log("SSH in with:\n    " + hint)
            return True
        ok = self._destroy_with_retries()
        if ok:
            self._clear_ledger()
        else:
            self._log("!! DESTROY FAILED. Resource left in ledger — run `make reap` "
                      f"or destroy '{self.cfg.resource_name}' manually NOW.")
        return ok

    def _destroy_with_retries(self) -> bool:
        """Idempotent, retrying destroy. Recovers the instance id by name if we
        never captured it (interrupted create)."""
        import time

        # If create was interrupted before we got an id, try to find it by name.
        if not self.resource.instance_id:
            try:
                found = self.find_instance_id_by_name(self.cfg.resource_name)
            except Exception as exc:  # noqa: BLE001
                found = None
                self._log(f"name lookup failed: {exc}")
            if found:
                self.resource.instance_id = found
                self._log(f"recovered instance id by name: {found}")

        self._log("Destroying resource ...")
        for attempt in range(1, self.DESTROY_ATTEMPTS + 1):
            try:
                if self.destroy_instance():
                    self._log("Resource destroyed.")
                    return True
                self._log(f"destroy attempt {attempt} reported failure")
            except Exception as exc:  # noqa: BLE001 - retry, never mask cleanup
                self._log(f"destroy attempt {attempt} error: {exc}")
            if attempt < self.DESTROY_ATTEMPTS:
                time.sleep(min(5 * attempt, 30))
        return False

    def ssh_instructions(self) -> Optional[str]:
        if self._ssh is None:
            return None
        return self._ssh.ssh_command_hint(self._workdir or self.REMOTE_WORKDIR)

    # -- run strategies -------------------------------------------------- #
    def _execute_rsync(self, workdir: str) -> ExecResult:
        self._log(f"Syncing source -> {workdir} (rsync) ...")
        self.run_remote_command(f"mkdir -p {workdir}")
        from ..config import REPO_ROOT
        rs = _remote.rsync_up(
            self._ssh, REPO_ROOT, workdir,
            excludes=self.SYNC_EXCLUDES,
            timeout=600,
        )
        if rs.returncode != 0:
            return ExecResult(exit_code=rs.returncode, stdout=rs.stdout,
                              stderr="rsync failed:\n" + rs.stderr)

        # Create the artifacts dir AFTER the (--delete) sync so it survives.
        self.run_remote_command(f"mkdir -p {workdir}/{self.cfg.artifacts_dir}")
        remote_cmd = self._remote_run_script(workdir, self.cfg.command)
        return self._run_and_collect(remote_cmd, workdir)

    def _docker_bin(self) -> Optional[str]:
        """Return how to invoke docker on the instance ('docker' or 'sudo
        docker'), or None if docker isn't usable."""
        if self.run_remote_command("docker info", timeout=60).returncode == 0:
            return "docker"
        if self.run_remote_command("sudo docker info", timeout=60).returncode == 0:
            return "sudo docker"
        return None

    def _install_docker(self) -> None:
        """Best-effort install of docker + NVIDIA runtime (DOCKER_INSTALL=1).
        Bare images (e.g. Nebius worker-node) ship no docker CLI."""
        self._log("DOCKER_INSTALL=1: installing docker engine on the instance ...")
        script = (
            "curl -fsSL https://get.docker.com | sudo sh; "
            # Point docker at the NVIDIA runtime if the toolkit is present
            # (GPU node images usually have nvidia-ctk for containerd already).
            "if command -v nvidia-ctk >/dev/null 2>&1; then "
            "sudo nvidia-ctk runtime configure --runtime=docker && "
            "sudo systemctl restart docker; fi; true"
        )
        self.run_remote_command(script, timeout=600)

    def _execute_docker(self, workdir: str) -> ExecResult:
        image = self.cfg.registry_image
        if not image:
            return ExecResult(
                exit_code=2, stderr="SYNC_MODE=docker requires REGISTRY_IMAGE to be set.")

        docker = self._docker_bin()
        if docker is None and self.cfg.docker_install:
            self._install_docker()
            docker = self._docker_bin()
        if docker is None:
            hint = ("" if self.cfg.docker_install else
                    " Re-run with DOCKER_INSTALL=1 to auto-install it, or")
            return ExecResult(
                exit_code=127,
                stderr=f"docker not usable on the instance.{hint} use an image/host "
                       "with docker + nvidia-container-toolkit, or SYNC_MODE=rsync.")

        # Sync source in (deps come from the image, code from the mount — so you
        # iterate on code without rebuilding the image).
        self._log(f"Syncing source -> {workdir} (docker mode) ...")
        self.run_remote_command(f"mkdir -p {workdir}")
        from ..config import REPO_ROOT
        rs = _remote.rsync_up(self._ssh, REPO_ROOT, workdir,
                              excludes=self.SYNC_EXCLUDES, timeout=600)
        if rs.returncode != 0:
            return ExecResult(exit_code=rs.returncode, stdout=rs.stdout,
                              stderr="rsync failed:\n" + rs.stderr)
        self.run_remote_command(f"mkdir -p {workdir}/{self.cfg.artifacts_dir}")

        self._log(f"Pulling image {image} ...")
        pull = self.run_remote_command(f"{docker} pull {image}", timeout=1800)
        if pull.returncode != 0:
            return ExecResult(exit_code=pull.returncode, stdout=pull.stdout,
                              stderr="docker pull failed:\n" + pull.stderr)

        gpu_flag = "--gpus all" if self.cfg.spec.needs_gpu else ""
        # SYS_ADMIN is required for ncu/nsys GPU perf counters inside a container.
        cap_flag = "--cap-add=SYS_ADMIN" if self.cfg.profile != "none" else ""
        # Mount workdir so code is available and results/artifacts land on the
        # host for download.
        container_cmd = (
            f"{docker} run --rm {gpu_flag} {cap_flag} -v {workdir}:/workspace "
            f"-w /workspace {image} bash -lc {_shquote(self.cfg.command)}"
        )
        return self._run_and_collect(container_cmd, workdir)

    def _remote_run_script(self, workdir: str, command: str) -> str:
        """Wrap the benchmark command so it runs inside the synced workdir.

        Activates a .venv if one is present on the image, otherwise runs as-is.
        """
        inner = (
            f"cd {workdir} && "
            f"if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi && "
            f"{command}"
        )
        return f"bash -lc {_shquote(inner)}"

    def _run_and_collect(self, remote_cmd: str, workdir: str) -> ExecResult:
        self._log("Running benchmark on remote ...")
        proc = self.run_remote_command(remote_cmd, timeout=self.cfg.spec.timeout_seconds)
        bench_json = self._download_result(workdir)
        artifacts = self._download_artifacts(workdir)
        return ExecResult(
            exit_code=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            benchmark_result=bench_json,
            artifacts=artifacts,
        )

    def _download_artifacts(self, workdir: str) -> list:
        """rsync the remote artifacts dir (profiler reports, chrome traces, ...)
        down into <run_dir>/artifacts. Returns the relative paths collected."""
        remote_dir = f"{workdir}/{self.cfg.artifacts_dir}"
        local_dir = self.cfg.run_dir / self.cfg.artifacts_dir
        res = _remote.rsync_down(self._ssh, remote_dir, local_dir, timeout=900)
        if res.returncode != 0:
            self._log(f"note: no artifacts collected ({res.stderr.strip()[:120]})")
            return []
        collected = [str(p.relative_to(self.cfg.run_dir))
                     for p in sorted(local_dir.rglob("*")) if p.is_file()]
        if collected:
            self._log(f"Downloaded {len(collected)} artifact(s) -> "
                      f"{self.cfg.artifacts_dir}/")
        return collected

    def _download_result(self, workdir: str) -> Optional[str]:
        """Try to pull benchmark_result.json down from the remote workdir."""
        import tempfile
        remote_path = f"{workdir}/benchmark_result.json"
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "benchmark_result.json"
            res = _remote.scp_down(self._ssh, remote_path, local, timeout=120)
            if res.returncode == 0 and local.exists():
                self._log("Downloaded benchmark_result.json")
                return local.read_text()
        return None

    def _dry_run_execute(self) -> ExecResult:
        plan = [
            f"DRY_RUN plan for provider={self.name} bench={self.cfg.spec.name}",
            f"  resource name : {self.cfg.resource_name}",
            f"  gpu           : {self.cfg.spec.ngpus}x{self.cfg.spec.gpu}",
            f"  sync mode     : {self.cfg.sync_mode}",
            f"  registry image: {self.cfg.registry_image or '(none)'}",
            f"  command       : {self.cfg.command}",
            f"  timeout       : {self.cfg.spec.timeout_minutes}m",
            f"  cleanup       : {'KEEP' if self.cfg.keep_instance else 'destroy'}",
        ]
        text = "\n".join(plan)
        self._log(text)
        return ExecResult(exit_code=0, stdout=text + "\n")


def _shquote(s: str) -> str:
    """Single-quote a string for safe embedding in a remote bash -lc."""
    return "'" + s.replace("'", "'\"'\"'") + "'"


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

def get_provider(name: str, cfg: RunConfig) -> Provider:
    name = name.lower()
    if name == "local":
        from .local import LocalProvider
        return LocalProvider(cfg)
    if name == "sesterce":
        from .sesterce import SesterceProvider
        return SesterceProvider(cfg)
    if name == "nebius":
        from .nebius import NebiusProvider
        return NebiusProvider(cfg)
    if name == "modal":
        from .modal import ModalProvider
        return ModalProvider(cfg)
    raise ValueError(
        f"unknown provider {name!r}. Choose: local, sesterce, nebius, modal")


PROVIDERS = ("local", "sesterce", "nebius", "modal")


def reap_record(record: dict, env: dict, dry_run: bool = False) -> bool:
    """Destroy a single resource described by a ledger record, out-of-process.

    Used by `cloudbench.cli reap` to sweep orphans left by a hard kill.
    Reconstructs the right provider, restores the instance id (or recovers it
    by name), and calls destroy.
    """
    from ..config import BenchmarkSpec, RunConfig

    provider_name = record.get("provider", "")
    spec = BenchmarkSpec(
        name=record.get("bench", "reap"),
        command="",
        gpu=record.get("gpu", "none"),
        ngpus=int(record.get("ngpus", 0) or 0),
    )
    cfg = RunConfig(
        provider=provider_name,
        spec=spec,
        command="",
        env=env,
        timestamp="",
        run_dir=Path(record.get("run_dir", ".")),
    )
    provider = get_provider(provider_name, cfg)
    if not isinstance(provider, CloudProvider):
        return True  # local/modal never leak; nothing to do

    provider._created = True
    provider.resource.instance_id = record.get("instance_id")
    provider.resource.region = record.get("region")

    name = record.get("resource_name", "?")
    if dry_run:
        print(f"[reap] DRY_RUN would destroy {provider_name}:{name} "
              f"(id={record.get('instance_id')})")
        return True

    ok = provider._destroy_with_retries()
    if ok:
        from .. import ledger
        ledger.remove_record(name)
    return ok
