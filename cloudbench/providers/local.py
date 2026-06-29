"""Local provider — run the benchmark command on this machine.

No provisioning, no cleanup. Output is captured into the same results
format as the cloud providers so runs are comparable.
"""

from __future__ import annotations

import subprocess

from ..config import REPO_ROOT
from . import ExecResult, Provider
from . import _shquote


class LocalProvider(Provider):
    name = "local"

    def provision(self) -> None:
        self.resource.instance_id = "local"
        self.resource.instance_type = "localhost"
        self.resource.region = "local"
        if self.cfg.dry_run:
            self._log(f"DRY_RUN: would run locally: {self.cfg.command}")

    def _shell_command(self) -> str:
        """The actual shell command to run. In docker mode, wrap the benchmark
        command in `docker run` with the repo mounted at /workspace, so a local
        GPU box can use the same torch image as the cloud."""
        if self.cfg.sync_mode != "docker":
            return self.cfg.command
        image = self.cfg.registry_image
        if not image:
            raise RuntimeError("SYNC_MODE=docker requires REGISTRY_IMAGE to be set.")
        gpu_flag = "--gpus all" if self.cfg.spec.needs_gpu else ""
        cap_flag = "--cap-add=SYS_ADMIN" if self.cfg.profile != "none" else ""
        self._log(f"docker mode: image {image}")
        return (
            f"docker run --rm {gpu_flag} {cap_flag} -v {REPO_ROOT}:/workspace "
            f"-w /workspace {image} bash -lc {_shquote(self.cfg.command)}"
        )

    def execute(self) -> ExecResult:
        if self.cfg.dry_run:
            text = (f"DRY_RUN plan (local): {self.cfg.command}\n"
                    f"  cwd     : {REPO_ROOT}\n"
                    f"  sync    : {self.cfg.sync_mode}"
                    + (f" (image {self.cfg.registry_image})"
                       if self.cfg.sync_mode == "docker" else "") + "\n"
                    f"  timeout : {self.cfg.spec.timeout_minutes}m")
            self._log(text)
            return ExecResult(exit_code=0, stdout=text + "\n")

        # Ensure the artifacts dir exists so a profiler wrapper can write to it.
        (REPO_ROOT / self.cfg.artifacts_dir).mkdir(parents=True, exist_ok=True)

        try:
            command = self._shell_command()
        except RuntimeError as exc:
            return ExecResult(exit_code=2, stderr=str(exc))

        self._log(f"Running locally: {command}")
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=self.cfg.spec.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecResult(
                exit_code=124,
                stdout=exc.stdout or "",
                stderr=(exc.stderr or "") +
                f"\n[cloudbench] timed out after {self.cfg.spec.timeout_minutes}m",
            )

        bench_json = self._read_local_result()
        artifacts = self._collect_local_artifacts()
        return ExecResult(
            exit_code=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            benchmark_result=bench_json,
            artifacts=artifacts,
        )

    def _collect_local_artifacts(self) -> list:
        """Copy REPO_ROOT/<artifacts_dir> into the run dir."""
        import shutil
        src = REPO_ROOT / self.cfg.artifacts_dir
        if not src.exists():
            return []
        dst = self.cfg.run_dir / self.cfg.artifacts_dir
        files = [p for p in src.rglob("*") if p.is_file()]
        if not files:
            return []
        self.cfg.run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        self._log(f"Collected {len(files)} artifact(s) -> {self.cfg.artifacts_dir}/")
        return [str((dst / p.relative_to(src)).relative_to(self.cfg.run_dir))
                for p in files]

    def cleanup(self) -> bool:
        return True

    def _read_local_result(self):
        path = REPO_ROOT / "benchmark_result.json"
        if path.exists():
            self._log("Found benchmark_result.json")
            return path.read_text()
        return None
