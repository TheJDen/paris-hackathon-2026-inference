"""Modal provider — serverless GPU runs via the Modal CLI.

Intended ONLY for smoke tests because Modal credits are limited.

Modal is serverless: there is no long-lived instance to keep or destroy,
so ``KEEP_INSTANCE`` is a no-op here and cleanup always succeeds. We shell
out to ``modal run scripts/modal_bench.py``; that script runs the selected
benchmark command on a Modal GPU function and prints a JSON envelope on
stdout which we parse back out.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Optional

from ..config import REPO_ROOT, env_get
from . import ExecResult, Provider

# Sentinel the Modal entrypoint wraps its result envelope in.
RESULT_BEGIN = "__CLOUDBENCH_RESULT_BEGIN__"
RESULT_END = "__CLOUDBENCH_RESULT_END__"

MODAL_SCRIPT = REPO_ROOT / "scripts" / "modal_bench.py"


class ModalProvider(Provider):
    name = "modal"

    def provision(self) -> None:
        self.resource.instance_id = "modal-function"
        self.resource.instance_type = "modal-serverless"
        self.resource.region = env_get(self.cfg.env, "MODAL_REGION") or "modal"
        if self.cfg.spec.ngpus > 1:
            self._log("WARNING: Modal multi-GPU requires account support; "
                      "proceeding with NGPUS=%d." % self.cfg.spec.ngpus)
        if self.cfg.dry_run:
            self._log(f"DRY_RUN: would run via modal: {self.cfg.command}")

    def execute(self) -> ExecResult:
        if self.cfg.dry_run:
            text = (f"DRY_RUN plan (modal):\n"
                    f"  command : {self.cfg.command}\n"
                    f"  gpu     : {self.cfg.spec.ngpus}x{self.cfg.spec.gpu}\n"
                    f"  script  : {MODAL_SCRIPT}")
            self._log(text)
            return ExecResult(exit_code=0, stdout=text + "\n")

        if not MODAL_SCRIPT.exists():
            return ExecResult(exit_code=2,
                              stderr=f"Modal script not found: {MODAL_SCRIPT}")

        run_env = dict(os.environ)
        run_env.update({
            "CB_COMMAND": self.cfg.command,
            "CB_GPU": self.cfg.spec.gpu,
            "CB_NGPUS": str(self.cfg.spec.ngpus),
            "CB_TIMEOUT": str(self.cfg.spec.timeout_seconds),
            "CB_APP_NAME": self.cfg.resource_name,
            "CB_ARTIFACTS_DIR": self.cfg.artifacts_dir,
        })

        # Path/name of the modal CLI — keep it in an isolated install
        # (`uv tool install modal` or `uvx modal`), never the project env.
        modal_bin = env_get(self.cfg.env, "MODAL_BIN") or "modal"
        self._log(f"Launching: {modal_bin} run {MODAL_SCRIPT.name}")
        try:
            proc = subprocess.run(
                [modal_bin, "run", str(MODAL_SCRIPT)],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                env=run_env,
                timeout=self.cfg.spec.timeout_seconds + 300,
            )
        except FileNotFoundError:
            return ExecResult(exit_code=127,
                              stderr=f"modal CLI not found ({modal_bin!r}). Install in "
                                     "isolation: `uv tool install modal` (then "
                                     "`modal token new`), or set MODAL_BIN.")
        except subprocess.TimeoutExpired as exc:
            return ExecResult(exit_code=124, stdout=exc.stdout or "",
                              stderr=(exc.stderr or "") + "\n[cloudbench] modal run timed out")

        envelope = self._parse_envelope(proc.stdout or "")
        if envelope is None:
            # Fall back to raw modal CLI output.
            return ExecResult(exit_code=proc.returncode,
                              stdout=proc.stdout or "", stderr=proc.stderr or "")

        self.resource.extra["modal_app"] = self.cfg.resource_name
        artifacts = self._extract_artifacts(envelope)
        if envelope.get("artifacts_note"):
            self._log("note: " + envelope["artifacts_note"])
        return ExecResult(
            exit_code=int(envelope.get("exit_code", proc.returncode)),
            stdout=envelope.get("stdout", proc.stdout or ""),
            stderr=envelope.get("stderr", proc.stderr or ""),
            benchmark_result=(json.dumps(envelope["benchmark_result"])
                              if envelope.get("benchmark_result") is not None else None),
            artifacts=artifacts,
        )

    def _extract_artifacts(self, envelope: dict) -> list:
        """Unpack the base64 tar.gz of artifacts returned by the Modal function."""
        blob = envelope.get("artifacts_tar_b64")
        if not blob:
            return []
        import base64
        import io
        import tarfile
        self.cfg.run_dir.mkdir(parents=True, exist_ok=True)
        try:
            data = base64.b64decode(blob)
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                tar.extractall(self.cfg.run_dir)
        except (tarfile.TarError, OSError, ValueError) as exc:
            self._log(f"note: could not extract artifacts: {exc}")
            return []
        art = self.cfg.run_dir / self.cfg.artifacts_dir
        files = [str(p.relative_to(self.cfg.run_dir))
                 for p in sorted(art.rglob("*")) if p.is_file()]
        if files:
            self._log(f"Extracted {len(files)} artifact(s) -> {self.cfg.artifacts_dir}/")
        return files

    def cleanup(self) -> bool:
        # Serverless: nothing persistent to tear down.
        return True

    @staticmethod
    def _parse_envelope(stdout: str) -> Optional[dict]:
        if RESULT_BEGIN not in stdout or RESULT_END not in stdout:
            return None
        chunk = stdout.split(RESULT_BEGIN, 1)[1].split(RESULT_END, 1)[0].strip()
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            return None
