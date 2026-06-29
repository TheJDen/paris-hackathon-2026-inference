"""Shared SSH / rsync / scp helpers for cloud provider adapters.

These wrap plain ``subprocess`` calls to ``ssh``, ``rsync`` and ``scp`` so
the provider adapters only have to produce connection info, not reinvent
remote execution.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class SSHInfo:
    host: str
    user: str = "root"
    port: int = 22
    key_path: Optional[str] = None
    # Extra ssh -o options, e.g. {"StrictHostKeyChecking": "no"}.
    options: dict = field(default_factory=dict)

    def to_public_dict(self) -> dict:
        return {
            "host": self.host,
            "user": self.user,
            "port": self.port,
            "key_path": self.key_path,
        }

    def ssh_command_hint(self, remote_dir: Optional[str] = None) -> str:
        parts = ["ssh"]
        if self.key_path:
            parts += ["-i", self.key_path]
        if self.port != 22:
            parts += ["-p", str(self.port)]
        parts.append(f"{self.user}@{self.host}")
        hint = " ".join(parts)
        if remote_dir:
            hint += f"   # then: cd {remote_dir}"
        return hint


def _ssh_opts(info: SSHInfo) -> List[str]:
    opts: List[str] = []
    options = {
        "StrictHostKeyChecking": "no",
        "UserKnownHostsFile": "/dev/null",
        "ConnectTimeout": "30",
        **info.options,
    }
    for key, val in options.items():
        opts += ["-o", f"{key}={val}"]
    if info.key_path:
        # Expand ~ ourselves: these run via subprocess (no shell) so the shell
        # won't do it, and ssh/rsync don't expand ~ in a -i argument.
        opts += ["-i", os.path.expanduser(info.key_path)]
    return opts


def run_ssh(
    info: SSHInfo,
    command: str,
    *,
    timeout: Optional[int] = None,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    """Run a single command on the remote host over SSH."""
    argv = ["ssh", *_ssh_opts(info), "-p", str(info.port),
            f"{info.user}@{info.host}", command]
    return subprocess.run(
        argv,
        capture_output=capture,
        text=True,
        timeout=timeout,
    )


def rsync_up(
    info: SSHInfo,
    local_dir: Path,
    remote_dir: str,
    *,
    excludes: Optional[List[str]] = None,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess:
    """rsync a local directory up to the remote host."""
    excludes = excludes or []
    ssh_cmd = "ssh " + " ".join(_ssh_opts(info)) + f" -p {info.port}"
    argv = ["rsync", "-az", "--delete", "-e", ssh_cmd]
    for ex in excludes:
        argv += ["--exclude", ex]
    # Trailing slash on source => copy contents into remote_dir.
    argv += [f"{str(local_dir).rstrip('/')}/", f"{info.user}@{info.host}:{remote_dir}"]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def scp_down(
    info: SSHInfo,
    remote_path: str,
    local_path: Path,
    *,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess:
    """Copy a single file down from the remote host."""
    argv = ["scp", *_ssh_opts(info), "-P", str(info.port),
            f"{info.user}@{info.host}:{remote_path}", str(local_path)]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def rsync_down(
    info: SSHInfo,
    remote_dir: str,
    local_dir: Path,
    *,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess:
    """rsync a remote directory's contents down into local_dir."""
    local_dir.mkdir(parents=True, exist_ok=True)
    ssh_cmd = "ssh " + " ".join(_ssh_opts(info)) + f" -p {info.port}"
    argv = ["rsync", "-az", "-e", ssh_cmd,
            f"{info.user}@{info.host}:{remote_dir.rstrip('/')}/",
            f"{str(local_dir).rstrip('/')}/"]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
