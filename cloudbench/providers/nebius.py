"""Nebius provider — driven via the Nebius CLI (`nebius`).

Intended for 1xH200 sanity checks and 8xH200 distributed benches. Assumes the
CLI has an authenticated profile (see README — service account recommended).

Verified against Nebius CLI 0.12.x. Key facts that differ from a naive guess:
  * instance type = `--resources-platform` + `--resources-preset`. The platform
    is the SAME for 1x and 8x H200 (gpu-h200-sxm); only the preset changes.
  * boot image via `--boot-disk-managed-disk-source-image-id` + a disk size.
  * networking via `--network-interfaces` as a JSON list; an empty
    `public_ip_address: {}` requests a public IP.
  * there is NO --ssh-key flag — the public key is injected with cloud-init
    (`--cloud-init-user-data`).
  * status lives at status.state (RUNNING == ready); public IP at
    status.network_interfaces[].public_ip_address.address.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import List, Optional

from ..config import env_get
from . import CloudProvider
from ._remote import SSHInfo

READY_TIMEOUT_S = 1200
POLL_INTERVAL_S = 15

# Sub-command path after the binary; verbs centralized for easy patching.
CLI = {
    "service": ["compute", "instance"],
    "create":  "create",
    "get":     "get",
    "delete":  "delete",
    "list":    "list",
}


class NebiusProvider(CloudProvider):
    name = "nebius"

    def __init__(self, cfg):
        super().__init__(cfg)
        env = cfg.env
        self.project_id = env_get(env, "NEBIUS_PROJECT_ID")
        self.subnet_id = env_get(env, "NEBIUS_SUBNET_ID")
        self.image_id = env_get(env, "NEBIUS_IMAGE_ID")
        self.platform = env_get(env, "NEBIUS_PLATFORM_H200")
        self.preset_1x = env_get(env, "NEBIUS_PRESET_H200")
        self.preset_8x = env_get(env, "NEBIUS_PRESET_H200_8X")
        self.boot_disk_gib = int(env_get(env, "NEBIUS_BOOT_DISK_GIB") or 512)
        self.disk_type = env_get(env, "NEBIUS_DISK_TYPE") or "network_ssd"
        self.gpu_cluster_id = env_get(env, "NEBIUS_GPU_CLUSTER_ID")  # optional
        self.ssh_user = env_get(env, "NEBIUS_SSH_USER") or "ubuntu"
        self.ssh_key_path = env_get(env, "NEBIUS_SSH_KEY") or env_get(env, "SSH_KEY_PATH")
        self.ssh_public_key = env_get(env, "NEBIUS_SSH_PUBLIC_KEY") or (
            (self.ssh_key_path + ".pub") if self.ssh_key_path else None)
        self.bin = env_get(env, "NEBIUS_BIN") or "nebius"

    # -- which preset for this run -------------------------------------- #
    @property
    def _is_8x(self) -> bool:
        return self.cfg.spec.ngpus >= 8

    @property
    def _preset(self) -> Optional[str]:
        return self.preset_8x if self._is_8x else self.preset_1x

    def _require_config(self) -> None:
        required = {
            "NEBIUS_PROJECT_ID": self.project_id,
            "NEBIUS_SUBNET_ID": self.subnet_id,
            "NEBIUS_IMAGE_ID": self.image_id,
            "NEBIUS_PLATFORM_H200": self.platform,
            ("NEBIUS_PRESET_H200_8X" if self._is_8x else "NEBIUS_PRESET_H200"): self._preset,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise RuntimeError(
                "Nebius provider missing required config (configs/providers.conf): "
                + ", ".join(missing))

    # -- CLI helper ------------------------------------------------------ #
    def _cli(self, *args: str, check: bool = True, timeout: int = 300):
        argv = [self.bin, *CLI["service"], *args, "--format", "json"]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError:
            raise RuntimeError(
                f"nebius CLI not found ({self.bin!r}). Install it or set NEBIUS_BIN.")
        if check and proc.returncode != 0:
            raise RuntimeError(f"nebius CLI failed: {' '.join(argv)}\n{proc.stderr}")
        return proc

    @staticmethod
    def _parse_json(proc) -> dict:
        out = (proc.stdout or "").strip()
        if not out:
            return {}
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return {"_raw": out}

    # -- cloud-init for SSH key injection -------------------------------- #
    def _cloud_init(self) -> Optional[str]:
        if not self.ssh_public_key:
            return None
        pub_path = Path(self.ssh_public_key).expanduser()
        if not pub_path.exists():
            raise RuntimeError(f"SSH public key not found: {pub_path}")
        pub = pub_path.read_text().strip()
        return ("#cloud-config\n"
                "ssh_authorized_keys:\n"
                f"  - {pub}\n")

    def _network_interfaces(self) -> str:
        # Empty ip_address/public_ip_address objects => auto-assign both.
        return json.dumps([{
            "name": "eth0",
            "subnet_id": self.subnet_id,
            "ip_address": {},
            "public_ip_address": {},
        }])

    # -- lifecycle ------------------------------------------------------- #
    def create_instance(self) -> None:
        self._require_config()
        args: List[str] = [
            CLI["create"],
            "--parent-id", self.project_id,
            "--name", self.cfg.resource_name,
            "--resources-platform", self.platform,
            "--resources-preset", self._preset,
            "--boot-disk-attach-mode", "read_write",
            "--boot-disk-managed-disk-name", f"{self.cfg.resource_name}-boot",
            "--boot-disk-managed-disk-type", self.disk_type,
            "--boot-disk-managed-disk-source-image-id", self.image_id,
            "--boot-disk-managed-disk-size-gibibytes", str(self.boot_disk_gib),
            "--network-interfaces", self._network_interfaces(),
        ]
        cloud_init = self._cloud_init()
        if cloud_init:
            args += ["--cloud-init-user-data", cloud_init]
        if self._is_8x and self.gpu_cluster_id:
            args += ["--gpu-cluster-id", self.gpu_cluster_id]

        resp = self._parse_json(self._cli(*args, timeout=600))
        new_id = (resp.get("metadata", {}).get("id")
                  or resp.get("resource_id")
                  or resp.get("id"))
        if not new_id:
            # Create may be async/return an operation; fall back to name lookup.
            new_id = self.find_instance_id_by_name(self.cfg.resource_name)
        if not new_id:
            raise RuntimeError(f"Nebius create returned no instance id: {resp}")
        self.instance_id = new_id
        self.resource.instance_type = f"{self.platform}/{self._preset}"
        self.resource.extra["create_response"] = resp

    def _get_instance(self) -> dict:
        return self._parse_json(
            self._cli(CLI["get"], "--id", self.instance_id, check=False))

    @staticmethod
    def _state(info: dict) -> str:
        status = info.get("status", {})
        state = status.get("state") if isinstance(status, dict) else status
        return str(state or "").upper()

    def wait_ready(self) -> None:
        deadline = time.monotonic() + READY_TIMEOUT_S
        while time.monotonic() < deadline:
            info = self._get_instance()
            state = self._state(info)
            if state == "RUNNING":
                self.resource.extra["ready_response"] = info
                return
            if state in ("ERROR", "DELETING", "DELETED"):
                raise RuntimeError(f"Nebius instance entered state {state}: {info}")
            self._log(f"  state={state or '?'} ... waiting")
            time.sleep(POLL_INTERVAL_S)
        raise TimeoutError(f"Nebius instance not ready after {READY_TIMEOUT_S}s")

    def get_ssh_info(self) -> SSHInfo:
        info = self._get_instance()
        host = self._extract_public_ip(info)
        if not host:
            raise RuntimeError(f"Nebius instance has no public IP yet: {info}")
        return SSHInfo(host=host, user=self.ssh_user, port=22,
                       key_path=self.ssh_key_path)

    @staticmethod
    def _extract_public_ip(info: dict) -> Optional[str]:
        status = info.get("status", {})
        nics = status.get("network_interfaces") or info.get("network_interfaces") or []
        for nic in nics:
            pub = nic.get("public_ip_address") or {}
            addr = pub.get("address")
            if addr:
                return str(addr).split("/")[0]
        return None

    def destroy_instance(self) -> bool:
        if not self.instance_id:
            return True
        proc = self._cli(CLI["delete"], "--id", self.instance_id, check=False)
        if proc.returncode == 0:
            return True
        err = (proc.stderr or "").lower()
        if "not found" in err or "notfound" in err or "does not exist" in err:
            return True  # already gone
        return False

    def find_instance_id_by_name(self, name: str) -> Optional[str]:
        if not self.project_id:
            return None
        data = self._parse_json(
            self._cli(CLI["list"], "--parent-id", self.project_id, check=False))
        for item in data.get("items", []):
            meta = item.get("metadata", item)
            if meta.get("name") == name:
                return meta.get("id") or item.get("id")
        return None
