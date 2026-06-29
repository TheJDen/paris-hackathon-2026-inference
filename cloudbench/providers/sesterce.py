"""Sesterce provider — REST-API driven H200 instances.

Default for frequent single-H200 iteration; also supports 8xH200.

API shape (verified against https://docs.sesterce.com/api-reference):
  * Base URL : https://api.cloud.sesterce.com
  * Auth     : header  X-API-KEY: <key>   (NOT Authorization: Bearer)
               A normal User-Agent is required or Cloudflare returns 1010.
  * Endpoints:
      GET    /gpu-cloud/instances/offers      list catalog (regions, SKUs, OS)
      POST   /gpu-cloud/instances             create
      GET    /gpu-cloud/instances             list
      GET    /gpu-cloud/instances/{id}        get
      DELETE /gpu-cloud/instances/{id}        delete
  * Create body: name, cloudProvider, instanceId, region, vm.os (+ optional
      sshKeyId / dockerContainer.image)
  * Instance fields: _id, status (pending|active|error|deleted|deleting),
      ip, sshUser, sshPort

Endpoints/field names are centralized below so they're easy to patch.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Optional

from ..config import env_get
from . import CloudProvider
from ._remote import SSHInfo


# --------------------------------------------------------------------------- #
# Centralized API surface.
# --------------------------------------------------------------------------- #

DEFAULT_BASE_URL = "https://api.cloud.sesterce.com"

ENDPOINTS = {
    "offers":          "/gpu-cloud/instances/offers",
    "create_instance": "/gpu-cloud/instances",
    "list_instances":  "/gpu-cloud/instances",
    "get_instance":    "/gpu-cloud/instances/{id}",
    "delete_instance": "/gpu-cloud/instances/{id}",
}

FIELDS = {
    "instance_id":  "_id",
    "status":       "status",
    "ready_status": "active",
    "ssh_host":     "ip",
    "ssh_user":     "sshUser",
    "ssh_port":     "sshPort",
    "name":         "name",
}

# Cloudflare bans the default urllib UA; any normal UA passes.
USER_AGENT = "cloudbench/0.1"

READY_TIMEOUT_S = 900
POLL_INTERVAL_S = 10


class SesterceProvider(CloudProvider):
    name = "sesterce"

    def __init__(self, cfg):
        super().__init__(cfg)
        env = cfg.env
        self.api_key = env_get(env, "SESTERCE_API_KEY")
        self.base_url = (env_get(env, "SESTERCE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.region = env_get(env, "SESTERCE_REGION")
        self.cloud_provider = env_get(env, "SESTERCE_CLOUD_PROVIDER")
        self.os_image = env_get(env, "SESTERCE_OS")
        self.instance_type_1x = env_get(env, "SESTERCE_INSTANCE_TYPE_H200")
        self.instance_type_8x = env_get(env, "SESTERCE_INSTANCE_TYPE_H200_8X")
        self.ssh_key_id = env_get(env, "SESTERCE_SSH_KEY_ID")
        self.ssh_key_path = (env_get(env, "SESTERCE_SSH_KEY_PATH")
                             or env_get(env, "SESTERCE_SSH_KEY")
                             or env_get(env, "SSH_KEY_PATH"))

    # -- which SKU for this run ------------------------------------------ #
    @property
    def _is_8x(self) -> bool:
        return self.cfg.spec.ngpus >= 8

    @property
    def _instance_type(self) -> Optional[str]:
        return self.instance_type_8x if self._is_8x else self.instance_type_1x

    # -- config validation ---------------------------------------------- #
    def _require_config(self) -> None:
        required = {
            "SESTERCE_API_KEY": self.api_key,
            "SESTERCE_REGION": self.region,
            "SESTERCE_CLOUD_PROVIDER": self.cloud_provider,
            "SESTERCE_OS": self.os_image,
        }
        if self._is_8x:
            required["SESTERCE_INSTANCE_TYPE_H200_8X"] = self.instance_type_8x
        else:
            required["SESTERCE_INSTANCE_TYPE_H200"] = self.instance_type_1x
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise RuntimeError(
                "Sesterce provider missing required config (configs/providers.conf "
                "or .env): " + ", ".join(missing))

    # -- HTTP helper ----------------------------------------------------- #
    def _request(self, method: str, path: str, body: Optional[dict] = None):
        url = self.base_url + path
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("X-API-KEY", self.api_key or "")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", USER_AGENT)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(
                f"Sesterce API {method} {path} -> {exc.code}: {detail}") from exc

    # -- lifecycle ------------------------------------------------------- #
    def create_instance(self) -> None:
        self._require_config()
        payload = {
            "name": self.cfg.resource_name,
            "cloudProvider": self.cloud_provider,
            "instanceId": self._instance_type,
            "region": self.region,
            "vm": {"os": self.os_image},
        }
        if self.ssh_key_id:
            payload["sshKeyId"] = self.ssh_key_id
        resp = self._request("POST", ENDPOINTS["create_instance"], payload)
        new_id = resp.get(FIELDS["instance_id"]) or resp.get("id")
        if not new_id:
            raise RuntimeError(f"Sesterce create returned no instance id: {resp}")
        self.instance_id = str(new_id)
        self.resource.instance_type = self._instance_type
        self.resource.region = self.region
        self.resource.extra["create_response"] = resp

    def _get_instance(self) -> dict:
        path = ENDPOINTS["get_instance"].format(id=self.instance_id)
        return self._request("GET", path)

    def wait_ready(self) -> None:
        ready = str(FIELDS["ready_status"]).lower()
        deadline = time.monotonic() + READY_TIMEOUT_S
        while time.monotonic() < deadline:
            info = self._get_instance()
            status = str(info.get(FIELDS["status"], "")).lower()
            if status == ready:
                self.resource.extra["ready_response"] = info
                return
            if status in ("error", "deleted", "deleting"):
                raise RuntimeError(f"Sesterce instance entered status {status!r}: {info}")
            self._log(f"  status={status or '?'} ... waiting")
            time.sleep(POLL_INTERVAL_S)
        raise TimeoutError(f"Sesterce instance not ready after {READY_TIMEOUT_S}s")

    def get_ssh_info(self) -> SSHInfo:
        info = self._get_instance()
        host = info.get(FIELDS["ssh_host"])
        if not host:
            raise RuntimeError(f"Sesterce instance has no SSH host yet: {info}")
        return SSHInfo(
            host=str(host),
            user=str(info.get(FIELDS["ssh_user"]) or "root"),
            port=int(info.get(FIELDS["ssh_port"]) or 22),
            key_path=self.ssh_key_path,
        )

    def destroy_instance(self) -> bool:
        if not self.instance_id:
            return True
        path = ENDPOINTS["delete_instance"].format(id=self.instance_id)
        try:
            self._request("DELETE", path)
        except RuntimeError as exc:
            if "404" in str(exc) or "410" in str(exc):
                return True  # already gone == success
            raise
        return True

    def find_instance_id_by_name(self, name: str) -> Optional[str]:
        """Recover the id of an instance we created but whose id we lost
        (e.g. create was interrupted), by listing and matching on name."""
        try:
            resp = self._request("GET", ENDPOINTS["list_instances"])
        except RuntimeError:
            return None
        items = resp if isinstance(resp, list) else resp.get("instances", resp.get("data", []))
        for item in items or []:
            if isinstance(item, dict) and item.get(FIELDS["name"]) == name:
                return str(item.get(FIELDS["instance_id"]) or item.get("id"))
        return None
