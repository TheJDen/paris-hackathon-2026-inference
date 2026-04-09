"""DP=N reverse proxy for the paris-hackathon-2026-inference engine.

Sits in front of N independent engine instances (one per GPU) and provides
a single OpenAI-compatible endpoint to clients. This is pure process-level
data-parallelism: zero NCCL, no model surgery, no shared state between the
backend ranks.

Routing:
  - POST /v1/chat/completions  — least-loaded backend (inflight counter),
                                  with automatic failover around down ranks.
  - GET  /health               — 200 if ALL backends are healthy; 503 otherwise.
  - GET  /metrics              — aggregated snapshot: throughput counters
                                  summed, latency percentiles from rank-0.
  - GET  /metrics/regions      — region tables from all backends concatenated,
                                  each prefixed with [rank=N].

Failure handling:
  - On a connection error or 5xx to a backend, that backend is marked DOWN
    for BACKEND_COOLDOWN_S seconds, then re-tried automatically.
  - Requests are never silently dropped; if ALL backends are down the proxy
    returns 503.

Run:
    python -m server.dp_proxy --port 8765 \
        --backends http://localhost:9000 ... http://localhost:9004
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from typing import Sequence

from aiohttp import web, ClientSession, ClientError, ClientTimeout, TCPConnector

log = logging.getLogger("dp_proxy")

# How long to mark a backend as down after a connection-level failure (seconds).
BACKEND_COOLDOWN_S: float = 10.0

# Timeout for forwarding a single chat completion request to a backend.
# Large enough for long generations; tune via HEALTH_TIMEOUT / start_dp.sh.
FORWARD_TIMEOUT_S: float = 600.0

# Timeout for health checks sent to backends (quick).
HEALTH_TIMEOUT_S: float = 5.0

# Timeout for metrics fetches (should be fast).
METRICS_TIMEOUT_S: float = 10.0

DEFAULT_PROXY_PORT: int = 8765
DEFAULT_BACKENDS: list[str] = [f"http://localhost:{9000 + i}" for i in range(5)]


# ---------------------------------------------------------------------------
# Backend health + load tracker
# ---------------------------------------------------------------------------

class BackendState:
    """Tracks liveness and in-flight request count for a single backend URL."""

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")
        self._down_until: float = 0.0  # epoch-seconds; 0 means healthy
        self.inflight: int = 0         # active forwarded requests
        self.picks_total: int = 0      # lifetime count of times this backend was picked

    def is_healthy(self) -> bool:
        return time.monotonic() >= self._down_until

    def mark_down(self, cooldown_s: float = BACKEND_COOLDOWN_S) -> None:
        self._down_until = time.monotonic() + cooldown_s
        log.warning("backend %s marked DOWN for %.0fs", self.url, cooldown_s)

    def mark_up(self) -> None:
        if self._down_until > 0:
            log.info("backend %s back UP", self.url)
        self._down_until = 0.0


# ---------------------------------------------------------------------------
# Proxy application
# ---------------------------------------------------------------------------

class DPProxy:
    """Asyncio single-threaded load balancer over DP=N engine backends.

    Uses a least-inflight routing policy among healthy backends to avoid
    hot-spotting when one rank is temporarily slower (e.g. cache miss burst).
    """

    def __init__(self, backends: list[str], port: int) -> None:
        self.port = port
        self._backends = [BackendState(url) for url in backends]
        self._session: ClientSession | None = None
        self._rr_index: int = 0  # round-robin tie-breaker among equal-load backends

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pick_backend(
        self, exclude_urls: set[str] | None = None
    ) -> BackendState | None:
        """Pick a healthy backend and ATOMICALLY claim it (inflight += 1).

        The caller is responsible for decrementing `inflight` in a finally block
        once the forwarded request completes (or errors).

        Why claim-at-pick? Under high concurrency (e.g. 64 requests fanning out
        in a single event-loop tick), all handlers may call this between
        awaits. If we incremented `inflight` only after the upstream await,
        every concurrent handler would observe the same `min(inflight=0)` and
        stampede onto rank 0. Claiming synchronously at pick time is the only
        way to make least-loaded routing actually spread load.

        Ties are broken via round-robin so that N simultaneous picks with all
        inflight==0 go to N distinct backends.
        """
        healthy = [
            b for b in self._backends
            if b.is_healthy() and (exclude_urls is None or b.url not in exclude_urls)
        ]
        if not healthy:
            return None

        min_inflight = min(b.inflight for b in healthy)
        candidates = [b for b in healthy if b.inflight == min_inflight]

        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            # Round-robin among equally-loaded backends. Walk _backends in
            # index order starting from _rr_index so the cycle is stable.
            n = len(self._backends)
            chosen = candidates[0]
            for offset in range(n):
                idx = (self._rr_index + offset) % n
                cand = self._backends[idx]
                if cand in candidates:
                    chosen = cand
                    self._rr_index = (idx + 1) % n
                    break

        chosen.inflight += 1
        chosen.picks_total += 1
        return chosen

    def _healthy_backends(self) -> list[BackendState]:
        return [b for b in self._backends if b.is_healthy()]

    async def _get_session(self) -> ClientSession:
        if self._session is None or self._session.closed:
            # Use a shared connection pool: limit_per_host avoids fd exhaustion
            # at high concurrency, but is generous enough for DP=8.
            connector = TCPConnector(limit_per_host=256, limit=1024)
            self._session = ClientSession(connector=connector)
        return self._session

    # ------------------------------------------------------------------
    # Route handlers
    # ------------------------------------------------------------------

    async def handle_chat(self, request: web.Request) -> web.StreamResponse:
        """POST /v1/chat/completions — forward to the least-loaded healthy backend.

        Handles both streaming (SSE) and non-streaming responses transparently.
        In-flight counter is incremented before the upstream call and decremented
        in a finally block so the counter is accurate even on exceptions.
        """
        body = await request.read()
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length", "transfer-encoding")
        }

        # Peek at whether the client wants streaming so we can propagate it.
        try:
            payload = json.loads(body)
            is_streaming = bool(payload.get("stream", False))
        except Exception:
            is_streaming = False

        timeout = ClientTimeout(total=FORWARD_TIMEOUT_S)
        session = await self._get_session()

        n = len(self._backends)
        attempted_urls: set[str] = set()
        last_err: str = "no backends available"

        while len(attempted_urls) < n:
            backend = self._pick_backend(exclude_urls=attempted_urls)
            if backend is None:
                break
            attempted_urls.add(backend.url)

            url = f"{backend.url}/v1/chat/completions"
            rank = self._backends.index(backend)
            try:
                async with session.post(
                    url,
                    data=body,
                    headers=headers,
                    timeout=timeout,
                    allow_redirects=False,
                ) as resp:
                    if resp.status >= 500:
                        text = await resp.text()
                        log.warning(
                            "backend %s returned %d: %s", backend.url, resp.status, text[:200]
                        )
                        backend.mark_down()
                        last_err = f"backend {backend.url} returned {resp.status}"
                        continue

                    backend.mark_up()

                    # Build response headers to pass through.
                    resp_headers = {
                        k: v for k, v in resp.headers.items()
                        if k.lower() not in (
                            "content-length", "transfer-encoding", "connection"
                        )
                    }
                    resp_headers["X-DP-Rank"] = str(rank)

                    if is_streaming:
                        # Stream SSE chunks back to client without buffering.
                        stream_resp = web.StreamResponse(
                            status=resp.status,
                            headers=resp_headers,
                        )
                        await stream_resp.prepare(request)
                        async for chunk in resp.content.iter_any():
                            await stream_resp.write(chunk)
                        await stream_resp.write_eof()
                        return stream_resp
                    else:
                        resp_body = await resp.read()
                        # Drop Content-Type from forwarded headers — aiohttp
                        # forbids passing both `headers` and `content_type`.
                        clean_headers = {
                            k: v for k, v in resp_headers.items()
                            if k.lower() != "content-type"
                        }
                        return web.Response(
                            status=resp.status,
                            body=resp_body,
                            headers=clean_headers,
                            content_type=resp.content_type or "application/json",
                        )
            except (ClientError, asyncio.TimeoutError, OSError) as exc:
                log.warning("backend %s connection error: %s", backend.url, exc)
                backend.mark_down()
                last_err = str(exc)
                continue
            finally:
                backend.inflight -= 1

        log.error("all backends failed or down; last error: %s", last_err)
        return web.Response(
            status=503,
            text=json.dumps({"error": f"all backends unavailable: {last_err}"}),
            content_type="application/json",
        )

    async def handle_health(self, request: web.Request) -> web.Response:
        """GET /health — probe ALL backends; return 200 only when all are alive.

        Individual backend liveness is updated as a side effect so the routing
        table stays current between benchmark calls.
        """
        timeout = ClientTimeout(total=HEALTH_TIMEOUT_S)
        session = await self._get_session()

        async def _check(b: BackendState) -> bool:
            try:
                async with session.get(
                    f"{b.url}/health", timeout=timeout, allow_redirects=False
                ) as resp:
                    ok = resp.status == 200
                    if ok:
                        b.mark_up()
                    else:
                        b.mark_down(BACKEND_COOLDOWN_S)
                    return ok
            except Exception:
                b.mark_down(BACKEND_COOLDOWN_S)
                return False

        results = await asyncio.gather(*(_check(b) for b in self._backends))
        healthy = sum(results)
        total = len(self._backends)

        if healthy == total:
            return web.Response(
                status=200,
                text=json.dumps({
                    "status": "ok",
                    "healthy_backends": healthy,
                    "total_backends": total,
                }),
                content_type="application/json",
            )
        return web.Response(
            status=503,
            text=json.dumps({
                "status": "degraded",
                "healthy_backends": healthy,
                "total_backends": total,
            }),
            content_type="application/json",
        )

    async def handle_metrics(self, request: web.Request) -> web.Response:
        """GET /metrics — sum throughput counters across all backends.

        Strategy:
          - Fetch /metrics JSON from every backend in parallel.
          - Sum all numeric fields (throughput counters, token counts, etc.).
          - For latency percentiles (p50/p99) use rank-0's values as a
            representative sample — summing percentiles is statistically
            meaningless and the per-backend values are similar at steady state.
          - Add synthetic 'dp_healthy_backends' and per-rank inflight keys.
        """
        timeout = ClientTimeout(total=METRICS_TIMEOUT_S)
        session = await self._get_session()

        async def _fetch(b: BackendState) -> dict | None:
            try:
                async with session.get(f"{b.url}/metrics", timeout=timeout) as resp:
                    if resp.status == 200:
                        return await resp.json(content_type=None)
            except Exception:
                pass
            return None

        snapshots = await asyncio.gather(*(_fetch(b) for b in self._backends))
        valid = [s for s in snapshots if s is not None]

        if not valid:
            return web.Response(
                status=503,
                text=json.dumps({"error": "no backends reachable"}),
                content_type="application/json",
            )

        # Keys whose values we sum across ranks.
        SUMMABLE = {
            "requests_total",
            "requests_failed",
            "prompt_tokens_total",
            "completion_tokens_total",
            "tok_per_s_recent",
            "tok_per_s_lifetime",
            "prompt_tok_per_s_lifetime",
            "completion_tok_per_s_lifetime",
            "batch_tok_per_s_p50",
            "batch_tok_per_s_p99",
            "running",
            "waiting",
            "kv_used",
            "kv_total",
            "state_used",
            "state_total",
        }

        aggregated: dict = {}
        # Use rank-0 (first valid) as base for non-summable fields.
        aggregated.update(valid[0])

        for key in SUMMABLE:
            total = 0.0
            for snap in valid:
                val = snap.get(key, 0)
                try:
                    total += float(val)
                except (TypeError, ValueError):
                    pass
            orig = valid[0].get(key, 0)
            aggregated[key] = int(total) if isinstance(orig, int) else round(total, 1)

        aggregated["dp_healthy_backends"] = len(valid)
        aggregated["dp_total_backends"] = len(self._backends)
        aggregated["dp_inflight"] = {
            str(i): b.inflight for i, b in enumerate(self._backends)
        }
        aggregated["dp_picks_total"] = {
            str(i): b.picks_total for i, b in enumerate(self._backends)
        }

        return web.Response(
            status=200,
            text=json.dumps(aggregated),
            content_type="application/json",
        )

    async def handle_metrics_regions(self, request: web.Request) -> web.Response:
        """GET /metrics/regions — concatenate region tables with [rank=N] headers."""
        sort_by = request.rel_url.query.get("sort_by", "total")
        timeout = ClientTimeout(total=METRICS_TIMEOUT_S)
        session = await self._get_session()

        async def _fetch(rank: int, b: BackendState) -> tuple[int, str | None]:
            try:
                async with session.get(
                    f"{b.url}/metrics/regions",
                    params={"sort_by": sort_by},
                    timeout=timeout,
                ) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        return rank, text
            except Exception:
                pass
            return rank, None

        results = await asyncio.gather(
            *(_fetch(i, b) for i, b in enumerate(self._backends))
        )

        parts: list[str] = []
        for rank, text in results:
            if text is not None:
                parts.append(f"[rank={rank}]\n{text}")
            else:
                parts.append(f"[rank={rank}] (unavailable)")

        body = "\n\n".join(parts)
        return web.Response(status=200, text=body, content_type="text/plain")

    # ------------------------------------------------------------------
    # App factory + lifecycle
    # ------------------------------------------------------------------

    def build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.handle_chat)
        app.router.add_get("/health", self.handle_health)
        app.router.add_get("/metrics", self.handle_metrics)
        app.router.add_get("/metrics/regions", self.handle_metrics_regions)

        async def _on_shutdown(app: web.Application) -> None:
            # Log pick distribution so we can verify least-loaded routing
            # actually spread traffic across backends.
            dist = ", ".join(
                f"rank{i}={b.picks_total}" for i, b in enumerate(self._backends)
            )
            log.info("dp_proxy shutdown pick distribution: %s", dist)
            if self._session and not self._session.closed:
                await self._session.close()

        app.on_shutdown.append(_on_shutdown)
        return app

    def run(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )
        log.info(
            "dp_proxy starting on port %d with %d backends: %s",
            self.port,
            len(self._backends),
            [b.url for b in self._backends],
        )
        app = self.build_app()
        web.run_app(app, port=self.port, access_log=None)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DP=N round-robin reverse proxy for paris-hackathon-2026-inference"
    )
    p.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PROXY_PORT,
        help=f"front-end port (default {DEFAULT_PROXY_PORT})",
    )
    p.add_argument(
        "--backends",
        nargs="+",
        default=DEFAULT_BACKENDS,
        metavar="URL",
        help="backend base URLs (default: http://localhost:9000..9004)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    proxy = DPProxy(backends=args.backends, port=args.port)
    proxy.run()


if __name__ == "__main__":
    main()
