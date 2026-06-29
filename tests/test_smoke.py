"""Fast CPU-only smoke test for the cloudbench harness.

Verifies the registry loads, the config resolves, and the local provider
can run a trivial command end-to-end (a real DRY_RUN-free local run).
"""

from __future__ import annotations

import json

from cloudbench.benchmark_registry import get_benchmark, load_benchmarks
from cloudbench.config import REPO_ROOT, build_run_config
from cloudbench.providers import get_provider
from cloudbench.runner import git_info, write_results


def test_minimal_yaml_fallback_matches_schema():
    # The stdlib fallback must parse our benchmarks.yaml shape without PyYAML.
    from cloudbench.benchmark_registry import _minimal_yaml_parse
    text = (
        "smoke:\n"
        "  command: python -m pytest -q tests/test_smoke.py\n"
        "  gpu: none\n"
        "  ngpus: 0\n"
        "  timeout_minutes: 10\n"
        "# a comment line\n"
        "e2e:\n"
        "  command: torchrun --nproc_per_node=8 serve.py\n"
        "  ngpus: 8\n"
    )
    data = _minimal_yaml_parse(text)
    assert data["smoke"]["command"] == "python -m pytest -q tests/test_smoke.py"
    assert data["smoke"]["ngpus"] == 0
    assert data["e2e"]["ngpus"] == 8
    assert data["e2e"]["command"] == "torchrun --nproc_per_node=8 serve.py"


def test_dotenv_parser_strips_inline_comments(tmp_path):
    from cloudbench.config import _parse_dotenv
    f = tmp_path / "providers.conf"
    f.write_text(
        "# full line comment\n"
        "SESTERCE_SSH_KEY_ID=6a409101c7d985540fad72ff   # registered key\n"
        "SESTERCE_REGION=paris-france-3\n"
        'QUOTED="has # hash inside"  # trailing comment\n'
        "export EXPORTED=value\n"
    )
    d = _parse_dotenv(f)
    assert d["SESTERCE_SSH_KEY_ID"] == "6a409101c7d985540fad72ff"
    assert d["SESTERCE_REGION"] == "paris-france-3"
    assert d["QUOTED"] == "has # hash inside"
    assert d["EXPORTED"] == "value"


def test_config_split_precedence(tmp_path):
    # providers.conf is the base; .env overlays (secrets win).
    from cloudbench.config import load_env
    conf = tmp_path / "providers.conf"
    env = tmp_path / ".env"
    conf.write_text("SESTERCE_REGION=paris-france-3\nSESTERCE_BASE_URL=https://x\n")
    env.write_text("SESTERCE_API_KEY=secret123\n")
    merged = load_env(env_file=env, providers_conf=conf)
    assert merged["SESTERCE_REGION"] == "paris-france-3"   # from providers.conf
    assert merged["SESTERCE_API_KEY"] == "secret123"        # from .env


def test_registry_loads_expected_benches():
    benches = load_benchmarks()
    for name in ("smoke", "decode_smoke", "single_h200_decode",
                 "single_h200_prefill", "e2e_8xh200"):
        assert name in benches, f"missing benchmark {name}"
    assert benches["e2e_8xh200"].ngpus == 8
    assert benches["smoke"].ngpus == 0


def test_bench_cmd_override():
    spec = get_benchmark("decode_smoke")
    cfg = build_run_config("local", spec, bench_cmd="echo hi", env={"USER": "tester"})
    assert cfg.command == "echo hi"
    assert cfg.resource_name.startswith("paris-bench-tester-")
    assert cfg.resource_name.endswith("-decode_smoke")


def test_local_run_end_to_end(tmp_path):
    spec = get_benchmark("smoke")
    cfg = build_run_config("local", spec, bench_cmd="echo cloudbench-ok",
                           env={"USER": "tester"})
    cfg.run_dir = tmp_path / "run"

    provider = get_provider("local", cfg)
    provider.provision()
    result = provider.execute()
    assert result.exit_code == 0
    assert "cloudbench-ok" in result.stdout

    import datetime as dt
    now = dt.datetime.now()
    meta = write_results(cfg, git_info(), provider.resource, result, now, now, True)
    assert meta["provider"] == "local"
    assert meta["exit_code"] == 0
    assert (cfg.run_dir / "metadata.json").exists()
    assert (cfg.run_dir / "provider.json").exists()

    loaded = json.loads((cfg.run_dir / "metadata.json").read_text())
    assert loaded["bench"] == "smoke"


def test_dry_run_local():
    spec = get_benchmark("single_h200_decode")
    cfg = build_run_config("local", spec, env={"USER": "tester", "DRY_RUN": "1"})
    assert cfg.dry_run is True
    provider = get_provider("local", cfg)
    provider.provision()
    result = provider.execute()
    assert result.exit_code == 0
    assert "DRY_RUN" in result.stdout


# --------------------------------------------------------------------------- #
# Cleanup / interruption safety — the priority is NEVER leaking a GPU.
# --------------------------------------------------------------------------- #

from cloudbench import ledger  # noqa: E402
from cloudbench.config import build_run_config as _brc  # noqa: E402
from cloudbench.providers import CloudProvider, reap_record  # noqa: E402
from cloudbench.providers._remote import SSHInfo  # noqa: E402


class _FakeCloud(CloudProvider):
    """A cloud provider with no real backend, for cleanup testing."""
    name = "fake"
    DESTROY_ATTEMPTS = 3

    def __init__(self, cfg, *, interrupt_create=False, lose_id=False,
                 destroy_fails=0):
        super().__init__(cfg)
        self.destroy_calls = 0
        self.interrupt_create = interrupt_create
        self.lose_id = lose_id
        self.destroy_fails = destroy_fails

    def create_instance(self):
        # Simulate a real backend that creates the resource first...
        if self.interrupt_create:
            # ...then we get killed before the id is recorded.
            raise KeyboardInterrupt("killed mid-create")
        if not self.lose_id:
            self.instance_id = "i-123"

    def wait_ready(self):
        pass

    def _wait_for_ssh(self, timeout=300):
        pass  # no real host to SSH to in tests

    def get_ssh_info(self):
        return SSHInfo(host="10.0.0.1")

    def destroy_instance(self):
        self.destroy_calls += 1
        if self.destroy_calls <= self.destroy_fails:
            return False
        return True

    def find_instance_id_by_name(self, name):
        return "i-recovered"


def _fake_cfg():
    return _brc("fake", get_benchmark("decode_smoke"), env={"USER": "tester"})


def _point_ledger_at_tmp(monkeypatch, tmp_path):
    monkeypatch.setattr(ledger, "ACTIVE_DIR", tmp_path / "active")
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)


def test_normal_run_destroys_and_clears_ledger(monkeypatch, tmp_path):
    _point_ledger_at_tmp(monkeypatch, tmp_path)
    prov = _FakeCloud(_fake_cfg())
    prov.provision()
    assert ledger.list_records(), "provision should write a ledger record"
    assert prov.cleanup() is True
    assert prov.destroy_calls == 1
    assert ledger.list_records() == [], "successful destroy should clear ledger"


def test_interrupt_during_create_still_destroys(monkeypatch, tmp_path):
    """The dangerous window: killed mid-create before the id is captured."""
    _point_ledger_at_tmp(monkeypatch, tmp_path)
    prov = _FakeCloud(_fake_cfg(), interrupt_create=True)
    try:
        prov.provision()
    except KeyboardInterrupt:
        pass
    assert prov._created is True, "ownership must be claimed before create"
    assert ledger.list_records(), "a breadcrumb must exist even mid-create"
    assert prov.cleanup() is True
    # id was never captured, so it must be recovered by name and destroyed.
    assert prov.destroy_calls == 1
    assert ledger.list_records() == []


def test_destroy_retries_until_success(monkeypatch, tmp_path):
    _point_ledger_at_tmp(monkeypatch, tmp_path)
    prov = _FakeCloud(_fake_cfg(), destroy_fails=2)
    prov.provision()
    assert prov.cleanup() is True
    assert prov.destroy_calls == 3  # 2 failures + 1 success


def test_destroy_failure_keeps_ledger_record(monkeypatch, tmp_path):
    _point_ledger_at_tmp(monkeypatch, tmp_path)
    prov = _FakeCloud(_fake_cfg(), destroy_fails=99)
    prov.provision()
    assert prov.cleanup() is False
    # Record MUST survive so `reap` can finish the job.
    assert ledger.list_records(), "failed destroy must leave the ledger record"


def test_keep_instance_leaves_resource_in_ledger(monkeypatch, tmp_path):
    _point_ledger_at_tmp(monkeypatch, tmp_path)
    cfg = _brc("fake", get_benchmark("decode_smoke"),
               env={"USER": "tester", "KEEP_INSTANCE": "1"})
    prov = _FakeCloud(cfg)
    prov.provision()
    assert prov.cleanup() is True
    assert prov.destroy_calls == 0, "KEEP_INSTANCE must not destroy"
    recs = ledger.list_records()
    assert recs and recs[0]["keep"] is True


def test_profile_command_wrapping():
    from cloudbench.config import wrap_command_for_profile
    base = "python bench_decode.py --tokens 64"
    assert wrap_command_for_profile(base, "none", None, "artifacts", "r") == base

    nsys = wrap_command_for_profile(base, "nsys", None, "artifacts", "myrun")
    assert nsys.startswith("nsys profile")
    assert "-o artifacts/myrun_nsys" in nsys
    assert nsys.endswith(base)

    ncu = wrap_command_for_profile(base, "ncu", "--section SpeedOfLight",
                                   "artifacts", "myrun")
    assert ncu.startswith("ncu --set full")
    assert "-o artifacts/myrun_ncu" in ncu
    assert "--section SpeedOfLight" in ncu


def test_profile_flows_into_run_config():
    spec = get_benchmark("single_h200_decode")
    cfg = build_run_config("sesterce", spec,
                           env={"USER": "tester", "PROFILE": "nsys"})
    assert cfg.profile == "nsys"
    assert cfg.command.startswith("nsys profile")
    assert cfg.base_command == spec.command


def test_local_run_collects_artifacts(tmp_path):
    # A command that writes a fake profiler/chrome-trace file into artifacts/.
    spec = get_benchmark("smoke")
    cmd = ("mkdir -p artifacts && echo '{\"traceEvents\":[]}' > "
           "artifacts/trace.json && echo done")
    cfg = build_run_config("local", spec, bench_cmd=cmd, env={"USER": "tester"})
    cfg.run_dir = tmp_path / "run"

    provider = get_provider("local", cfg)
    provider.provision()
    result = provider.execute()
    assert result.exit_code == 0
    assert any("trace.json" in a for a in result.artifacts)
    assert (cfg.run_dir / "artifacts" / "trace.json").exists()

    # Clean up the artifacts dir this test created in the repo root.
    import shutil
    shutil.rmtree(REPO_ROOT / "artifacts", ignore_errors=True)


def test_reap_dry_run_does_not_destroy(monkeypatch, tmp_path):
    _point_ledger_at_tmp(monkeypatch, tmp_path)
    ledger.write_record({"resource_name": "paris-bench-x", "provider": "nebius",
                         "instance_id": "i-9", "ngpus": 1, "gpu": "H200"})
    ok = reap_record(ledger.list_records()[0], env={"USER": "t"}, dry_run=True)
    assert ok is True
    assert ledger.list_records(), "dry-run reap must not remove records"
