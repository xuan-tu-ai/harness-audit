"""Run isolation for the Claude adapter: hidden directories and the network gate."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from harness_audit.adapters import claude_agent_sdk as mod
from harness_audit.adapters.netfilter import allowed


def test_gate_allows_only_listed_hosts_on_443() -> None:
    allow = {"tokenharbor.ai"}
    assert allowed("tokenharbor.ai:443", allow) and allowed("TokenHarbor.ai:443", allow)
    assert not allowed("tokenharbor.ai:80", allow)
    assert not allowed("raw.githubusercontent.com:443", allow)
    assert not allowed("evil.tokenharbor.ai:443", allow)


def test_isolation_env_and_defaults() -> None:
    env = mod.isolation_env(
        "/usr/bin/claude", ["gw.example"], {"NODE_EXTRA_CA_CERTS": "/root/ca.crt", "SSL_CERT_FILE": "/etc/ssl/c.pem"}
    )
    assert env == {
        "HARNESS_AUDIT_CLAUDE_BIN": "/usr/bin/claude",
        "HARNESS_AUDIT_ALLOW_HOSTS": "gw.example",
        "HARNESS_AUDIT_KEEP_FILES": "/root/ca.crt",  # only files under hidden directories are copied in
    }
    assert mod.ClaudeAgentSDKAdapter(base_url="https://tokenharbor.ai", isolate=True).allow_hosts == ["tokenharbor.ai"]
    assert mod.ClaudeAgentSDKAdapter(isolate=True).allow_hosts == ["api.anthropic.com"]


def test_cli_under_a_hidden_directory_is_refused() -> None:
    from harness_audit.adapters import FatalAdapterError

    with pytest.raises(FatalAdapterError, match="hidden"):
        mod.ClaudeAgentSDKAdapter(isolate=True, claude_bin="/tmp/claude")._isolation_bin()


def _can_isolate() -> bool:
    if not shutil.which("unshare") or os.geteuid() != 0:
        return False
    return subprocess.run(["unshare", "--mount", "--pid", "--net", "--fork", "true"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _can_isolate(), reason="needs root and mount/PID/network namespaces")
def test_isolate_script_hides_files_and_network(tmp_path: Path) -> None:
    secret = Path("/tmp") / f"harness-audit-secret-{os.getpid()}.txt"
    secret.write_text("hidden test")
    work, cfg = tmp_path / "work", tmp_path / "cfg"
    work.mkdir()
    cfg.mkdir()
    (work / "mine.txt").write_text("visible")
    script = (
        "cat mine.txt; echo; test -e " + str(secret) + " && echo LEAK; ls /home | wc -l; "
        "echo written > out.txt; "
        "python3 -I -c \"import urllib.request as u; u.urlopen('https://example.com', timeout=5)\" 2>/dev/null || echo blocked"
    )
    env = {
        **os.environ,
        "CLAUDE_CONFIG_DIR": str(cfg),
        "HARNESS_AUDIT_CLAUDE_BIN": "/bin/sh",
        "HARNESS_AUDIT_ALLOW_HOSTS": "tokenharbor.ai",
    }
    try:
        out = subprocess.run(
            [str(mod.ISOLATE_SCRIPT), "-c", script], cwd=work, env=env, capture_output=True, text=True, timeout=60
        )
    finally:
        secret.unlink()
    assert out.returncode == 0, out.stderr
    lines = out.stdout.split()
    assert lines[0] == "visible" and "LEAK" not in lines and lines[1] == "0" and lines[-1] == "blocked"
    assert (work / "out.txt").read_text() == "written\n"  # the run's own files reach the scorer
