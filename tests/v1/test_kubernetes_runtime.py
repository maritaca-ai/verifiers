"""Kubernetes runtime against a fake `kubectl` on PATH (hermetic: no cluster).

The fake records every invocation (argv + stdin) to a log and emulates the
handful of subcommands the runtime uses: create, wait, exec, get, delete.
"""

import asyncio
import json
import os
import stat
from pathlib import Path

import pytest

from verifiers.v1.runtimes import KubernetesConfig, make_runtime, runtime_is_local
import verifiers.v1.runtimes.kubernetes as kubernetes_mod
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.kubernetes import KubernetesRuntime

FAKE_KUBECTL = r'''#!/usr/bin/env python3
import json, os, sys
log = os.environ["FAKE_KUBECTL_LOG"]
args = sys.argv[1:]
stdin = sys.stdin.buffer.read() if not sys.stdin.isatty() else b""
with open(log, "a") as fh:
    fh.write(json.dumps({"args": args, "stdin": stdin.decode(errors="replace")}) + "\n")
# strip global flags
while args and args[0] in ("-n", "--context", "--kubeconfig"):
    args = args[2:]
cmd = args[0] if args else ""
if cmd == "create":
    print("pod/x created"); sys.exit(0)
if cmd == "wait":
    sys.exit(int(os.environ.get("FAKE_WAIT_RC", "0")))
if cmd == "get":
    print(os.environ.get("FAKE_PHASE", "Running")); sys.exit(0)
if cmd == "delete":
    sys.exit(0)
if cmd == "exec":
    # argv after "--": sh -c <prelude> vf-exec <workdir> <program...>
    prog = args[args.index("--") + 1:]
    inner = prog[5:]
    fail_first = int(os.environ.get("FAKE_EXEC_TRANSIENT_FIRST", "0"))
    marker = os.environ.get("FAKE_EXEC_COUNTER", "")
    if fail_first and marker:
        n = int(open(marker).read() or "0") if os.path.exists(marker) else 0
        open(marker, "w").write(str(n + 1))
        if n < fail_first:
            if os.environ.get("FAKE_EXEC_PARTIAL_STDOUT"):
                sys.stdout.write("partial output\n")
            sys.stderr.write(os.environ.get("FAKE_EXEC_TRANSIENT_MSG",
                                            "error: unable to upgrade connection: container not found") + "\n")
            sys.exit(1)
    if inner[:2] == ["cat", "/tmp/answer"]:
        sys.stdout.buffer.write(b"42\n"); sys.exit(0)
    if inner[:2] == ["head", "-c"] and inner[3] == "/tmp/answer":
        sys.stdout.buffer.write(b"42\n"[:int(inner[2])]); sys.exit(0)
    if inner[0] == "sh" and "cat >" in inner[2]:
        sys.exit(0)
    if inner[0] == "false":
        sys.stderr.write("command terminated with exit code 1\n"); sys.exit(1)
    print("ran:", " ".join(inner)); sys.exit(0)
sys.exit(2)
'''


@pytest.fixture
def fake_kubectl(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "kubectl"
    exe.write_text(FAKE_KUBECTL)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "kubectl.log"
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_KUBECTL_LOG", str(log))
    monkeypatch.setenv("FAKE_EXEC_COUNTER", str(tmp_path / "counter"))
    monkeypatch.delenv("FAKE_EXEC_TRANSIENT_FIRST", raising=False)
    monkeypatch.delenv("FAKE_EXEC_TRANSIENT_MSG", raising=False)
    monkeypatch.delenv("FAKE_EXEC_PARTIAL_STDOUT", raising=False)
    monkeypatch.setattr(kubernetes_mod, "_backoff", lambda attempt, base=1.0: 0.0)

    def calls():
        return [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []

    return calls


def _cfg(**kw):
    return KubernetesConfig(namespace="sandboxes", context="ctx", image="registry/img:tag", cpu=1, memory=4, **kw)


def test_is_remote_and_registered():
    assert not runtime_is_local(_cfg())
    rt = make_runtime(_cfg(), name="vf-abc123")
    assert isinstance(rt, KubernetesRuntime) and rt._pod == "vf-abc123"


def test_start_manifest_exec_env_write_read_teardown(fake_kubectl):
    rt = make_runtime(_cfg(labels={"run": "smoke"}, disk=2,
                           annotations={"cluster-autoscaler.kubernetes.io/safe-to-evict": "false"}),
                      name="vf-Test_01")
    rt.env = {"TASK_VAR": "x y"}

    async def flow():
        await rt.start()
        out = await rt.run(["echo", "hi"], {"K": "v"})
        await rt.write("/tmp/a/b.txt", b"payload")
        data = await rt.read("/tmp/answer")
        # Capped read: the base read() contract passes max_bytes to _read — the
        # taskset status/reward reads use it (missing support = TypeError =
        # every episode "agent died", exp2015 try 8).
        capped = await rt.read("/tmp/answer", max_bytes=16)
        from verifiers.v1.errors import SandboxError
        try:
            await rt.read("/tmp/answer", max_bytes=2)
            overflow = False
        except SandboxError:
            overflow = True
        alive = await rt.alive()
        await rt.stop()
        return out, data, alive, capped, overflow

    out, data, alive, capped, overflow = asyncio.run(flow())
    assert capped == b"42\n" and overflow is True
    head_call = next(c for c in fake_kubectl() if "head" in c["args"])
    assert head_call["args"][head_call["args"].index("head"):][:3] == ["head", "-c", "17"]
    calls = fake_kubectl()
    create = calls[0]
    assert create["args"][:2] == ["-n", "sandboxes"] and "--context" in create["args"]
    manifest = json.loads(create["stdin"])
    assert manifest["metadata"]["name"] == "vf-test-01"          # DNS-1123 normalized
    c = manifest["spec"]["containers"][0]
    assert c["image"] == "registry/img:tag" and c["command"] == ["sleep", "infinity"]
    assert c["resources"]["requests"] == {"cpu": "1.0", "memory": "4096Mi", "ephemeral-storage": "2048Mi"}
    assert c["resources"]["limits"] == c["resources"]["requests"]
    # requests below limits (Autopilot bursting): billed on the request, capped at the limit
    burst = KubernetesRuntime(_cfg(cpu_request=0.25, memory_request=1.0, disk=2), name="vf-b")._manifest()
    bc = burst["spec"]["containers"][0]["resources"]
    assert bc["requests"] == {"cpu": "0.25", "memory": "1024Mi", "ephemeral-storage": "2048Mi"}
    assert bc["limits"] == {"cpu": "1.0", "memory": "4096Mi", "ephemeral-storage": "2048Mi"}
    assert manifest["metadata"]["labels"] == {"app": "vf-sandbox", "run": "smoke"}
    assert manifest["metadata"]["annotations"] == {"cluster-autoscaler.kubernetes.io/safe-to-evict": "false"}
    assert calls[1]["args"][calls[1]["args"].index("wait"):][:2] == ["wait", "pod/vf-test-01"]
    ex = calls[2]["args"]
    prog = ex[ex.index("--") + 1:]
    assert prog[0:2] == ["sh", "-c"] and prog[3:5] == ["vf-exec", "/app"] and prog[5:] == ["echo", "hi"]
    assert "TASK_VAR='x y'" in prog[2] and "K=v" in prog[2]     # runtime env + call env
    assert out.exit_code == 0 and out.stdout.strip() == "ran: echo hi"
    wr = calls[3]
    assert "-i" in wr["args"] and wr["stdin"] == "payload" and "mkdir -p /tmp/a && cat > /tmp/a/b.txt" in " ".join(wr["args"])
    assert data == b"42\n" and alive is True
    assert any("delete" in c["args"] and "--grace-period=0" in c["args"] for c in calls)


WEBHOOK_TIMEOUT = ('error: unable to upgrade connection: Internal error occurred: failed calling webhook '
                   '"warden-validating.common-webhooks.networking.gke.io": failed to call webhook: '
                   'Post "https://localhost:5443/webhook/warden-validating?timeout=10s": context deadline exceeded')


def test_transient_exec_is_retried_and_program_retries_only_refused_starts(fake_kubectl, monkeypatch):
    rt = make_runtime(_cfg(exec_retries=2), name="vf-r")
    monkeypatch.setenv("FAKE_EXEC_TRANSIENT_FIRST", "1")

    async def flow():
        await rt.start()
        first = await rt.run(["echo", "ok"], {})          # 1 transient failure, then success
        monkeypatch.setenv("FAKE_EXEC_COUNTER", str(Path(os.environ["FAKE_EXEC_COUNTER"]).with_name("c2")))
        prog = await rt.run_program(["echo", "ok"], {})   # refused before start: retried, nothing replayed
        return first, prog

    first, prog = asyncio.run(flow())
    assert first.exit_code == 0 and prog.exit_code == 0 and "ran: echo ok" in prog.stdout
    execs = [c for c in fake_kubectl() if "exec" in c["args"]]
    assert len(execs) == 4  # 2 for run + 2 for run_program


def test_webhook_timeout_is_retried_for_program_read_and_write(fake_kubectl, monkeypatch):
    rt = make_runtime(_cfg(exec_retries=3), name="vf-w")
    monkeypatch.setenv("FAKE_EXEC_TRANSIENT_MSG", WEBHOOK_TIMEOUT)
    counters = iter(range(100))

    def fail_next(n):
        monkeypatch.setenv("FAKE_EXEC_COUNTER", str(Path(os.environ["FAKE_EXEC_COUNTER"]).with_name(f"w{next(counters)}")))
        monkeypatch.setenv("FAKE_EXEC_TRANSIENT_FIRST", str(n))

    async def flow():
        await rt.start()
        fail_next(2)
        prog = await rt.run_program(["echo", "ok"], {})
        fail_next(2)
        await rt.write("/tmp/x", b"data")
        fail_next(2)
        data = await rt.read("/tmp/answer")
        return prog, data

    prog, data = asyncio.run(flow())
    assert prog.exit_code == 0 and data == b"42\n"


def test_program_failure_after_start_is_not_replayed(fake_kubectl, monkeypatch):
    rt = make_runtime(_cfg(exec_retries=3), name="vf-p")

    async def flow():
        await rt.start()
        monkeypatch.setenv("FAKE_EXEC_TRANSIENT_FIRST", "1")
        monkeypatch.setenv("FAKE_EXEC_PARTIAL_STDOUT", "1")      # the program had started and printed
        started = await rt.run_program(["echo", "ok"], {})
        monkeypatch.setenv("FAKE_EXEC_COUNTER", str(Path(os.environ["FAKE_EXEC_COUNTER"]).with_name("c3")))
        monkeypatch.delenv("FAKE_EXEC_PARTIAL_STDOUT")
        monkeypatch.setenv("FAKE_EXEC_TRANSIENT_MSG", "error: unexpected EOF")  # mid-stream loss
        mid = await rt.run_program(["echo", "ok"], {})
        return started, mid

    started, mid = asyncio.run(flow())
    assert started.exit_code == 1 and mid.exit_code == 1
    assert len([c for c in fake_kubectl() if "exec" in c["args"]]) == 2


def test_read_exhausts_retries_into_sandbox_error(fake_kubectl, monkeypatch):
    rt = make_runtime(_cfg(exec_retries=2), name="vf-x")

    async def flow():
        await rt.start()
        monkeypatch.setenv("FAKE_EXEC_TRANSIENT_FIRST", "99")
        await rt.read("/tmp/answer")

    with pytest.raises(SandboxError, match="read"):
        asyncio.run(flow())
    assert len([c for c in fake_kubectl() if "exec" in c["args"]]) == 3


def test_nonzero_command_exit_is_not_retried(fake_kubectl):
    rt = make_runtime(_cfg(), name="vf-f")

    async def flow():
        await rt.start()
        return await rt.run(["false"], {})

    res = asyncio.run(flow())
    assert res.exit_code == 1
    assert len([c for c in fake_kubectl() if "exec" in c["args"]]) == 1


def test_not_ready_raises_sandbox_error_and_deletes(fake_kubectl, monkeypatch):
    from verifiers.v1.errors import SandboxError
    monkeypatch.setenv("FAKE_WAIT_RC", "1")
    monkeypatch.setenv("FAKE_PHASE", "Pending")
    rt = make_runtime(_cfg(startup_timeout=5), name="vf-p")
    with pytest.raises(SandboxError, match="not Ready"):
        asyncio.run(rt.start())
    assert any("delete" in c["args"] for c in fake_kubectl())


def test_gpu_request_rejected(fake_kubectl):
    from verifiers.v1.errors import SandboxError
    rt = make_runtime(_cfg(gpu="A100"), name="vf-g")
    with pytest.raises(SandboxError, match="GPU"):
        asyncio.run(rt.start())


def test_admission_cap_bounds_in_flight_short_calls(monkeypatch):
    """`exec_concurrency` caps concurrent short kubectl calls process-wide (across
    runtimes), and a retrying call releases its slot while it sleeps."""
    from verifiers.v1.runtimes.base import ProgramResult

    state = {"now": 0, "peak": 0, "failed_once": set()}

    async def fake_kubectl(self, *args, stdin=None):
        state["now"] += 1
        state["peak"] = max(state["peak"], state["now"])
        await asyncio.sleep(0.01)
        state["now"] -= 1
        if self._pod not in state["failed_once"]:  # every runtime's first call is transient
            state["failed_once"].add(self._pod)
            return ProgramResult(exit_code=1, stdout="", stderr="error: unable to upgrade connection: x")
        return ProgramResult(exit_code=0, stdout="ok", stderr="")

    monkeypatch.setattr(KubernetesRuntime, "_kubectl", fake_kubectl)
    rts = [KubernetesRuntime(_cfg(exec_concurrency=3, exec_retries=1), name=f"vf-c{i}") for i in range(12)]

    async def flow():
        results = await asyncio.gather(*(rt.run(["true"], {}) for rt in rts),
                                       *(rt.write("/tmp/x", b"y") for rt in rts))
        return results

    results = asyncio.run(flow())
    assert all(r is None or r.exit_code == 0 for r in results)
    assert state["peak"] == 3

