"""Kubernetes runtime: one Pod per sandbox on a cluster reachable through `kubectl`.

The remote counterpart of the docker runtime for clusters we operate ourselves
(GKE Autopilot in our case): `start` creates a Pod from the task's image with
`sleep infinity` as its entrypoint and waits for it to be Ready, every `run`
is a `kubectl exec`, `write`/`read` stream through exec's stdin/stdout, and
`teardown` deletes the Pod. Everything goes through the `kubectl` CLI rather
than the Python client: the client's exec websocket is known to race under
concurrency, while `kubectl exec` is solid under load (one fork per call).

Remote by construction (`is_local = False`): programs inside the Pod reach the
host's interception server through the configured `Tunnel` — for a Pod that
shares a VPC with the host, a `custom` tunnel with `http://<host-ip>:<port>`
is enough (the server binds all interfaces on that port).

No framework network policy: the Pod's egress is the cluster's business
(NetworkPolicy / Autopilot); a task with a concrete `network_allow` is rejected
by `resolve_runtime_config` like any non-policy runtime — set `["*"]`.

Autopilot notes: requests must be set (they become the limits and the bill);
1 CPU : 1-6.5 GiB memory; image streaming makes Artifact Registry images start
in seconds, other registries pull cold per node.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import re
import shlex
import subprocess
from pathlib import PurePosixPath
from typing import ClassVar, Literal

from pydantic import Field
from pydantic_config import BaseConfig

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import BaseRuntimeInfo, ProgramResult, Runtime
from verifiers.v1.utils.aio import run_shielded

logger = logging.getLogger(__name__)

# kubectl exec failures that are the API server / kubelet proxy, not the command.
_TRANSIENT = (
    "unable to upgrade connection",
    "error dialing backend",
    "no agent available",
    "connection reset by peer",
    "tls: internal error",
    "context deadline exceeded",
    "TLS handshake timeout",
    "unexpected EOF",
    "i/o timeout",
    "connection refused",
    "the server is currently unable to handle the request",
    "too many requests",
    "failed calling webhook",
)
# The subset that fails BEFORE the command starts in the container: the exec
# upgrade itself was refused (admission webhook, kubelet proxy). Retrying such
# a failure cannot replay anything, so even the harness program may retry it.
_PRE_START = (
    "unable to upgrade connection",
    "error dialing backend",
    "no agent available",
    "failed calling webhook",
)
_EXEC_PREFIX = "command terminated with exit code "
_BACKOFF_CAP_S = 30.0


def _backoff(attempt: int, base: float = 1.0) -> float:
    """Exponential backoff with jitter. Under an overloaded admission webhook
    (GKE Autopilot's `warden` on pods/exec times out at 10 s when the cluster
    sees an exec storm) fixed short sleeps make every sandbox retry in lockstep
    and exhaust their budget inside one slow window; spread and stretch them."""
    return min(base * 2 ** attempt, _BACKOFF_CAP_S) * random.uniform(0.5, 1.5)


class KubernetesConfig(BaseConfig):
    type: Literal["kubernetes"] = "kubernetes"
    image: str = "python:3.11-slim"
    """Default image; a task's `image` overrides it (resolve_runtime_config)."""
    workdir: str = "/app"
    namespace: str = "default"
    context: str | None = None
    """kubeconfig context (None = current-context)."""
    kubeconfig: str | None = None
    """kubeconfig path (None = KUBECONFIG / ~/.kube/config)."""
    cpu: float | None = 1.0
    """CPU request (= limit). Autopilot minimum 0.25; rounded up by the platform."""
    memory: float | None = 4.0
    """Memory request (= limit) in GB."""
    disk: float | None = None
    cpu_request: float | None = None
    """Scheduling request for CPU when it should be below the limit (`cpu`); None =
    request == limit. On GKE Autopilot billing follows requests and Pods may
    burst to their limits, so an idle-mostly sandbox (a shell waiting on the
    model) is cheap to request and still has its full ceiling."""
    memory_request: float | None = None
    """Scheduling request for memory in GiB below the limit (`memory`); None =
    request == limit."""
    """Ephemeral-storage request in GB (None = the platform default)."""
    gpu: str | None = None
    """Unsupported here (kept for TaskResources symmetry): a task asking for a GPU is a loud error."""
    gvisor: bool = False
    """`runtimeClassName: gvisor` for untrusted code (tasks needing ptrace etc. break)."""
    node_selector: dict[str, str] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)
    """Pod annotations. On GKE Autopilot the scale-down of an under-used node
    evicts the sandboxes on it unless they carry
    cluster-autoscaler.kubernetes.io/safe-to-evict: "false" (an evicted
    sandbox is a SandboxError: the episode restarts from scratch)."""
    startup_timeout: float = Field(300.0, gt=0)
    """Seconds for the Pod to become Ready (cold image pulls dominate)."""
    max_lifetime: float = Field(6 * 3600.0, gt=0)
    """Pod `activeDeadlineSeconds`: a sandbox whose owner died without tearing it down
    (env server killed mid-episode) stops consuming the cluster after this long."""
    exec_retries: int = Field(6, ge=0)
    """Retries for SHORT operations (write/read/infra `run`) on transient kubelet errors,
    with jittered exponential backoff capped at 30 s (~1-2 min of budget at the default).
    The harness program itself (`run_program`) is retried only for failures that happen
    before it starts (the exec upgrade was refused) — it is never replayed."""
    exec_concurrency: int = Field(16, ge=0)
    """Process-wide cap on in-flight SHORT kubectl calls that pass the cluster's
    admission webhooks (exec for write/read/`run`, and pod create); 0 = no cap.
    Retries and backoff survive an overloaded webhook; this cap keeps a process
    from overloading it. On GKE Autopilot every pods/exec CONNECT goes through
    the warden admission webhook (10 s timeout), and the cluster serves exec at
    only ~6-7/s (tmax-sandbox, 2026-11-18: 100 concurrent execs took 17 s; 100
    gets took 1.3 s). An exec still queued there after 10 s fails, and its
    retries keep the queue full: a step-start burst of ~200 new pods x ~5 setup
    execs took exec down for an hour (exp2116). Waiting here keeps the server
    queue short while its throughput stays saturated. The cap is per process
    (per event loop): size it for every process that shares the cluster.
    `run_program` (long-lived) and `kubectl wait` are not capped."""


# One semaphore per (event loop, cap): asyncio primitives are bound to their loop.
_ADMISSION_SLOTS: dict[tuple[int, int], asyncio.Semaphore] = {}


def _admission_slot(cap: int) -> contextlib.AbstractAsyncContextManager:
    if cap <= 0:
        return contextlib.nullcontext()
    key = (id(asyncio.get_running_loop()), cap)
    sem = _ADMISSION_SLOTS.get(key)
    if sem is None:
        sem = _ADMISSION_SLOTS[key] = asyncio.Semaphore(cap)
    return sem


class KubernetesRuntimeInfo(KubernetesConfig, BaseRuntimeInfo):
    pass


def _quantity(gb: float | None, suffix: str = "Gi") -> str | None:
    if gb is None:
        return None
    mi = max(int(round(gb * 1024)), 1)
    return f"{mi}Mi" if suffix == "Gi" else f"{gb}"


class KubernetesRuntime(Runtime):
    is_local: ClassVar[bool] = False

    def __init__(self, config: KubernetesConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config
        self.info = KubernetesRuntimeInfo(**config.model_dump())
        # DNS-1123: lowercase alphanumerics and '-'; the base name is `vf-<hex>`.
        self._pod = re.sub(r"[^a-z0-9-]", "-", self.name.lower())[:63].strip("-")
        self._created = False
        self._stopped = False

    # ---- kubectl plumbing --------------------------------------------------------------
    def _base_args(self) -> list[str]:
        args = ["kubectl", "-n", self.config.namespace]
        if self.config.context:
            args += ["--context", self.config.context]
        if self.config.kubeconfig:
            args += ["--kubeconfig", self.config.kubeconfig]
        return args

    async def _kubectl(self, *args: str, stdin: bytes | None = None) -> ProgramResult:
        proc = await asyncio.create_subprocess_exec(
            *self._base_args(), *args,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await proc.communicate(stdin)
        except BaseException:
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
            await run_shielded(proc.communicate())
            raise
        return ProgramResult(
            exit_code=proc.returncode or 0,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
        )

    async def _kubectl_admitted(self, *args: str, stdin: bytes | None = None) -> ProgramResult:
        """`_kubectl` under the process-wide admission cap (see `exec_concurrency`).
        The slot is held only while the call runs, never across retry sleeps."""
        async with _admission_slot(self.config.exec_concurrency):
            return await self._kubectl(*args, stdin=stdin)

    @staticmethod
    def _transient(result: ProgramResult) -> bool:
        err = result.stderr.lower()
        return result.exit_code != 0 and any(p.lower() in err for p in _TRANSIENT) \
            and _EXEC_PREFIX not in result.stderr

    @staticmethod
    def _not_started(result: ProgramResult) -> bool:
        """The exec was refused before the command ran (nothing to replay)."""
        err = result.stderr.lower()
        return result.exit_code != 0 and not result.stdout and _EXEC_PREFIX not in result.stderr \
            and any(p in err for p in _PRE_START)

    def _exec_argv(self, argv: list[str], env: dict[str, str], *, stdin: bool) -> list[str]:
        """`kubectl exec` has neither --env nor --workdir: both ride on a `sh -c` prelude.
        The workdir is created when the image lacks it (docker's --workdir would fail;
        a sandbox whose task never made /app should still exec)."""
        env_words = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        prelude = (
            'd="$1"; shift; mkdir -p "$d" 2>/dev/null; cd "$d" || exit 126; '
            + (f"exec env {env_words} \"$@\"" if env_words else 'exec "$@"')
        )
        return [
            "exec", *(["-i"] if stdin else []), self._pod, "-c", "sandbox", "--",
            "sh", "-c", prelude, "vf-exec", self.config.workdir, *argv,
        ]

    # ---- lifecycle ---------------------------------------------------------------------
    def _manifest(self) -> dict:
        if self.config.gpu:
            raise SandboxError(f"kubernetes runtime: GPU requests are not supported ({self.config.gpu!r})")
        limits: dict[str, str] = {}
        requests: dict[str, str] = {}
        if self.config.cpu is not None:
            limits["cpu"] = str(self.config.cpu)
            requests["cpu"] = str(self.config.cpu_request if self.config.cpu_request is not None else self.config.cpu)
        if (mem := _quantity(self.config.memory)) is not None:
            limits["memory"] = mem
            requests["memory"] = _quantity(self.config.memory_request) if self.config.memory_request is not None else mem
        if (disk := _quantity(self.config.disk)) is not None:
            limits["ephemeral-storage"] = disk
            requests["ephemeral-storage"] = disk
        container = {
            "name": "sandbox",
            "image": self.config.image,
            "command": ["sleep", "infinity"],
            "workingDir": "/",
            "resources": {"requests": requests, "limits": limits},
        }
        spec: dict = {
            "containers": [container],
            "restartPolicy": "Never",
            "terminationGracePeriodSeconds": 1,
            "activeDeadlineSeconds": int(self.config.max_lifetime),
        }
        if self.config.gvisor:
            spec["runtimeClassName"] = "gvisor"
        if self.config.node_selector:
            spec["nodeSelector"] = dict(self.config.node_selector)
        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": self._pod,
                "labels": {"app": "vf-sandbox", **self.config.labels},
                **({"annotations": dict(self.config.annotations)} if self.config.annotations else {}),
            },
            "spec": spec,
        }

    async def start(self) -> None:
        manifest = json.dumps(self._manifest()).encode()
        # The API server is reached over the network (public endpoint through NAT,
        # or the private one): a burst of concurrent creates sees the occasional
        # `dial tcp ...: i/o timeout` / TLS handshake timeout — retry the create
        # itself with backoff (idempotent: a name clash means it did land).
        for attempt in range(self.config.exec_retries + 1):
            created = await self._kubectl_admitted("create", "-f", "-", stdin=manifest)
            if created.exit_code == 0 or "AlreadyExists" in created.stderr:
                break
            if attempt == self.config.exec_retries or not self._transient(created):
                raise SandboxError(f"kubectl create pod failed: {created.stderr.strip()[-500:]}")
            logger.warning("kubernetes: create pod %s hit a transient API error (attempt %d): %s",
                           self._pod, attempt + 1, created.stderr.strip()[-160:])
            await asyncio.sleep(_backoff(attempt, base=3.0))
        self._created = True
        self.info.id = self._pod
        ready = await self._kubectl(
            "wait", f"pod/{self._pod}", "--for=condition=Ready",
            f"--timeout={int(self.config.startup_timeout)}s",
        )
        if ready.exit_code != 0:
            # The reason lives in the Pod's events (ImagePullBackOff, Unschedulable, OOM...).
            describe = await self._kubectl("get", f"pod/{self._pod}", "-o",
                                           "jsonpath={.status.phase} {.status.containerStatuses[*].state}")
            events = await self._kubectl("get", "events", f"--field-selector=involvedObject.name={self._pod}",
                                         "-o", "jsonpath={range .items[*]}{.reason}: {.message}{'\\n'}{end}")
            self.cleanup()
            raise SandboxError(
                f"pod {self._pod} not Ready within {self.config.startup_timeout:.0f}s: "
                f"{describe.stdout.strip()[-300:]} | events: {events.stdout.strip()[-600:]}"
            )
        logger.info("kubernetes: pod %s ready (image=%s, ns=%s)", self._pod, self.config.image, self.config.namespace)

    def cleanup(self) -> None:
        if not self._created or self._stopped:
            return
        self._stopped = True
        subprocess.run(
            [*self._base_args(), "delete", f"pod/{self._pod}", "--grace-period=0", "--wait=false",
             "--ignore-not-found"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False, timeout=120,
        )

    # ---- execution ---------------------------------------------------------------------
    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        """Short, idempotent operations: retried on transient kubelet-proxy errors."""
        env = self.process_env(env)
        argv_k = self._exec_argv(argv, env, stdin=False)
        for attempt in range(self.config.exec_retries + 1):
            result = await self._kubectl_admitted(*argv_k)
            if not self._transient(result) or attempt == self.config.exec_retries:
                return result
            await asyncio.sleep(_backoff(attempt))
        return result  # pragma: no cover

    async def run_program(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        """The rollout itself: never replayed (a replay would fork the trace). Only an exec
        refused before the program started (`_not_started`) is attempted again."""
        argv_k = self._exec_argv(argv, self.process_env(env), stdin=False)
        for attempt in range(self.config.exec_retries + 1):
            result = await self._kubectl(*argv_k)
            if not self._not_started(result) or attempt == self.config.exec_retries:
                return result
            logger.warning("kubernetes: exec of the program in %s refused before start (attempt %d): %s",
                           self._pod, attempt + 1, result.stderr.strip()[-160:])
            await asyncio.sleep(_backoff(attempt))
        return result  # pragma: no cover

    async def _read(self, path: str, max_bytes: int | None = None) -> bytes:
        # `max_bytes` is the base `read()` contract (capped reads pass max+1 and
        # check the overflow themselves): enforce it in the box with head -c.
        # Without this parameter every capped read (the taskset's status file,
        # reward.txt) raised TypeError — swallowed by the taskset's read guard
        # into "agent died", 128/128 episodes of exp2015 try 8.
        argv = ["cat", path] if max_bytes is None else ["head", "-c", str(int(max_bytes)), path]
        for attempt in range(self.config.exec_retries + 1):
            async with _admission_slot(self.config.exec_concurrency):
                proc = await asyncio.create_subprocess_exec(
                    *self._base_args(), *self._exec_argv(argv, {}, stdin=False),
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await proc.communicate()
                except BaseException:
                    if proc.returncode is None:
                        with contextlib.suppress(ProcessLookupError):
                            proc.kill()
                    await run_shielded(proc.communicate())
                    raise
            if proc.returncode == 0:
                return stdout
            failed = ProgramResult(exit_code=proc.returncode or 1, stdout="",
                                   stderr=stderr.decode(errors="replace"))
            if not self._transient(failed) or attempt == self.config.exec_retries:
                raise SandboxError(f"read {path!r}: {failed.stderr.strip()[-500:]}")
            await asyncio.sleep(_backoff(attempt))
        raise AssertionError("unreachable")  # pragma: no cover

    async def write(self, path: str, data: bytes) -> None:
        parent = shlex.quote(str(PurePosixPath(path).parent))
        cmd = ["sh", "-c", f"mkdir -p {parent} && cat > {shlex.quote(path)}"]
        for attempt in range(self.config.exec_retries + 1):
            result = await self._kubectl_admitted(*self._exec_argv(cmd, {}, stdin=True), stdin=data)
            if result.exit_code == 0:
                return
            if not self._transient(result) or attempt == self.config.exec_retries:
                raise SandboxError(f"write {path!r}: {result.stderr.strip()[-500:]}")
            await asyncio.sleep(_backoff(attempt))

    async def alive(self) -> bool:
        phase = await self._kubectl("get", f"pod/{self._pod}", "-o", "jsonpath={.status.phase}")
        return phase.exit_code == 0 and phase.stdout.strip() == "Running"
