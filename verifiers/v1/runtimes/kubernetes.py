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
)
_EXEC_PREFIX = "command terminated with exit code "


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
    exec_retries: int = Field(3, ge=0)
    """Retries for SHORT operations (write/read/infra `run`) on transient kubelet errors.
    The harness program itself (`run_program`) is never replayed."""


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

    @staticmethod
    def _transient(result: ProgramResult) -> bool:
        err = result.stderr.lower()
        return result.exit_code != 0 and any(p.lower() in err for p in _TRANSIENT) \
            and _EXEC_PREFIX not in result.stderr

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
        requests: dict[str, str] = {}
        if self.config.cpu is not None:
            requests["cpu"] = str(self.config.cpu)
        if (mem := _quantity(self.config.memory)) is not None:
            requests["memory"] = mem
        if (disk := _quantity(self.config.disk)) is not None:
            requests["ephemeral-storage"] = disk
        container = {
            "name": "sandbox",
            "image": self.config.image,
            "command": ["sleep", "infinity"],
            "workingDir": "/",
            "resources": {"requests": requests, "limits": requests},
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
            created = await self._kubectl("create", "-f", "-", stdin=manifest)
            if created.exit_code == 0 or "AlreadyExists" in created.stderr:
                break
            if attempt == self.config.exec_retries or not self._transient(created):
                raise SandboxError(f"kubectl create pod failed: {created.stderr.strip()[-500:]}")
            logger.warning("kubernetes: create pod %s hit a transient API error (attempt %d): %s",
                           self._pod, attempt + 1, created.stderr.strip()[-160:])
            await asyncio.sleep(min(3 * 2 ** attempt, 20))
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
            result = await self._kubectl(*argv_k)
            if not self._transient(result) or attempt == self.config.exec_retries:
                return result
            await asyncio.sleep(min(2 ** attempt, 8))
        return result  # pragma: no cover

    async def run_program(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        """The rollout itself: one attempt, never replayed (a replay would fork the trace)."""
        return await self._kubectl(*self._exec_argv(argv, self.process_env(env), stdin=False))

    async def _read(self, path: str) -> bytes:
        proc = await asyncio.create_subprocess_exec(
            *self._base_args(), *self._exec_argv(["cat", path], {}, stdin=False),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise SandboxError(f"read {path!r}: {stderr.decode(errors='replace').strip()[-500:]}")
        return stdout

    async def write(self, path: str, data: bytes) -> None:
        parent = shlex.quote(str(PurePosixPath(path).parent))
        cmd = ["sh", "-c", f"mkdir -p {parent} && cat > {shlex.quote(path)}"]
        for attempt in range(self.config.exec_retries + 1):
            result = await self._kubectl(*self._exec_argv(cmd, {}, stdin=True), stdin=data)
            if result.exit_code == 0:
                return
            if not self._transient(result) or attempt == self.config.exec_retries:
                raise SandboxError(f"write {path!r}: {result.stderr.strip()[-500:]}")
            await asyncio.sleep(min(2 ** attempt, 8))

    async def alive(self) -> bool:
        phase = await self._kubectl("get", f"pod/{self._pod}", "-o", "jsonpath={.status.phase}")
        return phase.exit_code == 0 and phase.stdout.strip() == "Running"
