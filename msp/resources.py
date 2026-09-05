"""What this process may actually use — inside a Slurm allocation the node's
totals are a lie (os.cpu_count() reports the node's 32 cores while the job
was given 8; psutil sees 250 GB while --mem was 48G). Nothing here needs the
caller to pass anything in: CPU comes from the scheduler-set affinity mask,
memory from the enforced cgroup limit (walking up the hierarchy and taking
the tightest bound — Slurm's step cgroup can be tighter than the job's).
Outside a cgroup/affinity-restricted environment both fall back to the
machine totals. Used for parallelism decisions (integrate's DEG pool, zmip's
per-lineage concurrency)."""

from __future__ import annotations

import contextlib
import os

_UNLIMITED = 1 << 60  # cgroup v1 reports ~9.2e18 for "no limit"


def _env_int(name: str) -> int | None:
    try:
        v = int(os.environ.get(name, ""))
        return v if v > 0 else None
    except ValueError:
        return None


def available_cpus() -> int:
    """CPUs this process may use: the scheduler affinity mask (what Slurm /
    docker --cpuset actually grant), capped by MSP_MAX_THREADS when a parent
    that runs several of us side by side (zmip's lineage pool) sets it. Never
    raises: no affinity API (macOS), no /proc, or a container that hides it
    all fall back to os.cpu_count(), then to 1."""
    try:
        n = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except Exception:
        n = 0
    if n <= 0:
        try:
            n = os.cpu_count() or 1
        except Exception:
            n = 1
    cap = _env_int("MSP_MAX_THREADS")
    return max(1, min(n, cap) if cap else n)


def _cgroup_memory_limits(proc_cgroup: str = "/proc/self/cgroup", sysfs: str = "/sys/fs/cgroup") -> list[int]:
    """Every memory limit that applies to this process, from its own cgroup up
    to the root (cgroup v1 `memory.limit_in_bytes`, v2 `memory.max`). Works
    for Slurm (path /slurm/uid_N/job_N/...), docker (/docker/<id> or v2
    "0::/"), plain VMs (no limit → empty list); a container that namespaces
    /sys/fs/cgroup so the listed path doesn't exist just yields nothing.
    ``proc_cgroup`` and ``sysfs`` exist so tests can point at fixture trees."""
    limits: list[int] = []
    try:
        with open(proc_cgroup) as fh:
            lines = fh.read().splitlines()
    except Exception:
        return limits
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        controllers, path = parts[1], parts[2]
        if controllers == "":  # cgroup v2 unified
            roots, fname = [sysfs], "memory.max"
        elif "memory" in controllers.split(","):
            roots, fname = [os.path.join(sysfs, "memory")], "memory.limit_in_bytes"
        else:
            continue
        for root in roots:
            p = path.rstrip("/")
            while True:
                try:
                    with open(os.path.join(root + p, fname)) as fh:
                        raw = fh.read().strip()
                    if raw != "max":
                        v = int(float(raw))
                        if 0 < v < _UNLIMITED:
                            limits.append(v)
                except (OSError, ValueError):
                    pass
                if not p:
                    break
                p = p.rsplit("/", 1)[0]
    return limits


def available_memory_bytes() -> int:
    """Tightest enforced memory limit, else physical RAM, else (nothing
    readable at all — never an exception) a conservative 8 GiB."""
    try:
        limits = _cgroup_memory_limits()
    except Exception:
        limits = []
    phys = 0
    with contextlib.suppress(Exception):
        phys = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    candidates = [v for v in [*limits, phys] if v > 0]
    return min(candidates) if candidates else 8 << 30


def current_rss_bytes() -> int:
    """This process's resident set (0 when /proc is unavailable)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def describe() -> str:
    return (
        f"{available_cpus()} cpu(s), {available_memory_bytes() / 2**30:.1f} GiB memory available to this "
        f"process (rss {current_rss_bytes() / 2**30:.1f} GiB)"
    )


if __name__ == "__main__":
    print(describe())
