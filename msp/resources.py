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

import os

_UNLIMITED = 1 << 60  # cgroup v1 reports ~9.2e18 for "no limit"


def available_cpus() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _cgroup_memory_limits() -> list[int]:
    """Every memory limit that applies to this process, from its own cgroup up
    to the root (cgroup v1 `memory.limit_in_bytes`, v2 `memory.max`)."""
    limits: list[int] = []
    try:
        lines = open("/proc/self/cgroup").read().splitlines()
    except OSError:
        return limits
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        controllers, path = parts[1], parts[2]
        if controllers == "":  # cgroup v2 unified
            roots, fname = ["/sys/fs/cgroup"], "memory.max"
        elif "memory" in controllers.split(","):
            roots, fname = ["/sys/fs/cgroup/memory"], "memory.limit_in_bytes"
        else:
            continue
        for root in roots:
            p = path.rstrip("/")
            while True:
                try:
                    raw = open(os.path.join(root + p, fname)).read().strip()
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
    """Tightest enforced memory limit, else physical RAM."""
    limits = _cgroup_memory_limits()
    if limits:
        return min(limits)
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 8 << 30


def describe() -> str:
    return f"{available_cpus()} cpu(s), {available_memory_bytes() / 2**30:.1f} GiB memory available to this process"


if __name__ == "__main__":
    print(describe())
