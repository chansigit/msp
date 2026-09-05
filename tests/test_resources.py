"""Resource discovery against cgroup fixture trees and patched affinity."""

import os

import msp.resources as resources
from msp.resources import _cgroup_memory_limits, available_cpus, available_memory_bytes, describe

GiB = 1 << 30


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_cgroup_v1_walks_up_the_hierarchy_and_ignores_the_unlimited_root(tmp_path):
    proc = tmp_path / "cgroup"
    write(proc, "12:pids:/slurm/uid_1/job_2\n9:memory:/slurm/uid_1/job_2/step_0\n1:name=systemd:/x\n")
    sysfs = tmp_path / "sys"
    write(sysfs / "memory/slurm/uid_1/job_2/step_0/memory.limit_in_bytes", f"{4 * GiB}\n")
    write(sysfs / "memory/slurm/uid_1/job_2/memory.limit_in_bytes", f"{8 * GiB}\n")
    write(sysfs / "memory/slurm/uid_1/memory.limit_in_bytes", "9223372036854771712\n")  # v1 "no limit"
    write(sysfs / "memory/memory.limit_in_bytes", "9223372036854771712\n")
    assert _cgroup_memory_limits(str(proc), str(sysfs)) == [4 * GiB, 8 * GiB]


def test_cgroup_v2_reads_memory_max_and_skips_max(tmp_path):
    proc = tmp_path / "cgroup"
    write(proc, "0::/docker/abc\n")
    sysfs = tmp_path / "sys"
    write(sysfs / "docker/abc/memory.max", "max\n")
    write(sysfs / "docker/memory.max", f"{2 * GiB}\n")
    write(sysfs / "memory.max", "max\n")
    assert _cgroup_memory_limits(str(proc), str(sysfs)) == [2 * GiB]


def test_cgroup_lookup_never_raises(tmp_path):
    assert _cgroup_memory_limits(str(tmp_path / "missing"), str(tmp_path)) == []
    proc = tmp_path / "cgroup"
    write(proc, "garbage line\n0::/nowhere\n")
    assert _cgroup_memory_limits(str(proc), str(tmp_path / "no-sysfs")) == []


def test_available_memory_prefers_the_tightest_limit(monkeypatch):
    monkeypatch.setattr(resources, "_cgroup_memory_limits", lambda: [8 * GiB, 4 * GiB])
    assert available_memory_bytes() <= 4 * GiB
    monkeypatch.setattr(resources, "_cgroup_memory_limits", lambda: (_ for _ in ()).throw(OSError("boom")))
    assert available_memory_bytes() > 0


def test_available_cpus_uses_affinity_and_the_thread_cap(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2, 3, 4, 5}, raising=False)
    monkeypatch.delenv("MSP_MAX_THREADS", raising=False)
    assert available_cpus() == 6
    monkeypatch.setenv("MSP_MAX_THREADS", "2")
    assert available_cpus() == 2
    monkeypatch.setenv("MSP_MAX_THREADS", "not a number")
    assert available_cpus() == 6
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: (_ for _ in ()).throw(AttributeError()), raising=False)
    assert available_cpus() >= 1


def test_describe_mentions_cpus_and_memory():
    text = describe()
    assert "cpu(s)" in text and "GiB memory" in text
