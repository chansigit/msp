"""Pluggable agent execution backend for every model-facing call in this
package (ecarsi/osp/msp/zmip each carry an identical copy — no shared
cross-repo import, per this project's independent-repos convention; zmip is
the one exception, already depending on msp for other helpers, so it reuses
msp's copy instead of a fourth one).

Every call site builds a small, self-contained tool table (the "submit tool"
pattern: one designated tool ends the run and its handler is the only place
the actual answer is produced/validated — the model never needs filesystem
write access to get its answer out) and hands it to `run_agent()`. Which SDK
actually drives the model is an env-var choice, not a call-site choice:

    HARNESS=claude     (default) claude_agent_sdk, in-process MCP tools
    HARNESS=deepseek    DeepSeek Harness (dsh) via its Python SDK, tools
                         bridged over an in-process streamable-http MCP
                         server (dsh's mcp-client only attaches to an
                         external server; the tool handlers are Python
                         closures that can't cross a real subprocess
                         boundary, so the "external" server is HTTP-local
                         rather than a spawned stdio child)

The tool `handler` return shape (`{"content": [{"type": "text", ...}],
"is_error": bool}`) is already the real MCP `CallToolResult` wire shape —
Claude Agent SDK's in-process server is itself an MCP server — so the same
handler bodies serve both backends unchanged.

Every run_agent() call is wrapped in `retry_transient` (ported from msp's
agent_util.run_query, generalized to both backends — DeepSeek Harness is
alpha-stage software and has shown the same kind of transient subprocess
death Claude's CLI does).
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

ToolHandler = Callable[[dict], Awaitable[dict]]
T = TypeVar("T")

# Ported from msp/agent_util.py's run_query, generalized to wrap either
# backend's whole run rather than just Claude's message-stream generator.
# Concurrent Slurm job starts (a batch of jobs all launching agent sessions
# around the same time) can blow a local control handshake or kill the
# subprocess transport — nothing to do with the account's usage limit, just
# local contention — and get a short, immediate retry. A genuine usage/rate
# limit gets a long wait instead, bounded by a total wait budget, since a
# self-driving loop must not stop for that.
LIMIT_PATTERN = re.compile(
    r"usage limit|rate[ _-]?limit|limit will reset|resets at|too many requests|overloaded|"
    r"quota|429|capacity|out of extra usage|spend limit",
    re.IGNORECASE,
)
TRANSIENT_PATTERN = re.compile(
    r"control request timeout|broken pipe|connection reset|econnreset|epipe|"
    r"process exited unexpectedly|failed to start|connection closed|stdout closed|"
    r"transportclosed|initialize timed out|timed out waiting|returned an error result",
    re.IGNORECASE,
)
MAX_TRANSIENT_ATTEMPTS = 5
TRANSIENT_BACKOFF_SECONDS = 20  # linear: 20s, 40s, 60s, 80s


class AgentLimitExhausted(RuntimeError):
    pass


async def retry_transient(coro_fn: Callable[[], Awaitable[T]], label: str) -> T:
    """Run coro_fn(); on a transient-looking failure, retry a bounded number
    of times with linear backoff; on a usage/rate-limit-looking failure,
    wait and retry, bounded by a total wait budget (env AGENT_LIMIT_WAIT_MIN
    minutes between tries, default 10; AGENT_LIMIT_WAIT_MAX_H total hours,
    default 12). Any other failure raises immediately."""
    wait_min = float(os.environ.get("AGENT_LIMIT_WAIT_MIN", "10"))
    max_h = float(os.environ.get("AGENT_LIMIT_WAIT_MAX_H", "12"))
    waited = 0.0
    limit_attempt = 0
    transient_attempts = 0
    while True:
        try:
            return await coro_fn()
        except Exception as e:
            msg = str(e)
            if TRANSIENT_PATTERN.search(msg):
                transient_attempts += 1
                if transient_attempts >= MAX_TRANSIENT_ATTEMPTS:
                    raise RuntimeError(
                        f"[{label}] transient agent-startup failure persisted after "
                        f"{transient_attempts} attempts: {msg}"
                    ) from None
                wait = TRANSIENT_BACKOFF_SECONDS * transient_attempts
                print(f"== [{label}] transient agent-startup failure (attempt {transient_attempts}/"
                      f"{MAX_TRANSIENT_ATTEMPTS}): {msg[:160]!r} — retrying in {wait}s", flush=True)
                await asyncio.sleep(wait)
                continue
            if LIMIT_PATTERN.search(msg):
                limit_attempt += 1
                if waited / 3600 >= max_h:
                    raise AgentLimitExhausted(
                        f"[{label}] usage limit still in force after {waited / 3600:.1f} h: {msg}"
                    ) from None
                print(f"== [{label}] usage/rate limit (attempt {limit_attempt}): {msg[:160]!r} — "
                      f"waiting {wait_min:.0f} min, {max_h - waited / 3600:.1f} h of wait budget left",
                      flush=True)
                t0 = time.time()
                await asyncio.sleep(wait_min * 60)
                waited += time.time() - t0
                continue
            raise


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    # {param_name: python_type} — the same flat shape claude_agent_sdk's
    # @tool() takes as its third argument; translated to real JSON Schema
    # for the DeepSeek backend's MCP server.
    input_schema: dict[str, type]
    handler: ToolHandler


@dataclass
class AgentRunResult:
    submitted: dict | None  # whatever the submit tool's handler captured; None if it never fired
    transcript_text: str | None  # best-effort final assistant text, for *_notes.md-style logging
    cost_usd: float | None  # best-effort; None where the backend doesn't report it


class AgentIncompleteError(RuntimeError):
    """The run ended without the submit tool ever firing."""


def backend_name() -> str:
    return os.environ.get("HARNESS", "claude")


_DEFAULT_MODEL = {
    "claude": "claude-sonnet-5",
    # HARNESS=deepseek's default provider is Doubao via dsh's pi-ai adapter
    # (see _harness_deepseek); DSH_PROVIDER=deepseek-official switches to a
    # real DeepSeek model, in which case override MODEL too.
    "deepseek": "doubao-seed-2-1-turbo-260628",
}


def default_model() -> str:
    """A model id a standalone call (no caller-supplied model=...) can fall
    back to — MODEL env, else the HARNESS-appropriate default. Callers that
    already receive a resolved model string (e.g. from ecarsi) never need
    this; it exists so this package's own CLI works un-orchestrated."""
    backend = backend_name()
    return os.environ.get("MODEL", _DEFAULT_MODEL.get(backend, _DEFAULT_MODEL["claude"]))


async def run_agent(
    *,
    tools: list[ToolSpec],
    submit_tool: str,
    prompt: str,
    system_prompt: str | None = None,
    cwd: str,
    model: str | None = None,
    effort: str | None = None,
    max_turns: int = 30,
    allowed_builtin: tuple[str, ...] = ("read", "glob", "grep"),
    label: str = "agent",
    max_buffer_size: int | None = None,
) -> AgentRunResult:
    """Run one agent turn to completion; raise AgentIncompleteError if
    `submit_tool` never fired. `allowed_builtin` is the read-only filesystem
    exploration surface ("read", "glob", "grep") plus "tasks" — a session
    task list the model keeps as its own progress checklist (Claude Code's
    TaskCreate/TaskUpdate/TaskList/TaskGet; the DeepSeek backend serves
    same-named in-memory tools so prompts stay identical). The model never
    gets write access under either backend."""
    backend = backend_name()
    if backend == "claude":
        from ._harness_claude import run_agent as _run
    elif backend == "deepseek":
        from ._harness_deepseek import run_agent as _run
    else:
        raise ValueError(f"unknown HARNESS backend {backend!r} (expected 'claude' or 'deepseek')")

    async def _attempt() -> AgentRunResult:
        return await _run(
            tools=tools, submit_tool=submit_tool, prompt=prompt, system_prompt=system_prompt,
            cwd=cwd, model=model, effort=effort, max_turns=max_turns,
            allowed_builtin=allowed_builtin, label=label, max_buffer_size=max_buffer_size,
        )

    return await retry_transient(_attempt, label)
