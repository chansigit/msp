"""HARNESS=deepseek backend for ecarsi.harness.run_agent — DeepSeek Harness
(dsh) via its Python SDK.

dsh has no native JSON-schema-constrained structured output (unlike Claude
Agent SDK's `output_format`), so the submit-tool pattern is load-bearing
here, not cosmetic: the actual answer comes from our own tool handler firing
(exactly like the Claude backend), never from parsing dsh's free-text
`final_response`.

Tool bridging: dsh's real MCP-client plugin (`@deepseek-ai/dsh-mcp-client`)
only attaches to an *external* server (stdio subprocess or HTTP), so the
`ToolSpec` handlers — Python closures that can't cross a process boundary —
are served from an in-process `FastMCP` streamable-http server on an
ephemeral localhost port for the lifetime of one run_agent() call; dsh
connects to it as any other MCP client would.

Read-only exploration is enforced by dsh's OS-level `sandbox/mode: read-only`
(bwrap/Landlock on Linux) on the profile's persistent bash + editor tools —
stronger than, and the intended equivalent of, the Claude backend's
Read/Glob/Grep-only `allowed_tools` allowlist. When `allowed_builtin` is
empty (a call site that today passes Claude's `allowed_tools=[]`), bash and
the editor are disabled outright via patch so only the submit tool is
reachable.

Model provider: dsh's built-in `deepseek-official` route only ever reaches
DeepSeek's own models. Everything else — including this codebase's default,
Doubao via Volcengine's Ark platform — goes through dsh's generic
`@deepseek-ai/dsh-llm-pi-ai` adapter as a hand-declared OpenAI-compatible
route (Ark's chat-completions endpoint is OpenAI-wire-compatible, function
calling included; verified against Volcengine's own docs and a third-party
harness's working config, not against a live call — Ark's tool-schema
validator is known to drop a few JSON Schema keywords like minLength/
maxLength/minItems/maxItems, which none of this codebase's tool schemas use).

Env:
  AGENT_WALL_MIN      wall-clock budget per run in minutes (default 180, 0 =
                      unlimited; see harness.wall_seconds) — a watchdog closes
                      the dsh runtime when it is hit. max_turns is enforced
                      here too, by counting assistant/message events.
  DSH_TRACE_EVENTS    =1 prints every dsh session event type as it streams.
  DSH_KEEP_SESSIONS   =1 keeps every run's dsh session.jsonl in cwd (a run that
                      ends without a submit keeps it regardless).
  DSH_BIN            path to the built `dsh` CLI entrypoint (apps/cli/lib/bin.js
                      from a source build — see the eca-rsi harness-deepseek
                      branch notes; the PyPI-published native runtime wheel
                      is glibc-2.28-only and unusable on Sherlock as of
                      2026-09). Default: $SCRATCH/tools/deepseek-harness-src/
                      apps/cli/lib/bin.js when that file exists.
  DSH_HOME_ROOT       parent dir for the disposable per-call dsh home
                      (default: $SCRATCH or /tmp).
  DSH_PROVIDER        pi-ai route name to select (default 'doubao'); pass
                      'deepseek-official' to go back to DeepSeek's own models.
  DOUBAO_BASE_URL     Ark OpenAI-compatible base URL (default the public
                      Beijing endpoint).
  ARK_API_KEY         Volcengine Ark credential — read by dsh itself (via
                      apiKeyEnv, resolved in the spawned dsh process's own
                      environment), never touched by this module directly.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import socket
import tempfile
import threading

import yaml

for _name in ("mcp", "mcp.server", "mcp.server.streamable_http", "mcp.server.streamable_http_manager",
              "mcp.server.lowlevel.server"):
    logging.getLogger(_name).setLevel(logging.WARNING)  # per-request INFO lines are pure noise here
# (uvicorn's own loggers are configured at server start, so they are quieted
# through uvicorn.Config(log_level=...) in run_agent, not here)

from .harness import AgentIncompleteError, AgentRunResult, AgentTimeout, ToolSpec

# sse-starlette (the SSE layer under mcp's streamable-http transport) keeps a
# PROCESS-GLOBAL `AppStatus.should_exit`; its per-loop watcher copies uvicorn's
# `server.should_exit` into it, so the graceful teardown of the FIRST
# run_agent() server in a process flips it for good and every SSE response of
# every later server in that process ends the moment it starts ("ASGI callable
# returned without completing response") — dsh's mcp-client then cannot attach
# and the model runs on with no tools (msp inspect → annotate in one process,
# 2026-09-03). We close streams ourselves (dsh exits, server task cancelled),
# so the automatic drain is disabled and the flag reset before each server.
try:
    from sse_starlette.sse import AppStatus as _SseAppStatus

    _SseAppStatus.disable_automatic_graceful_drain()
except Exception:  # pragma: no cover - older/newer sse-starlette without the knob
    _SseAppStatus = None

_BUILTIN_DISABLE_IDS = ("persistent-bash", "terminal-bash", "persistent-pwsh",
                         "terminal-pwsh", "str-replace-editor")

DOUBAO_BASE_URL_DEFAULT = "https://ark.cn-beijing.volces.com/api/v3"


def _default_dsh_bin() -> str | None:
    """The source build this cluster uses when DSH_BIN is unset (HARNESS=
    deepseek is the default, so a bare `eca-rsi run` must find dsh)."""
    root = os.environ.get("SCRATCH")
    if root:
        cand = os.path.join(root, "tools", "deepseek-harness-src", "apps", "cli", "lib", "bin.js")
        if os.path.isfile(cand):
            return cand
    return None


def _keep_session_log(dsh_home: str, cwd: str, label: str) -> None:
    """Copy the run's dsh session transcript (every model step and tool call)
    out of the disposable dsh home into cwd, for post-mortems of runs that
    ended without a submit — the transcript is the only record of what the
    model actually did in dsh (our trace only sees our own tools)."""
    import glob
    import shutil

    for src in glob.glob(os.path.join(dsh_home, "sessions", "*", "*", "session.jsonl")):
        dst = os.path.join(cwd, f"dsh_session_{label.replace(' ', '_').replace('/', '_')}.jsonl")
        try:
            shutil.copy(src, dst)
            print(f"== [{label}] dsh session transcript kept at {dst}", flush=True)
        except OSError as e:
            print(f"== [{label}] could not keep dsh session transcript: {e}", flush=True)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _tool_fn(spec: ToolSpec, submitted_holder: dict, is_submit: bool, label: str):
    import mcp.types as types

    params = [inspect.Parameter(k, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=t)
              for k, t in spec.input_schema.items()]

    async def fn(**kwargs):
        arg_hint = str(next(iter(kwargs.values()), ""))[:80]  # same trace line as the Claude backend
        print(f"== [{label}] agent: {spec.name}({arg_hint})", flush=True)
        result = await spec.handler(kwargs)
        if result.get("is_error"):
            text = " ".join(str(c.get("text", "")) for c in result.get("content", []))
            print(f"== [{label}] tool error in {spec.name}: {text[:200]!r}", flush=True)
        if is_submit and not result.get("is_error"):
            submitted_holder["value"] = result.get("_submitted", kwargs)
        content: list[types.ContentBlock] = [types.TextContent(**c) for c in result["content"]]
        return types.CallToolResult(content=content, isError=bool(result.get("is_error", False)))

    fn.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    fn.__name__ = spec.name
    return fn


def _task_tools() -> list[ToolSpec]:
    """Host-side stand-in for Claude Code's session task list (allowed_builtin
    "tasks"): same tool names and the same create/update/list/get shape, kept
    in memory for one run_agent() call. Purely the model's own progress
    checklist — coverage is enforced by each call site's finalize/submit
    validation, never by this list."""
    tasks: dict[str, dict] = {}
    statuses = ("pending", "in_progress", "completed")

    def _text(s):
        return {"content": [{"type": "text", "text": s}]}

    def _err(s):
        return {"content": [{"type": "text", "text": s}], "is_error": True}

    def _render(t):
        return f"#{t['id']} [{t['status']}] {t['subject']}" + (f" — {t['description']}" if t["description"] else "")

    async def task_create(args):
        tid = str(len(tasks) + 1)
        tasks[tid] = {"id": tid, "subject": str(args.get("subject") or "").strip(),
                      "description": str(args.get("description") or "").strip(), "status": "pending"}
        if not tasks[tid]["subject"]:
            del tasks[tid]
            return _err("subject is required")
        return _text(f"created task #{tid}: {tasks[tid]['subject']}")

    async def task_update(args):
        tid = str(args.get("taskId") or "").lstrip("#")
        if tid not in tasks:
            return _err(f"no task #{tid}; existing: {sorted(tasks, key=int)}")
        status = str(args.get("status") or "").strip()
        if status not in statuses:
            return _err(f"status must be one of {statuses}")
        tasks[tid]["status"] = status
        return _text(f"updated {_render(tasks[tid])}")

    async def task_list(args):
        if not tasks:
            return _text("no tasks yet")
        done = sum(t["status"] == "completed" for t in tasks.values())
        return _text("\n".join(_render(tasks[k]) for k in sorted(tasks, key=int))
                     + f"\n({done}/{len(tasks)} completed)")

    async def task_get(args):
        tid = str(args.get("taskId") or "").lstrip("#")
        if tid not in tasks:
            return _err(f"no task #{tid}; existing: {sorted(tasks, key=int)}")
        return _text(_render(tasks[tid]))

    return [
        ToolSpec("TaskCreate", "Create a task on your session task list (a progress checklist). "
                 "Returns its id.", {"subject": str, "description": str}, task_create),
        ToolSpec("TaskUpdate", "Set a task's status: pending | in_progress | completed.",
                 {"taskId": str, "status": str}, task_update),
        ToolSpec("TaskList", "List every task on your session task list with its status.", {}, task_list),
        ToolSpec("TaskGet", "Show one task by id.", {"taskId": str}, task_get),
    ]


def _render_patch(mcp_url: str, allowed_builtin: tuple[str, ...], provider: str, model: str | None) -> str:
    insert: list[dict] = [{
        "id": "ecarsi-tools",
        "name": "@deepseek-ai/dsh-mcp-client",
        # failOnStartupError: dsh's default is to log the failed MCP attach,
        # keep reconnecting in the background (10 attempts, ~2.5 min) and let
        # the model run on with only its builtin tools meanwhile — which is
        # how a run with no submit tool wandered the filesystem for hours
        # (2026-09-03). Failing the session at load turns that into an error
        # the SDK surfaces immediately, which retry_transient retries.
        "config": {"serverName": "ecarsi", "transport": "streamable-http", "url": mcp_url,
                   "failOnStartupError": True},
    }]
    rows: list[dict] = [{"id": "sandbox-policy", "config": {"mode": "read-only"}}]
    if provider != "deepseek-official":
        # hand-declared OpenAI-compatible route (pi-ai's catalog doesn't ship
        # third-party providers like Doubao) — api/baseURL/models are all
        # required for a route the catalog doesn't describe.
        if not model:
            raise ValueError(f"HARNESS=deepseek with DSH_PROVIDER={provider!r} needs a model id (MODEL env)")
        base_url = os.environ.get("DOUBAO_BASE_URL", DOUBAO_BASE_URL_DEFAULT)
        insert.append({
            "id": "ecarsi-llm-provider",
            "name": "@deepseek-ai/dsh-llm-pi-ai",
            "config": {"providers": {provider: {
                "apiKeyEnv": "ARK_API_KEY",
                "api": "openai-completions",
                "baseURL": base_url,
                "models": [{"id": model}],
            }}},
        })
    if not allowed_builtin:
        rows += [{"id": row_id, "disabled": True} for row_id in _BUILTIN_DISABLE_IDS]
    return yaml.safe_dump([{"insert": insert}, *rows], sort_keys=False)


class _TurnsExceeded(RuntimeError):
    pass


MCP_LIST_GRACE_SECONDS = 90.0  # dsh must have asked our server for tools/list by then


def _run_sync(*, dsh_bin: str, cwd: str, dsh_home: str, provider: str, model: str | None,
              effort: str | None, system_prompt: str | None, prompt: str, session_id: str,
              patch_path: str, label: str, max_turns: int, wall_seconds: float | None,
              listed: threading.Event, http_trace: dict):
    """One dsh run, bounded two ways dsh itself doesn't offer: a turn cap
    (assistant/message events, the model's turns, counted as they stream —
    the same quantity Claude's max_turns bounds) and a wall-clock budget
    (a watchdog thread closes the runtime, which fails the run's
    subscription from inside). Either bound kills the subprocess."""
    from deepseek_harness import DeepSeekHarness

    trace = os.environ.get("DSH_TRACE_EVENTS", "") not in ("", "0")
    turns = 0
    timed_out = threading.Event()
    no_tools = threading.Event()

    def on_notification(n):
        nonlocal turns
        if n.method != "session.event":
            return
        event = n.payload.get("event") or {}
        kind = event.get("type")
        if trace:
            print(f"== [{label}] dsh event: {kind}", flush=True)
        if kind == "assistant/message":
            turns += 1
            if turns > max_turns:
                raise _TurnsExceeded(f"[{label}] HARNESS=deepseek run exceeded max_turns={max_turns}")

    with DeepSeekHarness(
        provider=provider,
        model=model,
        reasoning_effort=effort,
        cwd=cwd,
        dsh_home=dsh_home,
        profile="sdk-minimal",
        patches=(patch_path,),
        dsh_bin=dsh_bin,
        env={"DSH_SYSTEM_PROMPT": system_prompt} if system_prompt else {},
        # default 30s has flaked on the composed profile (mcp-client + pi-ai
        # insert on top of sdk-minimal has more to boot than the bare
        # profile); retry_transient covers a genuine timeout too, but a
        # longer budget means fewer retries in the common case.
        initialize_timeout_seconds=90.0,
    ) as harness:
        timers = []
        if wall_seconds is not None:
            def _kill():
                timed_out.set()
                print(f"== [{label}] wall-clock budget of {wall_seconds / 60:g} min hit — closing dsh runtime",
                      flush=True)
                harness.close()
            timers.append(threading.Timer(wall_seconds, _kill))

        def _check_listed():
            # a dsh whose mcp-client failed to attach runs on with only its
            # builtin editor/bash: the model then wanders the filesystem for
            # hours and can never submit (seen 2026-09-03: 71 str_replace_editor
            # views of an unrelated directory). Kill it early, retry as transient.
            if not listed.is_set():
                no_tools.set()
                print(f"== [{label}] dsh never requested tools/list from our MCP server within "
                      f"{MCP_LIST_GRACE_SECONDS:g} s — closing dsh runtime", flush=True)
                harness.close()
        timers.append(threading.Timer(MCP_LIST_GRACE_SECONDS, _check_listed))
        for t in timers:
            t.daemon = True
            t.start()

        def _stderr_tail(n=25):
            lines = list(getattr(harness.client, "_stderr_lines", []))[-n:]
            return "\n".join(lines)

        try:
            result = harness.run(prompt, session_id=session_id, on_notification=on_notification)
        except Exception as e:
            if timed_out.is_set():
                raise AgentTimeout(f"[{label}] agent run exceeded the wall-clock budget of "
                                   f"{wall_seconds / 60:g} min (AGENT_WALL_MIN)") from None
            if no_tools.is_set():
                print(f"== [{label}] http requests seen by our MCP server: {dict(http_trace)}\n"
                      f"== [{label}] dsh stderr tail:\n{_stderr_tail()}", flush=True)
                raise RuntimeError(f"[{label}] mcp tools never listed by dsh (mcp-client failed to attach)") from None
            if isinstance(e, _TurnsExceeded):
                raise AgentIncompleteError(f"{e} without a successful submit call") from None
            raise
        finally:
            for t in timers:
                t.cancel()
        print(f"== [{label}] dsh run: {turns} model turn(s); http requests seen by our MCP server "
              f"(method path: [requests, responses started]): {dict(http_trace)}", flush=True)
        if not listed.is_set():
            print(f"== [{label}] dsh stderr tail:\n{_stderr_tail()}", flush=True)
        return result


async def run_agent(
    *,
    tools: list[ToolSpec],
    submit_tool: str,
    prompt: str,
    system_prompt: str | None,
    cwd: str,
    model: str | None,
    effort: str | None,
    max_turns: int,
    allowed_builtin: tuple[str, ...],
    label: str,
    max_buffer_size: int | None,
    wall_seconds: float | None = None,
) -> AgentRunResult:
    from mcp.server.fastmcp import FastMCP

    dsh_bin = os.environ.get("DSH_BIN") or _default_dsh_bin()
    if not dsh_bin:
        raise RuntimeError(
            "HARNESS=deepseek needs DSH_BIN pointing at a built dsh CLI entrypoint "
            "(apps/cli/lib/bin.js from a `pnpm install && pnpm run build` checkout of "
            "deepseek-ai/deepseek-harness — the published native runtime wheel needs "
            "glibc >= 2.28)"
        )

    provider = os.environ.get("DSH_PROVIDER", "doubao")
    submitted_holder: dict = {}
    port = _free_port()
    mcp_server = FastMCP(name=f"ecarsi-{label}", host="127.0.0.1", port=port, stateless_http=True)
    served = list(tools) + (_task_tools() if "tasks" in allowed_builtin else [])
    for spec in served:
        mcp_server.add_tool(_tool_fn(spec, submitted_holder, spec.name == submit_tool, label),
                            name=spec.name, description=spec.description)
    mcp_url = f"http://127.0.0.1:{port}{mcp_server.settings.streamable_http_path}"

    listed = threading.Event()  # set once dsh's mcp-client asks for tools/list
    _orig_list_tools = mcp_server._tool_manager.list_tools

    def _list_tools_hook(*a, **kw):
        listed.set()
        return _orig_list_tools(*a, **kw)
    mcp_server._tool_manager.list_tools = _list_tools_hook  # type: ignore[method-assign]

    http_trace: dict = {}  # "METHOD path" -> [requests, responses completed]
    inner_app = mcp_server.streamable_http_app()

    async def app(scope, receive, send):
        if scope["type"] != "http":
            return await inner_app(scope, receive, send)
        key = f"{scope['method']} {scope['path']}"
        rec = http_trace.setdefault(key, [0, 0])
        rec[0] += 1

        async def _send(msg):
            if msg["type"] == "http.response.start":
                rec[1] += 1
            await send(msg)
        await inner_app(scope, receive, _send)

    # run uvicorn ourselves instead of FastMCP.run_streamable_http_async() so
    # teardown is a graceful should_exit (no lifespan CancelledError traceback
    # on every call) and its log level is ours to set
    import uvicorn

    if _SseAppStatus is not None:
        _SseAppStatus.should_exit = False
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on"))
    server_task = asyncio.create_task(server.serve())
    try:
        for _ in range(50):  # wait for uvicorn to actually bind before dsh tries to connect
            await asyncio.sleep(0.05)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    break

        root = os.environ.get("DSH_HOME_ROOT") or os.environ.get("SCRATCH") or tempfile.gettempdir()
        with tempfile.TemporaryDirectory(prefix="dsh-home-", dir=root) as dsh_home:
            patch_text = _render_patch(mcp_url, allowed_builtin, provider, model)
            with tempfile.NamedTemporaryFile("w", suffix=".patch.yml", dir=dsh_home, delete=False) as pf:
                pf.write(patch_text)
                patch_path = pf.name
            print(f"== [{label}] HARNESS=deepseek provider={provider} model={model} "
                  f"dsh_home={dsh_home} mcp={mcp_url}", flush=True)
            # dsh has no pipe-buffer knob (max_buffer_size is Claude-only);
            # max_turns and the wall-clock budget are enforced in _run_sync
            _ = max_buffer_size
            keep = os.environ.get("DSH_KEEP_SESSIONS", "") not in ("", "0")
            try:
                result = await asyncio.to_thread(
                    _run_sync, dsh_bin=dsh_bin, cwd=cwd, dsh_home=dsh_home, provider=provider,
                    model=model, effort=effort, system_prompt=system_prompt, prompt=prompt,
                    session_id=f"{label}-{os.getpid()}", patch_path=patch_path, label=label,
                    max_turns=max_turns, wall_seconds=wall_seconds, listed=listed, http_trace=http_trace,
                )
                keep = keep or "value" not in submitted_holder
            except BaseException:
                keep = True
                raise
            finally:
                if keep:
                    _keep_session_log(dsh_home, cwd, label)
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(server_task, timeout=10)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            server_task.cancel()

    if result.finish_reason == "error":
        errs = [e for e in result.events if e.get("type") == "turn/end"
                and e.get("data", {}).get("reason", {}).get("kind") == "error"]
        detail = errs[-1]["data"]["reason"]["error"] if errs else result.final_response
        raise RuntimeError(f"[{label}] HARNESS=deepseek run ended in error: {detail}")

    if "value" not in submitted_holder:
        raise AgentIncompleteError(
            f"[{label}] agent finished without a successful {submit_tool} call. "
            f"Final reply:\n{result.final_response}"
        )
    return AgentRunResult(submitted=submitted_holder["value"], transcript_text=result.final_response,
                          cost_usd=None)
