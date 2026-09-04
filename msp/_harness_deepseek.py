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
  DSH_BIN            path to the built `dsh` CLI entrypoint (apps/cli/lib/bin.js
                      from a source build — see the eca-rsi harness-deepseek
                      branch notes; the PyPI-published native runtime wheel
                      is glibc-2.28-only and unusable on Sherlock as of
                      2026-09). Required.
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

import yaml

for _name in ("mcp", "mcp.server", "mcp.server.streamable_http", "mcp.server.streamable_http_manager",
              "mcp.server.lowlevel.server"):
    logging.getLogger(_name).setLevel(logging.WARNING)  # per-request INFO lines are pure noise here
# (uvicorn's own loggers are configured at server start, so they are quieted
# through uvicorn.Config(log_level=...) in run_agent, not here)

from .harness import AgentIncompleteError, AgentRunResult, ToolSpec

_BUILTIN_DISABLE_IDS = ("persistent-bash", "terminal-bash", "persistent-pwsh",
                         "terminal-pwsh", "str-replace-editor")

DOUBAO_BASE_URL_DEFAULT = "https://ark.cn-beijing.volces.com/api/v3"


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
        "config": {"serverName": "ecarsi", "transport": "streamable-http", "url": mcp_url},
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


def _run_sync(*, dsh_bin: str, cwd: str, dsh_home: str, provider: str, model: str | None,
              effort: str | None, system_prompt: str | None, prompt: str, session_id: str,
              patch_path: str):
    from deepseek_harness import DeepSeekHarness

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
        return harness.run(prompt, session_id=session_id)


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
) -> AgentRunResult:
    from mcp.server.fastmcp import FastMCP

    dsh_bin = os.environ.get("DSH_BIN")
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

    # run uvicorn ourselves instead of FastMCP.run_streamable_http_async() so
    # teardown is a graceful should_exit (no lifespan CancelledError traceback
    # on every call) and its log level is ours to set
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(mcp_server.streamable_http_app(), host="127.0.0.1", port=port,
                                           log_level="warning", lifespan="on"))
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
            # dsh has no per-run turn cap or pipe-buffer knob at this API level
            # (unlike Claude's max_turns/max_buffer_size); it runs to natural
            # completion (idle) or its own request_timeout_seconds.
            _ = (max_turns, max_buffer_size)
            result = await asyncio.to_thread(
                _run_sync, dsh_bin=dsh_bin, cwd=cwd, dsh_home=dsh_home, provider=provider,
                model=model, effort=effort, system_prompt=system_prompt, prompt=prompt,
                session_id=f"{label}-{os.getpid()}", patch_path=patch_path,
            )
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
