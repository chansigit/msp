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

logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)  # silence its own noisy
# lifespan-cancellation traceback when we tear the per-call MCP server down

from .harness import AgentIncompleteError, AgentRunResult, ToolSpec

_BUILTIN_DISABLE_IDS = ("persistent-bash", "terminal-bash", "persistent-pwsh",
                         "terminal-pwsh", "str-replace-editor")

DOUBAO_BASE_URL_DEFAULT = "https://ark.cn-beijing.volces.com/api/v3"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _tool_fn(spec: ToolSpec, submitted_holder: dict, is_submit: bool):
    import mcp.types as types

    params = [inspect.Parameter(k, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=t)
              for k, t in spec.input_schema.items()]

    async def fn(**kwargs):
        result = await spec.handler(kwargs)
        if is_submit and not result.get("is_error"):
            submitted_holder["value"] = result.get("_submitted", kwargs)
        content: list[types.ContentBlock] = [types.TextContent(**c) for c in result["content"]]
        return types.CallToolResult(content=content, isError=bool(result.get("is_error", False)))

    fn.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    fn.__name__ = spec.name
    return fn


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
    for spec in tools:
        mcp_server.add_tool(_tool_fn(spec, submitted_holder, spec.name == submit_tool),
                            name=spec.name, description=spec.description)
    mcp_url = f"http://127.0.0.1:{port}{mcp_server.settings.streamable_http_path}"

    server_task = asyncio.create_task(mcp_server.run_streamable_http_async())
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
        server_task.cancel()
        try:
            await server_task
        except (asyncio.CancelledError, Exception):
            pass

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
