"""HARNESS=claude backend for ecarsi.harness.run_agent — claude_agent_sdk,
in-process MCP tools. This is the pre-existing behavior every call site used
to hand-roll; unify it here so both backends share one call shape."""

from __future__ import annotations

import asyncio

from .harness import AgentIncompleteError, AgentRunResult, AgentTimeout, ToolSpec

# Oldest claude-agent-sdk this backend accepts. 0.2.139 bundles Claude Code
# 2.1.233, which rejects the default model on any image Read ("API Error: 400
# Claude Code 2.1.233 does not support this model; version 2.1.251 or newer is
# required") and surfaces it as the useless "returned an error result: success";
# 0.2.152 (CLI 2.1.259) is the first version verified clean here, so that is
# the floor — an older install fails loudly at the first agent call instead of
# five retries deep into a multi-hour job.
MIN_CLAUDE_AGENT_SDK = (0, 2, 152)


def _version_tuple(v: str) -> tuple[int, ...]:
    out = []
    for part in v.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


def check_claude_agent_sdk_version() -> str:
    """Raise RuntimeError if the installed claude-agent-sdk is older than
    MIN_CLAUDE_AGENT_SDK; return the installed version string otherwise."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("claude-agent-sdk")
    except PackageNotFoundError:
        raise RuntimeError("HARNESS=claude needs the claude-agent-sdk package (pip install claude-agent-sdk)") from None
    floor = ".".join(map(str, MIN_CLAUDE_AGENT_SDK))
    if _version_tuple(installed) < MIN_CLAUDE_AGENT_SDK:
        raise RuntimeError(
            f"claude-agent-sdk {installed} is too old for this pipeline (minimum {floor}): its bundled "
            f"Claude Code CLI rejects the default model on image reads. Upgrade: "
            f"pip install -U 'claude-agent-sdk>={floor}'"
        )
    return installed


_BUILTIN = {
    "read": ["Read"], "glob": ["Glob"], "grep": ["Grep"],
    # Claude Code's session task list (a progress checklist the model keeps
    # for itself — one entry per cluster in msp/zmip annotate); the DeepSeek
    # backend serves same-named host-side tools so prompts stay identical.
    "tasks": ["TaskCreate", "TaskUpdate", "TaskList", "TaskGet"],
}


async def _bounded(stream, deadline, label, wall_seconds):
    """Yield the SDK's messages until `deadline` (loop time); past it, close
    the generator (the SDK's own cleanup ends the CLI subprocess) and raise
    AgentTimeout — max_turns caps turns, not a turn that never returns."""
    it = stream.__aiter__()
    try:
        while True:
            timeout = None if deadline is None else max(0.0, deadline - asyncio.get_running_loop().time())
            try:
                message = await asyncio.wait_for(it.__anext__(), timeout=timeout)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError:
                raise AgentTimeout(f"[{label}] agent run exceeded the wall-clock budget of "
                                   f"{wall_seconds / 60:g} min (AGENT_WALL_MIN)") from None
            yield message
    finally:
        await it.aclose()


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
    check_claude_agent_sdk_version()
    from claude_agent_sdk import (
        AssistantMessage, ClaudeAgentOptions, ResultMessage, ToolResultBlock, ToolUseBlock,
        UserMessage, create_sdk_mcp_server, query, tool,
    )

    server_name = "ecarsi_tools"
    submitted_holder: dict = {}

    def _wrap(spec: ToolSpec):
        # capture the submit tool's own result so run_agent can hand back
        # what it validated/stashed without re-deriving it from transcript text
        is_submit = spec.name == submit_tool

        @tool(spec.name, spec.description, spec.input_schema)
        async def _handler(args):
            result = await spec.handler(args)
            if is_submit and not result.get("is_error"):
                submitted_holder["value"] = result.get("_submitted", args)
            return {k: v for k, v in result.items() if k != "_submitted"}

        return _handler

    wrapped = [_wrap(t) for t in tools]
    server = create_sdk_mcp_server(name=server_name, version="1.0.0", tools=wrapped)

    allowed_tools = [name for b in allowed_builtin for name in _BUILTIN[b]] + [
        f"mcp__{server_name}__{t.name}" for t in tools
    ]
    options = ClaudeAgentOptions(
        mcp_servers={server_name: server},
        allowed_tools=allowed_tools,
        # Agent/Task = Claude Code subagents: under bypassPermissions the model
        # can (and did — a 57-cluster NK lineage fanned out to 6 subagents at
        # 3× the usual cost, with the reasoning outside our transcript) spawn
        # them unless explicitly denied; every site here is one agent, one
        # session, by design
        disallowed_tools=["Bash", "Write", "Edit", "MultiEdit", "NotebookEdit", "WebFetch", "WebSearch",
                          "Agent", "Task"],
        permission_mode="bypassPermissions",
        cwd=cwd,
        max_turns=max_turns,
        system_prompt=system_prompt,
        model=model,
        effort=effort,  # type: ignore[arg-type]
        max_buffer_size=max_buffer_size,
    )

    result_text = None
    cost_usd = None
    pending: dict[str, str] = {}  # tool_use_id → tool name, so a failed result can be attributed
    deadline = None if wall_seconds is None else asyncio.get_running_loop().time() + wall_seconds
    async for message in _bounded(query(prompt=prompt, options=options), deadline, label, wall_seconds):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    arg_hint = str(next(iter(block.input.values()), ""))[:80]
                    print(f"== [{label}] agent: {block.name}({arg_hint})", flush=True)
                    pending[block.id] = block.name
        elif isinstance(message, UserMessage) and isinstance(message.content, list):
            # tool RESULTS come back as user-turn blocks; a silently failing
            # builtin (e.g. a Task* tool the CLI doesn't know) would otherwise
            # leave no trace in our logs — surface every error result
            for block in message.content:
                if isinstance(block, ToolResultBlock) and block.is_error:
                    text = block.content if isinstance(block.content, str) else str(block.content)
                    print(f"== [{label}] tool error in {pending.get(block.tool_use_id, '?')}: "
                          f"{text[:200]!r}", flush=True)
        elif isinstance(message, ResultMessage):
            result_text = message.result
            cost_usd = message.total_cost_usd
            if cost_usd:
                print(f"== [{label}] agent cost: ${cost_usd:.2f}", flush=True)

    if "value" not in submitted_holder:
        raise AgentIncompleteError(
            f"[{label}] agent finished without a successful {submit_tool} call. "
            f"Final reply:\n{result_text}"
        )
    return AgentRunResult(submitted=submitted_holder["value"], transcript_text=result_text, cost_usd=cost_usd)
