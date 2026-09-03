"""HARNESS=claude backend for ecarsi.harness.run_agent — claude_agent_sdk,
in-process MCP tools. This is the pre-existing behavior every call site used
to hand-roll; unify it here so both backends share one call shape."""

from __future__ import annotations

from .harness import AgentIncompleteError, AgentRunResult, ToolSpec

_BUILTIN = {"read": "Read", "glob": "Glob", "grep": "Grep"}


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
    from claude_agent_sdk import (
        AssistantMessage, ClaudeAgentOptions, ResultMessage, ToolUseBlock,
        create_sdk_mcp_server, query, tool,
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

    allowed_tools = [_BUILTIN[b] for b in allowed_builtin] + [
        f"mcp__{server_name}__{t.name}" for t in tools
    ]
    options = ClaudeAgentOptions(
        mcp_servers={server_name: server},
        allowed_tools=allowed_tools,
        disallowed_tools=["Bash", "Write", "Edit", "MultiEdit", "NotebookEdit", "WebFetch", "WebSearch"],
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
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    arg_hint = str(next(iter(block.input.values()), ""))[:80]
                    print(f"== [{label}] agent: {block.name}({arg_hint})", flush=True)
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
