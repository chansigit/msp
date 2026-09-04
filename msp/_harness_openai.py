"""HARNESS=openai backend using OpenAI Agents SDK with Volcengine Ark.

This is the direct-Python alternative to the dsh backend.  The Agents SDK
owns the model/tool loop, while our existing ``ToolSpec`` handlers remain the
only authority for validation and final submission.  No shell, editor, MCP
server, Node subprocess, or remote OpenAI tracing is enabled.

Environment:
  ARK_API_KEY          Volcengine Ark credential.
  DOUBAO_BASE_URL      OpenAI-compatible API root (default Beijing /api/v3).
  OPENAI_AGENTS_API    responses (default) or chat_completions.  The latter is
                       a text-only compatibility path; image tool results need
                       Responses.
  OPENAI_AGENTS_MAX_NUDGES
                       continuation attempts after a normal model reply that
                       omitted the required submit tool (default 2).
  OPENAI_AGENTS_MAX_CONTEXT_RESETS
                       fresh Responses sessions allowed after Ark rejects an
                       overlong image+text context (default 2). Host-side task
                       and submission state is retained.
  OPENAI_AGENTS_SERVER_STATE
                       Responses-only response chaining (default 1). Disable
                       with 0 to send the complete local history every turn.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from ._harness_host_tools import readonly_tools, task_tools
from .harness import AgentIncompleteError, AgentRunResult, AgentTimeout, ToolSpec

DOUBAO_BASE_URL_DEFAULT = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_API_MODE = "responses"
DEFAULT_MAX_NUDGES = 2
DEFAULT_MAX_CONTEXT_RESETS = 2

_JSON_TYPES: dict[type, dict[str, Any]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    # Every current list-valued ToolSpec is a list of gene symbols.
    list: {"type": "array", "items": {"type": "string"}},
    dict: {"type": "object"},
}


def _params_schema(spec: ToolSpec) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for name, python_type in spec.input_schema.items():
        if python_type not in _JSON_TYPES:
            raise TypeError(f"unsupported ToolSpec input type for {spec.name}.{name}: {python_type!r}")
        properties[name] = dict(_JSON_TYPES[python_type])
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _tool(spec: ToolSpec, submitted_holder: dict, is_submit: bool, label: str, api_mode: str):
    from agents import FunctionTool, ToolOutputImage, ToolOutputText

    async def invoke(_context, arguments_json: str):
        arguments = json.loads(arguments_json)
        arg_hint = str(next(iter(arguments.values()), ""))[:80]
        print(f"== [{label}] agent: {spec.name}({arg_hint})", flush=True)
        try:
            result = await spec.handler(arguments)
        except (KeyError, TypeError, ValueError) as exc:
            # Hand-written ToolSpec handlers sometimes validate a JSON string
            # nested inside otherwise schema-valid tool arguments.  Bad model
            # data there must be returned to the model for correction, not
            # escape FunctionTool and abort the entire agent run.
            message = f"{type(exc).__name__}: {exc}"
            print(f"== [{label}] tool exception in {spec.name}: {message!r}", flush=True)
            return ToolOutputText(
                text=f"ERROR: {spec.name} rejected the input ({message}). Fix it and call the tool again."
            )
        is_error = bool(result.get("is_error", False))
        if is_error:
            text = " ".join(str(block.get("text", "")) for block in result.get("content", []))
            print(f"== [{label}] tool error in {spec.name}: {text[:200]!r}", flush=True)
        if is_submit and not is_error:
            submitted_holder["value"] = result.get("_submitted", arguments)

        outputs = []
        for block in result.get("content", []):
            kind = block.get("type")
            if kind == "text":
                prefix = "ERROR: " if is_error else ""
                outputs.append(ToolOutputText(text=prefix + str(block.get("text", ""))))
            elif kind == "image":
                if api_mode != "responses":
                    outputs.append(ToolOutputText(
                        text="ERROR: image tool output requires OPENAI_AGENTS_API=responses"
                    ))
                    continue
                media_type = str(block.get("mimeType") or "application/octet-stream")
                outputs.append(ToolOutputImage(
                    image_url=f"data:{media_type};base64,{block.get('data', '')}", detail="high"
                ))
            else:
                raise ValueError(f"unsupported ToolSpec content block type: {kind!r}")
        return outputs or ToolOutputText(text="ERROR: tool returned no content")

    return FunctionTool(
        name=spec.name,
        description=spec.description,
        params_json_schema=_params_schema(spec),
        on_invoke_tool=invoke,
        strict_json_schema=True,
    )


def _model(model: str, api_mode: str):
    from agents import OpenAIChatCompletionsModel, OpenAIResponsesModel
    from openai import AsyncOpenAI

    key = os.environ.get("ARK_API_KEY")
    if not key:
        raise RuntimeError("HARNESS=openai needs ARK_API_KEY")
    client = AsyncOpenAI(
        api_key=key,
        base_url=os.environ.get("DOUBAO_BASE_URL", DOUBAO_BASE_URL_DEFAULT),
        max_retries=0,
    )
    if api_mode == "responses":
        return OpenAIResponsesModel(model=model, openai_client=client)
    if api_mode == "chat_completions":
        return OpenAIChatCompletionsModel(model=model, openai_client=client)
    raise ValueError(
        f"invalid OPENAI_AGENTS_API={api_mode!r} (expected 'responses' or 'chat_completions')"
    )


def _is_context_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return exc.__class__.__name__ == "BadRequestError" and (
        "exceed max message tokens" in text
        or "maximum context length" in text
        or "context length exceeded" in text
    )


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
    from agents import Agent, ItemHelpers, MaxTurnsExceeded, ModelSettings, RunConfig, Runner
    from agents.agent import ToolsToFinalOutputResult

    if not model:
        raise ValueError("HARNESS=openai needs a model id (MODEL env or caller model)")
    api_mode = os.environ.get("OPENAI_AGENTS_API", DEFAULT_API_MODE).strip().lower()
    max_nudges = int(os.environ.get("OPENAI_AGENTS_MAX_NUDGES", str(DEFAULT_MAX_NUDGES)))
    if max_nudges < 0:
        raise ValueError("OPENAI_AGENTS_MAX_NUDGES must be >= 0")
    max_context_resets = int(os.environ.get(
        "OPENAI_AGENTS_MAX_CONTEXT_RESETS", str(DEFAULT_MAX_CONTEXT_RESETS)
    ))
    if max_context_resets < 0:
        raise ValueError("OPENAI_AGENTS_MAX_CONTEXT_RESETS must be >= 0")
    server_state = (
        api_mode == "responses"
        and os.environ.get("OPENAI_AGENTS_SERVER_STATE", "1").strip().lower()
        not in {"0", "false", "no", "off"}
    )

    submitted_holder: dict = {}
    served = list(tools) + readonly_tools(cwd, allowed_builtin)
    if "tasks" in allowed_builtin:
        served += task_tools()
    if submit_tool not in {spec.name for spec in served}:
        raise ValueError(f"submit tool {submit_tool!r} is not present in the tool table")
    wrapped = [
        _tool(spec, submitted_holder, spec.name == submit_tool, label, api_mode)
        for spec in served
    ]

    runtime_context = (
        f"Harness runtime: the working directory is {cwd!r}. Relative paths in the task resolve from "
        "that directory. Use only the provided tools; do not guess alternate workspace roots. "
        f"The run is complete only after a successful {submit_tool} call."
    )
    instructions = (f"{system_prompt.rstrip()}\n\n{runtime_context}"
                    if system_prompt else runtime_context)
    reasoning = {"effort": effort} if effort else None
    settings = ModelSettings(
        parallel_tool_calls=True,
        reasoning=reasoning,
        # Response chaining prevents every earlier image/tool result from
        # being uploaded again on each turn. Ark has been verified to support
        # this Responses API contract. Setting SERVER_STATE=0 restores a
        # fully local history at the cost of rapidly growing input tokens.
        store=server_state,
    )

    def finish_after_valid_submit(_context, _tool_results):
        return ToolsToFinalOutputResult(
            is_final_output="value" in submitted_holder,
            final_output=submitted_holder.get("value"),
        )

    agent = Agent(
        name=label,
        instructions=instructions,
        tools=wrapped,
        model=_model(model, api_mode),
        model_settings=settings,
        tool_use_behavior=finish_after_valid_submit,
    )
    run_config = RunConfig(
        tracing_disabled=True,
        workflow_name=f"eca-rsi {label}",
    )
    _ = max_buffer_size  # OpenAI Agents SDK does not pipe image bytes through a CLI buffer.

    async def run_loop() -> AgentRunResult:
        run_input: str | list = prompt
        previous_response_id: str | None = None
        turns_used = 0
        nudges = 0
        context_resets = 0
        transcript_parts: list[str] = []
        usage_totals = {"requests": 0, "input": 0, "output": 0, "reasoning": 0}

        while True:
            remaining = max_turns - turns_used
            if remaining <= 0:
                raise AgentIncompleteError(
                    f"[{label}] HARNESS=openai exhausted max_turns={max_turns} without a successful "
                    f"{submit_tool} call"
                )
            try:
                runner_kwargs: dict[str, Any] = {}
                if server_state:
                    runner_kwargs["auto_previous_response_id"] = True
                    if previous_response_id is not None:
                        runner_kwargs["previous_response_id"] = previous_response_id
                result = await Runner.run(
                    agent, run_input, max_turns=remaining, run_config=run_config,
                    **runner_kwargs,
                )
            except MaxTurnsExceeded:
                raise AgentIncompleteError(
                    f"[{label}] HARNESS=openai exceeded max_turns={max_turns} without a successful "
                    f"{submit_tool} call"
                ) from None
            except Exception as exc:
                if not _is_context_limit_error(exc) or context_resets >= max_context_resets:
                    raise
                context_resets += 1
                previous_response_id = None
                run_input = (
                    "The provider rejected the previous response because the accumulated image and text "
                    "context was too large. Continue in this fresh session. Host-side tool state, completed "
                    "Tasks, and all valid partial submissions are still present. Use TaskList or the domain "
                    f"tools to recover only the evidence you still need, then finish with {submit_tool}."
                )
                print(
                    f"== [{label}] context limit reached — continuing with a fresh model session "
                    f"({context_resets}/{max_context_resets}); host tool state retained",
                    flush=True,
                )
                continue

            usage = result.context_wrapper.usage
            used = max(1, int(usage.requests))
            turns_used += used
            usage_totals["requests"] += int(usage.requests)
            usage_totals["input"] += int(usage.input_tokens)
            usage_totals["output"] += int(usage.output_tokens)
            usage_totals["reasoning"] += int(
                usage.output_tokens_details.reasoning_tokens or 0
            )
            text = ItemHelpers.text_message_outputs(result.new_items).strip()
            if text:
                transcript_parts.append(text)

            if "value" in submitted_holder:
                print(
                    f"== [{label}] HARNESS=openai api={api_mode} "
                    f"server_state={'on' if server_state else 'off'} model={model} run: "
                    f"{usage_totals['requests']} model request(s), {usage_totals['input']} input / "
                    f"{usage_totals['output']} output tokens "
                    f"({usage_totals['reasoning']} reasoning)",
                    flush=True,
                )
                return AgentRunResult(
                    submitted=submitted_holder["value"],
                    transcript_text="\n\n".join(transcript_parts) or None,
                    cost_usd=None,
                )

            if nudges >= max_nudges or turns_used >= max_turns:
                final_text = text or str(result.final_output or "")
                raise AgentIncompleteError(
                    f"[{label}] agent finished without a successful {submit_tool} call after "
                    f"{turns_used} model request(s) and {nudges} nudge(s). Final reply:\n{final_text}"
                )
            nudges += 1
            print(
                f"== [{label}] turn ended without {submit_tool} after {turns_used} model request(s) — "
                f"nudging the session to continue ({nudges}/{max_nudges})",
                flush=True,
            )
            nudge_input = {
                "role": "user",
                "content": (
                    f"Your previous turn ended without calling {submit_tool}. Continue exactly where you "
                    f"left off and finish by calling {submit_tool}."
                ),
            }
            if server_state and result.last_response_id is not None:
                previous_response_id = result.last_response_id
                run_input = [nudge_input]
            else:
                run_input = result.to_input_list()
                run_input.append(nudge_input)

    if wall_seconds is None:
        return await run_loop()
    try:
        return await asyncio.wait_for(run_loop(), timeout=wall_seconds)
    except asyncio.TimeoutError:
        raise AgentTimeout(
            f"[{label}] agent run exceeded the wall-clock budget of {wall_seconds / 60:g} min "
            "(AGENT_WALL_MIN)"
        ) from None
