"""Compatibility shim for the shared agent-harness-bridge package.

Application code may keep importing this module; every public object is the
same object exported by harness_bridge, so exception and dataclass identity
remain stable across repositories.

Deprecated: new code should import from ``harness_bridge`` directly. This
module stays until every in-tree caller (inspect, annotate, __main__) and the
external ones (zmip, eca-rsi) have moved; see TODO.md.
"""

from harness_bridge import (
    AgentConfig,
    AgentIncompleteError,
    AgentLimitExhausted,
    AgentRunResult,
    AgentTimeout,
    HarnessCapabilities,
    ToolSpec,
    backend_capabilities,
    backend_name,
    default_model,
    resolve_agent_config,
    retry_transient,
    run_agent,
    wall_seconds,
)
from harness_bridge.harness import (
    DEFAULT_BACKEND,
    DEFAULT_WALL_MINUTES,
    LIMIT_PATTERN,
    MAX_TIMEOUT_ATTEMPTS,
    MAX_TRANSIENT_ATTEMPTS,
    TRANSIENT_BACKOFF_SECONDS,
    TRANSIENT_PATTERN,
    ToolHandler,
)

__all__ = [
    "AgentConfig",
    "AgentIncompleteError",
    "AgentLimitExhausted",
    "AgentRunResult",
    "AgentTimeout",
    "DEFAULT_BACKEND",
    "DEFAULT_WALL_MINUTES",
    "HarnessCapabilities",
    "LIMIT_PATTERN",
    "MAX_TIMEOUT_ATTEMPTS",
    "MAX_TRANSIENT_ATTEMPTS",
    "TRANSIENT_BACKOFF_SECONDS",
    "TRANSIENT_PATTERN",
    "ToolHandler",
    "ToolSpec",
    "backend_capabilities",
    "backend_name",
    "default_model",
    "resolve_agent_config",
    "retry_transient",
    "run_agent",
    "wall_seconds",
]
