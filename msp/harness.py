"""Deprecated compatibility shim for the shared agent-harness-bridge package.

Every public object is the same object exported by ``harness_bridge``, so
exception and dataclass identity remain stable across repositories. msp itself
no longer imports this module; import from ``harness_bridge`` directly. The
shim emits a DeprecationWarning and will be removed in msp-sc 0.4.
"""

import warnings

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

warnings.warn(
    "msp.harness is deprecated and will be removed in msp-sc 0.4; import from harness_bridge instead",
    DeprecationWarning,
    stacklevel=2,
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
