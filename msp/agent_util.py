"""Shared agent-call helper for every claude-agent-sdk session in msp / zmip.

The CLI itself retries transient API errors (429 / overloaded / 5xx) with
backoff, but a subscription usage window that is used up ("Claude usage
limit reached …") ends the session with an error result and the CLI does
NOT wait for the reset. A self-driving loop must not stop for that: this
wrapper recognises limit-type failures and re-runs the whole query after a
wait (session state is in-memory on the host — submitted entries persist,
the agent simply starts its investigation again), bounded by a total wait
budget. Any other failure is raised immediately.

    async for message in run_query(prompt, options, label="inspect"):
        ...

Env: AGENT_LIMIT_WAIT_MIN (minutes between retries, default 10),
AGENT_LIMIT_WAIT_MAX_H (total hours to keep waiting, default 12).
"""

import asyncio
import os
import re
import time

LIMIT_PATTERN = re.compile(
    r"usage limit|rate[ _-]?limit|limit will reset|resets at|too many requests|overloaded|"
    r"quota|429|capacity|out of extra usage|spend limit",
    re.IGNORECASE,
)


def is_limit_error(text) -> bool:
    return bool(text) and bool(LIMIT_PATTERN.search(str(text)))


class AgentLimitExhausted(RuntimeError):
    pass


async def run_query(prompt, options, label="agent"):
    """Yield the SDK's messages exactly like query(); if the run ends in a
    limit-type error, wait and start over (bounded)."""
    from claude_agent_sdk import ResultMessage, query

    wait_min = float(os.environ.get("AGENT_LIMIT_WAIT_MIN", "10"))
    max_h = float(os.environ.get("AGENT_LIMIT_WAIT_MAX_H", "12"))
    waited = 0.0
    attempt = 0
    while True:
        attempt += 1
        limit_hit = None
        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage) and getattr(message, "is_error", False) \
                        and is_limit_error(getattr(message, "result", "")):
                    limit_hit = message.result
                    continue  # swallow: the retry below replaces this result
                yield message
        except Exception as e:  # transport-level failure carrying a limit message
            if not is_limit_error(str(e)):
                raise
            limit_hit = str(e)
        if limit_hit is None:
            return
        if waited / 3600 >= max_h:
            raise AgentLimitExhausted(f"[{label}] usage limit still in force after {waited / 3600:.1f} h: {limit_hit}")
        print(f"== [{label}] usage/rate limit (attempt {attempt}): {str(limit_hit)[:160]!r} — "
              f"waiting {wait_min:.0f} min, {max_h - waited / 3600:.1f} h of wait budget left", flush=True)
        t0 = time.time()
        await asyncio.sleep(wait_min * 60)
        waited += time.time() - t0
