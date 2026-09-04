"""Legacy Claude SDK helper retained for compatibility with external callers.

MSP's active agents use msp.harness and the shared agent-harness-bridge.
New callers should use that interface; this module is no longer part of the
MSP/ZMIP execution path. Its retry behavior is preserved for legacy imports.

The CLI itself retries transient API errors (429 / overloaded / 5xx) with
backoff, but a subscription usage window that is used up ("Claude usage
limit reached …") ends the session with an error result and the CLI does
NOT wait for the reset. A self-driving loop must not stop for that: this
wrapper recognises limit-type failures and re-runs the whole query after a
wait (session state is in-memory on the host — submitted entries persist,
the agent simply starts its investigation again), bounded by a total wait
budget. Any other failure is raised immediately.

Separately: many `claude` CLI subprocesses starting at once (e.g. a batch of
Slurm jobs all launching agent sessions around the same time) can blow the
SDK's own local control handshake ("Control request timeout: initialize") or
die with a transport-level hiccup (broken pipe / connection reset) — nothing
to do with the account's usage limit, just local contention. Those get a
short, immediate retry (a handful of attempts, seconds of backoff) instead
of the long limit-wait.

    async for message in run_query(prompt, options, label="inspect"):
        ...

Env: AGENT_LIMIT_WAIT_MIN (minutes between limit retries, default 10),
AGENT_LIMIT_WAIT_MAX_H (total hours to keep waiting on limits, default 12).
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

TRANSIENT_PATTERN = re.compile(
    r"control request timeout|broken pipe|connection reset|econnreset|epipe|"
    r"process exited unexpectedly|failed to start|connection closed",
    re.IGNORECASE,
)

MAX_TRANSIENT_ATTEMPTS = 5
TRANSIENT_BACKOFF_SECONDS = 20  # linear: 20s, 40s, 60s, 80s


def is_limit_error(text) -> bool:
    return bool(text) and bool(LIMIT_PATTERN.search(str(text)))


def is_transient_error(text) -> bool:
    return bool(text) and bool(TRANSIENT_PATTERN.search(str(text)))


class AgentLimitExhausted(RuntimeError):
    pass


async def run_query(prompt, options, label="agent"):
    """Yield the SDK's messages exactly like query(); if the run ends in a
    limit-type error, wait and start over (bounded). If it ends in a local
    transport hiccup (concurrent-startup contention), retry quickly (bounded,
    separate budget)."""
    from claude_agent_sdk import ResultMessage, query

    wait_min = float(os.environ.get("AGENT_LIMIT_WAIT_MIN", "10"))
    max_h = float(os.environ.get("AGENT_LIMIT_WAIT_MAX_H", "12"))
    waited = 0.0
    attempt = 0
    transient_attempts = 0
    while True:
        attempt += 1
        limit_hit = None
        transient_hit = None
        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage) and getattr(message, "is_error", False) \
                        and is_limit_error(getattr(message, "result", "")):
                    limit_hit = message.result
                    continue  # swallow: the retry below replaces this result
                yield message
        except Exception as e:  # transport-level failure carrying a limit or transient message
            msg = str(e)
            if is_limit_error(msg):
                limit_hit = msg
            elif is_transient_error(msg):
                transient_hit = msg
            else:
                raise
        if transient_hit is not None:
            transient_attempts += 1
            if transient_attempts >= MAX_TRANSIENT_ATTEMPTS:
                raise RuntimeError(
                    f"[{label}] transient agent-startup failure persisted after "
                    f"{transient_attempts} attempts: {transient_hit}"
                ) from None
            wait = TRANSIENT_BACKOFF_SECONDS * transient_attempts
            print(f"== [{label}] transient agent-startup failure (attempt {transient_attempts}/"
                  f"{MAX_TRANSIENT_ATTEMPTS}): {str(transient_hit)[:160]!r} — retrying in {wait}s",
                  flush=True)
            await asyncio.sleep(wait)
            continue
        if limit_hit is None:
            return
        if waited / 3600 >= max_h:
            raise AgentLimitExhausted(f"[{label}] usage limit still in force after {waited / 3600:.1f} h: {limit_hit}")
        print(f"== [{label}] usage/rate limit (attempt {attempt}): {str(limit_hit)[:160]!r} — "
              f"waiting {wait_min:.0f} min, {max_h - waited / 3600:.1f} h of wait budget left", flush=True)
        t0 = time.time()
        await asyncio.sleep(wait_min * 60)
        waited += time.time() - t0
