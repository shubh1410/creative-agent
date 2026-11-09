"""Job tracing: a JSONL event log per job, saved prompt/response blobs, token/cost accounting and a LangChain
callback that records every LLM call.

Layout inside a job folder::

    trace.jsonl          one JSON event per line (node_start, node_end, llm_call, input_fix, error, ...)
    trace/llm_NNNN_*.txt full prompt and response text of each LLM call, referenced from llm_call events
"""
import asyncio
import functools
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from langchain_core.callbacks import AsyncCallbackHandler

from .config import settings
from .llm import message_text
from .models import Usage


class Tracer:
    """Append-only event writer for one job. Safe to share between concurrent graph branches."""

    def __init__(self, job_dir: Path, job_id: str):
        self.job_id = job_id
        self.path = job_dir / "trace.jsonl"
        self.blobs = job_dir / "trace"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()           # five branches append at once
        self._n = 0                           # counter making blob file names unique and ordered

    async def emit(self, event: str, **fields: Any) -> None:
        """Append one event line. ``None``-valued fields are dropped; non-JSON values are stringified."""
        line = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "job_id": self.job_id,
                "event": event, **{k: v for k, v in fields.items() if v is not None}}
        async with self._lock:
            with self.path.open("a") as f:
                f.write(json.dumps(line, default=str) + "\n")

    async def blob(self, kind: str, text: str) -> str:
        """Store a large text (prompt / response) as its own file and return its path relative to the job folder."""
        async with self._lock:                # only the counter needs the lock; the write itself is independent
            self._n += 1
            name = f"llm_{self._n:04d}_{kind}.txt"
        (self.blobs / name).write_text(text)
        return f"trace/{name}"

    def events(self, event: str | None = None) -> list[dict]:
        """Read events back from disk, optionally only those with the given ``event`` name."""
        if not self.path.exists():
            return []
        rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        return [r for r in rows if event is None or r["event"] == event]

    def usage(self) -> Usage:
        """Aggregate all ``llm_call`` events into token counts and cost (total and per node).

        If any call used a model without a known price, the affected costs become ``None`` rather than
        silently under-reporting.
        """
        u = Usage()
        unpriced = False
        for c in self.events("llm_call"):
            u.llm_calls += 1
            u.input_tokens += c.get("input_tokens", 0)
            u.output_tokens += c.get("output_tokens", 0)
            cost, node = c.get("cost_usd"), c.get("node", "unknown")
            if cost is None:
                unpriced = True
                u.by_node[node] = None
            elif u.by_node.get(node, 0) is not None:      # skip nodes already marked unpriced
                u.by_node[node] = round(u.by_node.get(node, 0) + cost, 6)
            u.cost_usd = None if unpriced else round((u.cost_usd or 0) + (cost or 0), 6)
        return u

    def issues(self) -> list[dict]:
        """Input problems recorded during the job (``input_fix`` events), as plain dicts for ``Issue``."""
        return [{k: e.get(k) for k in ("code", "severity", "message", "fix")} for e in self.events("input_fix")]


async def record_issues(ctx, issues) -> None:
    """Log each :class:`~creative_agent.models.Issue` as an ``input_fix`` event so it appears in the result."""
    for i in issues:
        await ctx.tracer.emit("input_fix", **i.model_dump())


def traced(name: str):
    """Write node_start / node_end (or node_error) around a node. LangGraph reads the wrapped signature.

    Args:
        name: Node name recorded in the events.

    The wrapped function must have the signature ``(state, runtime, config=None)``. ``size`` and ``attempt``
    from the state (if present) are attached to every event so per-size branches can be told apart.
    """
    def deco(fn):
        @functools.wraps(fn)                  # preserves the signature LangGraph inspects
        async def wrapper(state, runtime, config=None):
            tracer = runtime.context.tracer
            tags = {k: state.get(k) for k in ("size", "attempt")}
            await tracer.emit("node_start", node=name, **tags)
            t0 = time.perf_counter()
            try:
                out = await fn(state, runtime, config)
            except Exception as e:
                await tracer.emit("node_error", node=name, error=repr(e), **tags)
                raise                          # tracing must not swallow failures
            await tracer.emit("node_end", node=name, duration_ms=round((time.perf_counter() - t0) * 1000), **tags)
            return out
        return wrapper
    return deco


def load_prices() -> dict:
    """Read the pricing file: ``{model: {input: usd_per_million, output: usd_per_million}}``. Empty if absent."""
    return (yaml.safe_load(settings.pricing_file.read_text()) or {}) if settings.pricing_file.exists() else {}


def cost(prices: dict, model: str, tokens_in: int, tokens_out: int) -> float | None:
    """USD cost of one call, or ``None`` if ``model`` has no price entry. Prices are per million tokens."""
    p = prices.get(model)
    return None if p is None else round((tokens_in * p["input"] + tokens_out * p["output"]) / 1e6, 6)


def render(content) -> str:
    """Flatten message content to text for logging; image blocks (huge base64) are replaced by ``[image]``."""
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") if b.get("type") == "text" else "[image]" for b in content if isinstance(b, dict))


class TraceCallback(AsyncCallbackHandler):
    """One llm_call event per model call, with the full prompt and response saved as files (images as [image])."""

    def __init__(self, tracer: Tracer):
        # ``started`` maps run_id -> (start time, metadata, prompt blob path) between start and end callbacks.
        self.tracer, self.prices, self.started = tracer, load_prices(), {}

    async def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kw):
        """Remember when the call began and save its prompt (``metadata`` comes from ``llm._config``)."""
        prompt = "\n\n".join(f"[{m.type}]\n{render(m.content)}" for m in messages[0])
        self.started[run_id] = (time.perf_counter(), metadata or {}, await self.tracer.blob("prompt", prompt))

    async def on_llm_end(self, response, *, run_id, **kw):
        """Emit the ``llm_call`` event with token usage, cost, latency and links to prompt/response files."""
        t0, meta, prompt_file = self.started.pop(run_id, (time.perf_counter(), {}, None))
        gen = response.generations[0][0] if response.generations and response.generations[0] else None
        msg = getattr(gen, "message", None)
        usage = getattr(msg, "usage_metadata", None) or {}
        tin, tout = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        # Structured-output calls return tool calls rather than text, so log those when there is no text.
        text = message_text(msg) or json.dumps(getattr(msg, "tool_calls", None), default=str)
        model = meta.get("model", "")
        await self.tracer.emit("llm_call", node=meta.get("node"), size=meta.get("size"), attempt=meta.get("attempt"),
                               modality=meta.get("modality", "text"), model=model, input_tokens=tin,
                               output_tokens=tout, cost_usd=cost(self.prices, model, tin, tout),
                               latency_ms=round((time.perf_counter() - t0) * 1000), prompt_file=prompt_file,
                               response_file=await self.tracer.blob("response", text))

    async def on_llm_error(self, error, *, run_id, **kw):
        """Emit an ``llm_error`` event and discard the pending start record."""
        _, meta, _ = self.started.pop(run_id, (0, {}, None))
        await self.tracer.emit("llm_error", node=meta.get("node"), size=meta.get("size"), error=repr(error))
