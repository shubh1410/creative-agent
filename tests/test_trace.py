import asyncio
import inspect
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from creative_agent.models import Issue
from creative_agent.trace import TraceCallback, Tracer, cost, record_issues, render, traced

PRICES = {"test:model": {"input": 3.0, "output": 15.0}}


@pytest.fixture
def tracer(tmp_path):
    return Tracer(tmp_path, "job1")


def lines(tracer: Tracer) -> list[dict]:
    return [json.loads(x) for x in tracer.path.read_text().splitlines()]


async def test_emit_writes_jsonl_and_drops_none(tracer):
    await tracer.emit("node_start", node="plan", size=None, attempt=2)
    [row] = lines(tracer)
    assert row["event"] == "node_start"
    assert row["job_id"] == "job1"
    assert row["node"] == "plan" and row["attempt"] == 2
    assert "size" not in row
    assert "ts" in row


async def test_concurrent_emits_produce_valid_lines(tracer):
    await asyncio.gather(*(tracer.emit("e", i=i) for i in range(50)))
    rows = lines(tracer)
    assert len(rows) == 50
    assert {r["i"] for r in rows} == set(range(50))


async def test_blob_numbering(tracer):
    a = await tracer.blob("prompt", "p")
    b = await tracer.blob("response", "r")
    assert a == "trace/llm_0001_prompt.txt"
    assert b == "trace/llm_0002_response.txt"
    assert (tracer.blobs / "llm_0001_prompt.txt").read_text() == "p"


async def test_usage_sums_and_by_node(tracer):
    await tracer.emit("llm_call", node="plan", input_tokens=100, output_tokens=10, cost_usd=0.5)
    await tracer.emit("llm_call", node="plan", input_tokens=50, output_tokens=5, cost_usd=0.25)
    await tracer.emit("llm_call", node="build", input_tokens=1, output_tokens=2, cost_usd=1.0)
    u = tracer.usage()
    assert u.llm_calls == 3
    assert u.input_tokens == 151 and u.output_tokens == 17
    assert u.cost_usd == pytest.approx(1.75)
    assert u.by_node == {"plan": pytest.approx(0.75), "build": pytest.approx(1.0)}


async def test_usage_unpriced_call_makes_cost_none(tracer):
    await tracer.emit("llm_call", node="plan", input_tokens=1, output_tokens=1, cost_usd=0.5)
    await tracer.emit("llm_call", node="build", input_tokens=1, output_tokens=1)          # unpriced
    await tracer.emit("llm_call", node="plan", input_tokens=1, output_tokens=1, cost_usd=0.5)
    u = tracer.usage()
    assert u.cost_usd is None
    assert u.by_node["build"] is None
    assert u.by_node["plan"] == pytest.approx(1.0)
    assert u.llm_calls == 3


def test_usage_empty(tracer):
    u = tracer.usage()
    assert u.llm_calls == 0 and u.cost_usd == 0.0


def make_runtime(tracer):
    return SimpleNamespace(context=SimpleNamespace(tracer=tracer))


async def test_traced_success(tracer):
    @traced("plan")
    async def node(state, runtime, config=None):
        return {"ok": True}

    out = await node({"size": "300x250", "attempt": 1}, make_runtime(tracer))
    assert out == {"ok": True}
    start, end = lines(tracer)
    assert start["event"] == "node_start" and start["node"] == "plan" and start["size"] == "300x250"
    assert end["event"] == "node_end" and end["duration_ms"] >= 0 and end["attempt"] == 1


async def test_traced_error_reraises(tracer):
    @traced("build")
    async def node(state, runtime, config=None):
        raise ValueError("boom")

    with pytest.raises(ValueError):
        await node({}, make_runtime(tracer))
    start, err = lines(tracer)
    assert start["event"] == "node_start"
    assert err["event"] == "node_error" and "boom" in err["error"]


def test_traced_keeps_signature_visible():
    @traced("plan")
    async def node(state, runtime, config=None):
        return {}

    assert list(inspect.signature(node).parameters) == ["state", "runtime", "config"]


def test_cost_helper():
    assert cost(PRICES, "test:model", 1_000_000, 1_000_000) == 18.0
    assert cost(PRICES, "other", 1, 1) is None


def test_render_replaces_images():
    parts = [{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    out = render(parts)
    assert "hello" in out and "[image]" in out and "AAAA" not in out


async def test_trace_callback_emits_llm_call(tracer):
    cb = TraceCallback(tracer)
    cb.prices = PRICES
    rid = uuid4()
    msgs = [[HumanMessage(content=[{"type": "text", "text": "describe"},
                                   {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}])]]
    await cb.on_chat_model_start({}, msgs, run_id=rid,
                                 metadata={"node": "ingest", "size": "300x250", "attempt": 1,
                                           "model": "test:model", "modality": "vision"})
    result = LLMResult(generations=[[ChatGeneration(message=AIMessage(
        content="x", usage_metadata={"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100}))]])
    await cb.on_llm_end(result, run_id=rid)

    [row] = tracer.events("llm_call")
    assert row["node"] == "ingest" and row["size"] == "300x250" and row["modality"] == "vision"
    assert row["input_tokens"] == 1000 and row["output_tokens"] == 100
    assert row["cost_usd"] == pytest.approx((1000 * 3 + 100 * 15) / 1e6)
    prompt = (tracer.path.parent / row["prompt_file"]).read_text()
    assert "describe" in prompt and "[image]" in prompt and "AAAA" not in prompt
    assert (tracer.path.parent / row["response_file"]).read_text() == "x"
    assert tracer.usage().llm_calls == 1


async def test_trace_callback_unpriced_model_has_no_cost(tracer):
    cb = TraceCallback(tracer)
    cb.prices = {}
    rid = uuid4()
    await cb.on_chat_model_start({}, [[HumanMessage(content="hi")]], run_id=rid, metadata={"node": "plan", "model": "x"})
    await cb.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="y"))]]), run_id=rid)
    [row] = tracer.events("llm_call")
    assert "cost_usd" not in row
    assert tracer.usage().cost_usd is None


async def test_trace_callback_llm_error(tracer):
    cb = TraceCallback(tracer)
    rid = uuid4()
    await cb.on_chat_model_start({}, [[HumanMessage(content="hi")]], run_id=rid, metadata={"node": "plan"})
    await cb.on_llm_error(RuntimeError("rate limit"), run_id=rid)
    [row] = tracer.events("llm_error")
    assert row["node"] == "plan" and "rate limit" in row["error"]


async def test_record_issues_writes_input_fix(tracer):
    ctx = SimpleNamespace(tracer=tracer)
    await record_issues(ctx, [Issue(code="CTA_ADDED", severity="info", message="m", fix="f"),
                              Issue(code="COLOR_SNAPPED", severity="warn", message="m2")])
    rows = tracer.events("input_fix")
    assert [r["code"] for r in rows] == ["CTA_ADDED", "COLOR_SNAPPED"]
    assert rows[0]["fix"] == "f" and "fix" not in rows[1]
    assert tracer.issues()[1] == {"code": "COLOR_SNAPPED", "severity": "warn", "message": "m2", "fix": None}
