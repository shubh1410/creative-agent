"""End-to-end: starts the real agent server and runs both input folders with a real model. Needs an API key in .env."""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from creative_agent.client import AgentClient

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parent.parent
KEY_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY")


def _has_key() -> bool:
    if any(os.environ.get(k) for k in KEY_VARS):
        return True
    env = ROOT / ".env"
    if not env.exists():
        return False
    for line in env.read_text().splitlines():
        k, _, v = line.partition("=")
        if k.strip() in KEY_VARS and v.split("#")[0].strip():
            return True
    return False


@pytest.fixture(scope="module")
def agent_url(tmp_path_factory):
    if not _has_key():
        pytest.skip("no provider API key in the environment or .env")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    data = tmp_path_factory.mktemp("e2e_data")
    env = {**os.environ, "AGENT_URL": url, "DATA_DIR": str(data)}
    env.pop("AGENT_TOKEN", None)
    proc = subprocess.Popen([sys.executable, "-c", "from creative_agent.agent_service.app import main; main()"],
                            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                if httpx.get(f"{url}/healthz", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        else:
            raise RuntimeError("agent server did not start")
        yield url, data
    finally:
        proc.terminate()
        proc.wait(timeout=20)


async def run(url: str, folder: str) -> dict:
    c = AgentClient(base_url=url)
    job = await c.submit(ROOT / "inputs" / folder)
    await c.wait(job["job_id"], timeout_s=1500)
    return await c.result(job["job_id"])


def trace_events(data: Path, job_id: str) -> list[dict]:
    lines = (data / "jobs" / job_id / "trace.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


async def test_maxis_completes(agent_url):
    url, data = agent_url
    res = await run(url, "maxis-merdeka")
    assert res["status"] == "completed", res
    assert len(res["sizes"]) == 5 and all(s["status"] == "approved" and s["attempts"] <= 3 for s in res["sizes"])
    events = trace_events(data, res["job_id"])
    assert {"node_start", "node_end", "llm_call", "qc_check"} <= {e["event"] for e in events}
    job_dir = data / "jobs" / res["job_id"]
    for e in events:
        if e["event"] == "llm_call":
            assert (job_dir / e["prompt_file"]).exists() and (job_dir / e["response_file"]).exists()


async def test_income_stops_at_check_inputs(agent_url):
    url, data = agent_url
    res = await run(url, "income-storyboard")
    assert res["status"] == "failed" and res["error_code"] == "BRAND_MISMATCH", res
    nodes = {e.get("node") for e in trace_events(data, res["job_id"]) if e["event"] == "llm_call"}
    assert "plan" not in nodes and "build" not in nodes


def test_no_api_key_in_data(agent_url):
    _, data = agent_url
    for var in KEY_VARS:
        key = os.environ.get(var)
        if key:
            assert not any(key in p.read_text(errors="ignore") for p in data.rglob("*") if p.is_file())
