import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport

from creative_agent.agent_service import jobs as jobs_mod
from creative_agent.agent_service.app import app
from creative_agent.agent_service.jobs import JobManager
from creative_agent.config import settings
from creative_agent.errors import JobInputError
from creative_agent.models import JobResult, JobStatus

ROOT = Path(__file__).resolve().parent.parent
MAXIS = ROOT / "inputs" / "maxis-merdeka"


class FakeGraph:
    """Stands in for JOB_GRAPH: waits for `gate` (or briefly), writes result.json and reports completion."""

    def __init__(self, gate: asyncio.Event | None = None):
        self.gate = gate

    async def ainvoke(self, state, context, config=None):
        if self.gate is not None:
            await self.gate.wait()
        else:
            await asyncio.sleep(0.05)
        (context.job_dir / "result.json").write_text(
            JobResult(job_id=context.job_id, status="completed").model_dump_json())
        return {"job_status": "completed"}


class FailingGraph:
    async def ainvoke(self, state, context, config=None):
        await context.tracer.emit("input_fix", code="CLICK_URL_REDIRECT", severity="warn",
                                  message="redirected", fix="used final url")
        raise JobInputError("BRAND_MISMATCH", "the master shows 'Income', not 'maxis'")


@pytest.fixture
def jobs_env(data_dir, monkeypatch):
    monkeypatch.setattr(jobs_mod, "make_llm", lambda model: None)
    monkeypatch.setattr(settings, "agent_token", None)
    manager = JobManager(browser=None)
    monkeypatch.setattr(app.state, "jobs", manager, raising=False)
    return manager


@pytest.fixture
async def client(jobs_env):
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


def maxis_request(**over) -> str:
    return json.dumps({**json.loads((MAXIS / "job.json").read_text()), "brief": "Sell phones.", **over})


async def post_job(client, request: str | None = None, master: tuple | None = None, headers=None):
    master = master or ("master.mp4", (MAXIS / "master.mp4").read_bytes(), "video/mp4")
    files = [("master", master),
             ("references", ("ref.png", (MAXIS / "reference-static.png").read_bytes(), "image/png"))]
    return await client.post("/jobs", data={"request": request or maxis_request()}, files=files, headers=headers)


async def wait_for(client, job_id: str, want: set[str], timeout: float = 5.0) -> dict:
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while True:
        st = (await client.get(f"/jobs/{job_id}")).json()
        if st["status"] in want or loop.time() > end:
            return st
        await asyncio.sleep(0.02)


async def test_job_lifecycle(client, jobs_env, monkeypatch):
    gate = asyncio.Event()
    monkeypatch.setattr(jobs_mod, "JOB_GRAPH", FakeGraph(gate))
    t0 = asyncio.get_running_loop().time()
    r = await post_job(client)
    assert asyncio.get_running_loop().time() - t0 < 1.0
    assert r.status_code == 202
    body = r.json()
    job_id = body["job_id"]
    assert body["status"] == "queued" and body["poll_after_s"] == 5

    first = (await client.get(f"/jobs/{job_id}")).json()
    assert first["status"] in {"queued", "running"}
    assert set(first["sizes"]) == {"300x250", "728x90", "320x50", "160x600", "970x250"}

    assert (await wait_for(client, job_id, {"running"}))["status"] == "running"
    assert (await client.get(f"/jobs/{job_id}/result")).status_code == 409

    gate.set()
    assert (await wait_for(client, job_id, {"completed"}))["status"] == "completed"
    res = await client.get(f"/jobs/{job_id}/result")
    assert res.status_code == 200 and res.json()["status"] == "completed"

    inputs = jobs_env.job_dir(job_id) / "inputs"
    assert (inputs / "master.mp4").exists() and (inputs / "reference_0.png").exists()
    assert (inputs / "request.json").exists()


async def test_copy_and_boxes_uploads_are_saved(client, jobs_env, monkeypatch):
    monkeypatch.setattr(jobs_mod, "JOB_GRAPH", FakeGraph())
    files = [("master", ("master.mp4", (MAXIS / "master.mp4").read_bytes(), "video/mp4")),
             ("copy", ("copy.json", b'{"s1.headline": "x"}', "application/json")),
             ("boxes", ("boxes.json", b"{}", "application/json"))]
    r = await client.post("/jobs", data={"request": maxis_request()}, files=files)
    assert r.status_code == 202
    inputs = jobs_env.job_dir(r.json()["job_id"]) / "inputs"
    assert (inputs / "copy.json").read_bytes() == b'{"s1.headline": "x"}'
    assert (inputs / "boxes.json").read_bytes() == b"{}"
    await wait_for(client, r.json()["job_id"], {"completed"})


async def test_unknown_job_is_404(client):
    assert (await client.get("/jobs/j_nope")).status_code == 404
    assert (await client.get("/jobs/j_nope/result")).status_code == 404


async def test_unknown_brand_is_422(client):
    r = await post_job(client, maxis_request(brand_id="nobody"))
    assert r.status_code == 422 and "nobody" in r.text


async def test_invalid_request_is_422(client):
    r = await post_job(client, '{"brand_id": "maxis"}')
    assert r.status_code == 422


async def test_corrupt_master_is_422(client, jobs_env):
    r = await post_job(client, master=("master.mp4", os.urandom(4096), "video/mp4"))
    assert r.status_code == 422 and "master.mp4" in r.text
    assert not list(jobs_env.root.iterdir())                      # the half-saved job folder is removed


async def test_path_traversal_is_404(client, monkeypatch):
    monkeypatch.setattr(jobs_mod, "JOB_GRAPH", FakeGraph())
    job_id = (await post_job(client)).json()["job_id"]
    assert (await client.get(f"/jobs/{job_id}/files/inputs/request.json")).status_code == 200
    assert (await client.get(f"/jobs/{job_id}/files/../../etc/passwd")).status_code == 404
    assert (await client.get(f"/jobs/{job_id}/files/%2e%2e/%2e%2e/%2e%2e/etc/passwd")).status_code == 404
    assert (await client.get(f"/jobs/{job_id}/files/inputs")).status_code == 404       # a directory
    await wait_for(client, job_id, {"completed"})


async def test_failure_writes_result_with_error_and_issues(client, monkeypatch):
    monkeypatch.setattr(jobs_mod, "JOB_GRAPH", FailingGraph())
    job_id = (await post_job(client)).json()["job_id"]
    st = await wait_for(client, job_id, {"failed"})
    assert st["status"] == "failed" and st["error_code"] == "BRAND_MISMATCH"
    res = await client.get(f"/jobs/{job_id}/result")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "failed" and body["error_code"] == "BRAND_MISMATCH"
    assert [i["code"] for i in body["issues"]] == ["CLICK_URL_REDIRECT"]


async def test_crash_is_reported_as_crash(client, monkeypatch):
    class Boom:
        async def ainvoke(self, *a, **k):
            raise RuntimeError("boom")

    monkeypatch.setattr(jobs_mod, "JOB_GRAPH", Boom())
    job_id = (await post_job(client)).json()["job_id"]
    st = await wait_for(client, job_id, {"failed"})
    assert st["error_code"] == "CRASH"
    assert (await client.get(f"/jobs/{job_id}/result")).json()["error_code"] == "CRASH"


async def test_single_prompt_mode_uses_baseline_graph(client, monkeypatch):
    monkeypatch.setattr(jobs_mod, "JOB_GRAPH", FailingGraph())
    monkeypatch.setattr(jobs_mod, "BASELINE_GRAPH", FakeGraph())
    job_id = (await post_job(client, maxis_request(mode="single_prompt"))).json()["job_id"]
    assert (await wait_for(client, job_id, {"completed", "failed"}))["status"] == "completed"


def test_recover_marks_running_job_interrupted(data_dir):
    d = data_dir / "jobs" / "j_old"
    d.mkdir(parents=True)
    t = datetime.now(timezone.utc)
    (d / "job.json").write_text(JobStatus(job_id="j_old", status="running", stage="building",
                                          created_at=t, updated_at=t).model_dump_json())
    done = data_dir / "jobs" / "j_done"
    done.mkdir()
    (done / "job.json").write_text(JobStatus(job_id="j_done", status="completed",
                                             created_at=t, updated_at=t).model_dump_json())
    m = JobManager(browser=None)
    m.recover()
    assert m.jobs["j_old"].status == "failed" and m.jobs["j_old"].error_code == "INTERRUPTED"
    assert m.jobs["j_done"].status == "completed"
    res = json.loads((d / "result.json").read_text())
    assert res["status"] == "failed" and res["error_code"] == "INTERRUPTED"
    assert JobStatus.model_validate_json((d / "job.json").read_text()).error_code == "INTERRUPTED"


async def test_token_required_when_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "agent_token", "s3cret")
    monkeypatch.setattr(jobs_mod, "JOB_GRAPH", FakeGraph())
    assert (await post_job(client)).status_code == 401
    assert (await client.get("/jobs/j_x")).status_code == 401
    assert (await client.get("/jobs/j_x", headers={"authorization": "Bearer wrong"})).status_code == 401
    ok = await post_job(client, headers={"authorization": "Bearer s3cret"})
    assert ok.status_code == 202
    job_id = ok.json()["job_id"]
    status = await client.get(f"/jobs/{job_id}", headers={"authorization": "Bearer s3cret"})
    assert status.status_code == 200
    await asyncio.sleep(0.2)                                       # let the fake graph finish


async def test_healthz_and_brands(client):
    assert (await client.get("/healthz")).json() == {"ok": True}
    assert "maxis" in (await client.get("/brands")).json()
