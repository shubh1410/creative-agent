import json
import shutil
from pathlib import Path

import httpx
import pytest
import respx
from mcp.server.mcpserver import Image
from mcp.server.mcpserver.exceptions import ToolError

from creative_agent.config import settings
from creative_agent.mcp_server import server

ROOT = Path(__file__).resolve().parent.parent
MAXIS = ROOT / "inputs" / "maxis-merdeka"
AGENT = "http://agent.test"
JPEG = b"\xff\xd8\xff\xe0fakejpeg"


@pytest.fixture(autouse=True)
def agent(monkeypatch):
    monkeypatch.setattr(settings, "agent_url", AGENT)
    monkeypatch.setattr(settings, "agent_token", None)
    monkeypatch.setattr(server, "_client", None)
    with respx.mock(base_url=AGENT, assert_all_called=False) as r:
        yield r


def result_json(**over):
    sizes = [{"size": s, "status": "approved", "attempts": 1, "failed_checks": [], "bytes": 20480,
              "backup_path": f"sizes/{s}/backup.jpg"} for s in ("300x250", "728x90")]
    return {"job_id": "j_1", "status": "completed", "sizes": sizes, "issues":
            [{"code": "CLICK_URL_REDIRECT", "severity": "warn", "message": "redirected", "fix": None}],
            "usage": {"cost_usd": 0.42}, "contact_sheet": "contact_sheet.png", **over}


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "job"
    d.mkdir()
    for f in ("job.json", "brief.md", "master.mp4", "reference-static.png"):
        shutil.copy(MAXIS / f, d / f)
    return d


async def test_create_job_returns_id_and_next(agent, folder):
    route = agent.post("/jobs").respond(202, json={"job_id": "j_1", "status": "queued", "poll_after_s": 5})
    out = await server.create_job(str(folder))
    assert out["job_id"] == "j_1" and out["poll_after_s"] == 5
    assert "get_job_status('j_1')" in out["next"] and "get_result('j_1'" in out["next"]
    body = route.calls.last.request.content
    assert b'name="master"' in body and b'name="references"' in body and b'name="request"' in body
    assert json.loads(body.split(b'name="request"')[1].split(b"\r\n\r\n", 1)[1].split(b"\r\n--")[0])["brief"]


async def test_create_job_includes_copy_and_boxes(agent, folder):
    (folder / "copy.json").write_text('{"s1.headline": "x"}')
    (folder / "boxes.json").write_text("{}")
    route = agent.post("/jobs").respond(202, json={"job_id": "j_1", "status": "queued", "poll_after_s": 5})
    await server.create_job(str(folder))
    body = route.calls.last.request.content
    assert b'name="copy"' in body and b'name="boxes"' in body


async def test_create_job_missing_files(agent, tmp_path, folder):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ToolError, match="no job.json"):
        await server.create_job(str(empty))
    (folder / "master.mp4").unlink()
    with pytest.raises(ToolError, match=r"no master\.\* file"):
        await server.create_job(str(folder))


async def test_agent_error_status_is_reported(agent, folder):
    agent.post("/jobs").respond(422, json={"detail": "unknown brand_id 'x'"})
    with pytest.raises(ToolError, match="422.*unknown brand_id"):
        await server.create_job(str(folder))


async def test_agent_down_mentions_make_agent_for_every_tool(agent, folder):
    agent.route().mock(side_effect=httpx.ConnectError("refused"))
    calls = [server.create_job(str(folder)), server.get_job_status("j_1"), server.get_result("j_1"),
             server.ad_html("j_1", "300x250"), server.get_result("j_1", include_screenshots=True)]
    for c in calls:
        with pytest.raises(ToolError, match="make agent"):
            await c


async def test_get_job_status(agent):
    agent.get("/jobs/j_1").respond(200, json={"job_id": "j_1", "status": "running", "stage": "building 300x250"})
    assert (await server.get_job_status("j_1"))["stage"] == "building 300x250"


async def test_get_result_409_says_not_finished(agent):
    agent.get("/jobs/j_1/result").respond(409, json={"detail": {"status": "running", "stage": "planning"}})
    with pytest.raises(ToolError, match="not finished yet"):
        await server.get_result("j_1")


async def test_get_result_wait_polls_until_done(agent, monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr("creative_agent.client.asyncio.sleep", no_sleep)
    states = iter(["running", "running", "completed"])
    agent.get("/jobs/j_1").mock(side_effect=lambda req: httpx.Response(200, json={"status": next(states)}))
    agent.get("/jobs/j_1/result").respond(200, json=result_json())
    out = await server.get_result("j_1", wait_s=30)
    assert out["status"] == "completed" and not list(states)


async def test_get_result_summary(agent):
    agent.get("/jobs/j_1/result").respond(200, json=result_json())
    out = await server.get_result("j_1")
    assert out["status"] == "completed" and out["error"] is None and out["cost_usd"] == 0.42
    assert [s["size"] for s in out["sizes"]] == ["300x250", "728x90"] and out["sizes"][0]["kb"] == 20.0
    assert out["sizes"][0]["html"] == f"{AGENT}/jobs/j_1/files/sizes/300x250/index.html"
    assert out["issues"] == ["[warn] CLICK_URL_REDIRECT: redirected"]
    assert out["contact_sheet"].endswith("/contact_sheet.png") and out["trace"].endswith("/trace.jsonl")


async def test_get_result_reports_failure(agent):
    agent.get("/jobs/j_2/result").respond(200, json={"job_id": "j_2", "status": "failed", "error_code": "BRAND_MISMATCH",
                                                     "error": "wrong brand", "sizes": [], "issues": []})
    out = await server.get_result("j_2")
    assert out["status"] == "failed" and out["error"] == "BRAND_MISMATCH: wrong brand" and out["sizes"] == []


async def test_get_result_with_screenshots(agent):
    agent.get("/jobs/j_1/result").respond(200, json=result_json())
    agent.get(url__regex=r"/jobs/j_1/files/sizes/.*/backup\.jpg").respond(200, content=JPEG)
    out = await server.get_result("j_1", include_screenshots=True)
    assert json.loads(out[0])["status"] == "completed"
    assert len(out) == 3 and all(isinstance(i, Image) for i in out[1:])


async def test_ad_html_resource(agent):
    agent.get("/jobs/j_1/files/sizes/300x250/index.html").respond(200, text="<html>ad</html>")
    assert await server.ad_html("j_1", "300x250") == "<html>ad</html>"
    contents = await server.mcp.read_resource("ads://jobs/j_1/300x250/html")
    assert list(contents)[0].content == "<html>ad</html>"


async def test_tools_and_resource_are_registered():
    names = {t.name for t in await server.mcp.list_tools()}
    assert names == {"create_job", "get_job_status", "get_result"}
    templates = await server.mcp.list_resource_templates()
    assert [t.uri_template for t in templates] == ["ads://jobs/{job_id}/{size}/html"]


async def test_tool_call_through_server_and_nothing_on_stdout(agent, capsys):
    agent.get("/jobs/j_1").respond(200, json={"job_id": "j_1", "status": "queued"})
    res = await server.mcp.call_tool("get_job_status", {"job_id": "j_1"})
    assert not res.is_error
    with pytest.raises(ToolError):
        agent.get("/jobs/j_9/result").respond(409, json={"detail": "x"})
        await server.mcp.call_tool("get_result", {"job_id": "j_9"})
    assert capsys.readouterr().out == ""


def test_token_header_is_sent(monkeypatch):
    from creative_agent.client import AgentClient

    monkeypatch.setattr(settings, "agent_token", "tok")
    assert AgentClient().http.headers["authorization"] == "Bearer tok"
