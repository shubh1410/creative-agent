"""MCP server exposing the agent as tools for AI assistants.

This is a thin adapter: every tool forwards to the agent HTTP server through :class:`AgentClient`, so the MCP
process holds no job state and the agent can run elsewhere. Run over stdio (default) or ``--http``.

Note: the docstrings of the ``@mcp.tool`` functions are sent to the calling model as the tool descriptions, so
they are written for that audience; implementation notes are kept as comments instead.
"""
import json
import logging
import sys

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

from ..client import AgentClient, AgentError, AgentUnavailable
from ..config import settings

logging.basicConfig(stream=sys.stderr, level=logging.INFO)     # stdout is the MCP channel: never print to it
mcp = MCPServer("creative-agent")
_client: AgentClient | None = None


def client() -> AgentClient:
    """Lazily create a single shared AgentClient (so importing this module doesn't open connections)."""
    global _client
    _client = _client or AgentClient()
    return _client


async def call(coro):
    """Await an AgentClient call, converting its errors into ``ToolError`` so the model sees a readable
    message instead of a stack trace."""
    try:
        return await coro
    except AgentUnavailable as e:
        raise ToolError(str(e))
    except AgentError as e:
        raise ToolError(f"The agent server returned {e.status}: {e.detail}")


@mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False))
async def create_job(input_dir: str) -> dict:
    """Start a resize job from a local folder containing job.json, brief.md and master.* (optionally reference*,
    copy.json, boxes.json). Returns at once with a job_id; the job runs in the background."""
    try:
        r = await call(client().submit(input_dir))
    except FileNotFoundError as e:               # raised locally while reading the folder, before any HTTP call
        raise ToolError(str(e))
    # Tell the model how to follow up so it doesn't have to guess the polling protocol.
    return {**r, "next": f"Call get_job_status('{r['job_id']}') every {r['poll_after_s']}s, "
                         f"or get_result('{r['job_id']}', wait_s=60)."}


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def get_job_status(job_id: str) -> dict:
    """Current stage and the status of each size."""
    return await call(client().status(job_id))


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def get_result(job_id: str, wait_s: int = 0, include_screenshots: bool = False):
    """Summary of a finished job: status per size, failed checks, input issues, cost and links to every file.
    wait_s (at most 60) waits for the job to finish first. include_screenshots adds the backup image of each size."""
    if wait_s:
        # Capped at 60 s to stay under typical MCP tool-call timeouts; the caller can simply call again.
        await call(client().wait(job_id, min(wait_s, 60)))
    try:
        res = await client().result(job_id)
    except AgentError as e:
        if e.status == 409:                      # server says "not finished yet"
            raise ToolError(f"Job not finished yet ({e.detail}). Call get_result again later, or pass wait_s.")
        raise ToolError(f"The agent server returned {e.status}: {e.detail}")
    except AgentUnavailable as e:
        raise ToolError(str(e))

    # Build a compact summary with file URLs instead of inlining HTML/images, which keeps the reply small.
    base = f"{settings.agent_url}/jobs/{job_id}/files"
    summary = {
        "job_id": job_id, "status": res["status"],
        "error": f"{res['error_code']}: {res['error']}" if res.get("error_code") else None,
        "sizes": [{"size": s["size"], "status": s["status"], "attempts": s["attempts"],
                   "failed_checks": s["failed_checks"], "kb": round((s.get("bytes") or 0) / 1024, 1),
                   "html": f"{base}/sizes/{s['size']}/index.html", "backup": f"{base}/sizes/{s['size']}/backup.jpg"}
                  for s in res.get("sizes", [])],
        "issues": [f"[{i['severity']}] {i['code']}: {i['message']}" for i in res.get("issues", [])],
        "cost_usd": res.get("usage", {}).get("cost_usd"),
        "contact_sheet": f"{base}/contact_sheet.png" if res.get("contact_sheet") else None,
        "trace": f"{base}/trace.jsonl",
    }
    if not include_screenshots:
        return summary
    # Optionally attach the backup JPEG of each size that has one, so the model can actually look at the ads.
    images = [Image(data=await call(client().file(job_id, f"sizes/{s['size']}/backup.jpg")), format="jpeg")
              for s in res.get("sizes", []) if s.get("backup_path")]
    return [json.dumps(summary, indent=2), *images]


@mcp.resource("ads://jobs/{job_id}/{size}/html", mime_type="text/html")
async def ad_html(job_id: str, size: str) -> str:
    """The final HTML of one size."""
    return (await call(client().file(job_id, f"sizes/{size}/index.html"))).decode()


def main() -> None:
    """Console-script entry point: stdio by default, or streamable HTTP when ``--http`` is passed."""
    if "--http" in sys.argv:
        mcp.run(transport="streamable-http", port=8001)         # 8000 is the agent server
    else:
        mcp.run(transport="stdio")
