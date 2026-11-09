"""Upload a job folder to the agent server and read results. The agent only ever receives content, so it works the
same whether it runs locally or remotely."""
import asyncio
import json
import mimetypes
from pathlib import Path

import httpx

from .config import settings

# Job statuses after which nothing further will change.
TERMINAL = {"completed", "completed_with_failures", "failed"}


class AgentUnavailable(RuntimeError):
    """The agent server could not be reached (not running, wrong URL, network down)."""


class AgentError(RuntimeError):
    """The agent server answered with an HTTP error status.

    Attributes:
        status: HTTP status code.
        detail: The server's ``detail`` field if the body was JSON, otherwise the raw response text.
    """

    def __init__(self, status: int, detail):
        super().__init__(f"{status}: {detail}")
        self.status, self.detail = status, detail


def _file(field: str, p: Path) -> tuple:
    """Build one ``(field, (filename, bytes, mime))`` entry in the format httpx expects for multipart uploads."""
    return field, (p.name, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "application/octet-stream")


def job_upload(input_dir: str | Path, mode: str | None = None) -> tuple[dict, list]:
    """Turn a local job folder into multipart form data for ``POST /jobs``.

    Expected folder layout: ``job.json`` and ``master.*`` (required); ``brief.md``, ``reference*``,
    ``copy.json`` and ``boxes.json`` (optional).

    Args:
        input_dir: Path to the job folder (``~`` is expanded).
        mode: Optional override for the request's ``mode`` ("agent" or "single_prompt").

    Returns:
        ``(form_data, files)`` ready to pass as ``data=`` and ``files=`` to httpx.

    Raises:
        FileNotFoundError: If ``job.json`` or the master file is missing.
    """
    d = Path(input_dir).expanduser().resolve()
    if not (d / "job.json").exists():
        raise FileNotFoundError(f"{d} has no job.json")
    req = json.loads((d / "job.json").read_text())
    # brief.md, when present, takes precedence over a "brief" key inside job.json.
    req["brief"] = (d / "brief.md").read_text() if (d / "brief.md").exists() else req.get("brief", "")
    if mode:
        req["mode"] = mode
    # The master creative may have any extension (master.mp4, master.png, ...), so match on the stem.
    master = next((p for p in sorted(d.iterdir()) if p.stem == "master"), None)
    if master is None:
        raise FileNotFoundError(f"{d} has no master.* file")
    files = [_file("master", master)] + [_file("references", p) for p in sorted(d.glob("reference*"))]
    # Optional structured side-inputs: pre-written copy and manually specified element boxes.
    for name, field in (("copy.json", "copy"), ("boxes.json", "boxes")):
        if (d / name).exists():
            files.append(_file(field, d / name))
    return {"request": json.dumps(req)}, files


class AgentClient:
    """Thin async HTTP client for the agent server's job API."""

    def __init__(self, base_url: str | None = None, timeout: float = 60):
        """
        Args:
            base_url: Server URL; defaults to ``settings.agent_url``.
            timeout: Per-request timeout in seconds.
        """
        headers = {"Authorization": f"Bearer {settings.agent_token}"} if settings.agent_token else {}
        self.http = httpx.AsyncClient(base_url=base_url or settings.agent_url, timeout=timeout, headers=headers)

    async def _call(self, method: str, path: str, **kw) -> httpx.Response:
        """Send a request and translate failures into :class:`AgentUnavailable` / :class:`AgentError`."""
        try:
            r = await self.http.request(method, path, **kw)
        except httpx.ConnectError:
            raise AgentUnavailable(f"Agent server not reachable at {self.http.base_url}. Start it with `make agent`.")
        if r.status_code >= 400:
            # FastAPI puts error messages under "detail"; fall back to the raw body for non-JSON errors.
            try:
                detail = r.json().get("detail", r.text)
            except (ValueError, AttributeError):
                detail = r.text
            raise AgentError(r.status_code, detail)
        return r

    async def submit(self, input_dir: str | Path, mode: str | None = None) -> dict:
        """Upload a job folder and start a job. Returns the server's response (includes the ``job_id``)."""
        data, files = job_upload(input_dir, mode)
        return (await self._call("POST", "/jobs", data=data, files=files)).json()

    async def status(self, job_id: str) -> dict:
        """Current :class:`~creative_agent.models.JobStatus` of a job, as a dict."""
        return (await self._call("GET", f"/jobs/{job_id}")).json()

    async def result(self, job_id: str) -> dict:
        """Final :class:`~creative_agent.models.JobResult` of a finished job, as a dict."""
        return (await self._call("GET", f"/jobs/{job_id}/result")).json()

    async def file(self, job_id: str, path: str) -> bytes:
        """Download one output file (HTML, screenshot, trace, ...) by its path relative to the job folder."""
        return (await self._call("GET", f"/jobs/{job_id}/files/{path}")).content

    async def wait(self, job_id: str, timeout_s: float, poll_s: float = 2.0) -> dict:
        """Poll a job until it reaches a terminal status or ``timeout_s`` elapses.

        Returns:
            The last status dict seen. On timeout the job may still be running, so callers should check
            ``status`` against :data:`TERMINAL`.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while True:
            st = await self.status(job_id)
            if st["status"] in TERMINAL or loop.time() >= deadline:
                return st
            await asyncio.sleep(poll_s)
