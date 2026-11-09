"""HTTP API of the agent server (FastAPI).

Endpoints:
    POST /jobs                          upload a job (multipart) and start it; returns a job_id
    GET  /jobs/{id}                     live status
    GET  /jobs/{id}/result              final result (409 while the job is still running)
    GET  /jobs/{id}/files/{path}        any file in the job folder (HTML, screenshots, trace, ...)
    GET  /brands, GET /healthz          discovery and liveness (no auth)

If ``AGENT_TOKEN`` is set, all ``/jobs`` endpoints require ``Authorization: Bearer <token>``.
"""
import json
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from playwright.async_api import async_playwright
from pydantic import ValidationError

from ..config import settings
from ..models import JobRequest, JobStatus
from .jobs import TERMINAL, JobManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start one shared headless Chromium and the JobManager; tear both down on shutdown."""
    pw = await async_playwright().start()
    browser = await pw.chromium.launch()
    app.state.jobs = JobManager(browser)
    app.state.jobs.recover()                  # mark jobs interrupted by a previous crash/restart as failed
    try:
        yield
    finally:
        await app.state.jobs.shutdown()       # cancel running jobs before closing the browser they use
        await browser.close()
        await pw.stop()


app = FastAPI(title="creative-agent", lifespan=lifespan)


def require_token(request: Request) -> None:
    """Dependency enforcing the bearer token. No-op when no token is configured (local development)."""
    if settings.agent_token and request.headers.get("authorization") != f"Bearer {settings.agent_token}":
        raise HTTPException(401, "missing or wrong bearer token")


def jobs() -> JobManager:
    """The JobManager created during startup."""
    return app.state.jobs


@app.post("/jobs", status_code=202, dependencies=[Depends(require_token)])
async def create_job(
    request: str = Form(...),                                   # JobRequest as a JSON string
    master: UploadFile = File(...),                             # the master video or image
    references: list[UploadFile] | None = File(None),           # optional reference layouts
    copy_file: UploadFile | None = File(None, alias="copy"),   # alias: "copy" shadows a pydantic attribute
    boxes: UploadFile | None = File(None),
):
    """Validate the request, store the uploads and queue the job. Responds 202 immediately; poll for progress."""
    try:
        req = JobRequest.model_validate_json(request)
    except ValidationError as e:
        raise HTTPException(422, json.loads(e.json()))
    if not (settings.brands_dir / req.brand_id / "brand_kit.json").exists():
        raise HTTPException(422, f"unknown brand_id '{req.brand_id}'")
    try:
        job_id = await jobs().submit(req, master, references or [], copy_file, boxes)
    except ValueError as e:                      # unreadable master, upload too large
        raise HTTPException(422, str(e))
    return {"job_id": job_id, "status": "queued", "poll_after_s": 5}


@app.get("/jobs/{job_id}", response_model=JobStatus, dependencies=[Depends(require_token)])
async def job_status(job_id: str):
    """Current status and stage of a job."""
    st = jobs().jobs.get(job_id)
    if not st:
        raise HTTPException(404, "unknown job id")
    return st


@app.get("/jobs/{job_id}/result", dependencies=[Depends(require_token)])
async def job_result(job_id: str):
    """The job's ``result.json``. Returns 409 (with current status/stage) if the job hasn't finished yet."""
    st = jobs().jobs.get(job_id)
    if not st:
        raise HTTPException(404, "unknown job id")
    if st.status not in TERMINAL:
        raise HTTPException(409, {"status": st.status, "stage": st.stage})
    return json.loads((jobs().job_dir(job_id) / "result.json").read_text())


@app.get("/jobs/{job_id}/files/{path:path}", dependencies=[Depends(require_token)])
async def job_file(job_id: str, path: str):
    """Serve a file from inside the job folder."""
    base = jobs().job_dir(job_id).resolve()
    target = (base / path).resolve()
    # Resolve first, then check containment: blocks "../" traversal and symlinks pointing outside the job folder.
    if not target.is_relative_to(base) or not target.is_file():
        raise HTTPException(404, "no such file")
    return FileResponse(target)


@app.get("/brands")
async def brands():
    """Ids of all brands that have a brand kit."""
    return sorted(p.parent.name for p in settings.brands_dir.glob("*/brand_kit.json"))


@app.get("/healthz")
async def healthz():
    """Liveness probe."""
    return {"ok": True}


def main() -> None:
    """Console-script entry point: run the server on the host/port derived from ``AGENT_URL``."""
    logging.basicConfig(level=logging.INFO)
    host, port = settings.agent_host_port
    uvicorn.run(app, host=host, port=port)
