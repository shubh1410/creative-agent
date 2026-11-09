"""Job lifecycle management for the agent server: accepting uploads, running the LangGraph pipeline in the
background under concurrency limits, persisting status to disk, and recovering after restarts.

On-disk layout per job (``<data_dir>/jobs/<job_id>/``)::

    job.json       JobStatus (rewritten atomically on every update)
    inputs/        uploaded master, references, copy.json, boxes.json, request.json
    trace.jsonl    event log (see trace.py)
    result.json    final JobResult (also written for failed jobs)
"""
import asyncio
import logging
import os
import shutil
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import imageio_ffmpeg
from fastapi import UploadFile
from PIL import Image

from ..config import settings
from ..errors import JobInputError, LLMOutputError
from ..graph.graph import BASELINE_GRAPH, JOB_GRAPH
from ..graph.state import JobContext, initial_state
from ..llm import make_llm
from ..models import Issue, JobRequest, JobResult, JobStatus, Usage
from ..trace import TraceCallback, Tracer

# Statuses after which a job will not change again.
TERMINAL = {"completed", "completed_with_failures", "failed"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm"}
MAX_UPLOAD = 50 * 1024 * 1024                    # total bytes of all files in one job
log = logging.getLogger(__name__)


def now() -> datetime:
    """Current time, timezone-aware (UTC)."""
    return datetime.now(timezone.utc)


def write_atomic(path: Path, text: str) -> None:
    """Write ``text`` so readers never see a half-written file: write a temp file, then rename over the target."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)                        # atomic on the same filesystem


async def save_upload(f: UploadFile, dest: Path) -> Path:
    """Stream an uploaded file to ``dest`` in 1 MiB chunks (avoids loading large videos into memory)."""
    with dest.open("wb") as out:
        while chunk := await f.read(1 << 20):
            out.write(chunk)
    return dest


def check_master_readable(p: Path) -> None:
    """Fail fast on corrupt masters, before spending any LLM budget.

    Videos: ffmpeg must report a frame size. Images: Pillow's ``verify`` must succeed.

    Raises:
        ValueError: If the file can't be opened as a video or image.
    """
    try:
        if p.suffix.lower() in VIDEO_EXT:
            gen = imageio_ffmpeg.read_frames(str(p))
            meta = next(gen)                     # first item is the stream metadata, not a frame
            gen.close()
            if not meta.get("size"):
                raise ValueError
        else:
            with Image.open(p) as im:
                im.verify()
    except Exception:
        raise ValueError(f"master file '{p.name}' can't be read as a video or image")


class JobManager:
    """Owns all jobs of a running server: status registry, background tasks and concurrency limits."""

    def __init__(self, browser):
        """
        Args:
            browser: Shared Playwright browser used for rendering and QC by all jobs.
        """
        self.browser = browser
        self.jobs: dict[str, JobStatus] = {}                       # job_id -> status (in memory + mirrored to disk)
        self._tasks: set[asyncio.Task] = set()
        self._job_sem = asyncio.Semaphore(settings.max_concurrent_jobs)   # jobs running at once
        self._page_sem = asyncio.Semaphore(settings.max_open_pages)       # browser pages open at once, all jobs
        self.root = settings.data_dir / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        """Folder holding everything for one job."""
        return self.root / job_id

    def _persist(self, job_id: str) -> None:
        """Mirror the in-memory status to ``job.json`` so it survives a restart."""
        write_atomic(self.job_dir(job_id) / "job.json", self.jobs[job_id].model_dump_json(indent=2))

    def update(self, job_id: str, *, status=None, stage=None, size=None, size_status=None,
               error_code=None, error=None) -> None:
        """Apply a partial update to a job's status (only the arguments given), stamp the time, and persist."""
        st = self.jobs[job_id]
        if status:
            st.status = status
        if stage:
            st.stage = stage
        if size and size_status:
            st.sizes[size] = size_status
        if error_code:
            st.error_code = error_code
        if error:
            st.error = error
        st.updated_at = now()
        self._persist(job_id)

    def recover(self) -> None:
        """Load past jobs; anything left running by a crash or restart is marked failed."""
        for f in self.root.glob("*/job.json"):
            st = JobStatus.model_validate_json(f.read_text())
            self.jobs[st.job_id] = st
            if st.status not in TERMINAL:
                self.update(st.job_id, status="failed", stage="interrupted",
                            error_code="INTERRUPTED", error="interrupted by restart")
                self._write_failure(st.job_id, None)     # no tracer: the old process's in-memory one is gone

    async def submit(self, req: JobRequest, master: UploadFile, references: list[UploadFile],
                     copy: UploadFile | None, boxes: UploadFile | None) -> str:
        """Save the uploads, validate them, register the job as queued and start it in the background.

        Returns:
            The new job id (``"j_"`` + 8 hex chars).

        Raises:
            ValueError: If the uploads exceed the size limit or the master can't be read. Any partial job
                folder is removed first.
        """
        job_id = "j_" + uuid.uuid4().hex[:8]
        inputs = self.job_dir(job_id) / "inputs"
        inputs.mkdir(parents=True)
        try:
            # Keep the original extension (lower-cased): later stages pick video vs image handling from it.
            master_path = await save_upload(master, inputs / f"master{Path(master.filename or '').suffix.lower()}")
            refs = [await save_upload(r, inputs / f"reference_{i}{Path(r.filename or '').suffix.lower()}")
                    for i, r in enumerate(references)]
            if copy:
                await save_upload(copy, inputs / "copy.json")
            if boxes:
                await save_upload(boxes, inputs / "boxes.json")
            if sum(p.stat().st_size for p in inputs.iterdir()) > MAX_UPLOAD:
                raise ValueError("upload is larger than 50 MB")
            check_master_readable(master_path)
        except ValueError:
            shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
            raise
        (inputs / "request.json").write_text(req.model_dump_json(indent=2))
        self.jobs[job_id] = JobStatus(job_id=job_id, status="queued", sizes={s: "pending" for s in req.sizes},
                                      created_at=now(), updated_at=now())
        self._persist(job_id)
        task = asyncio.create_task(self._run(job_id, req, master_path, refs))
        self._tasks.add(task)                    # keep a reference so the task isn't garbage-collected
        task.add_done_callback(self._tasks.discard)
        return job_id

    async def _run(self, job_id: str, req: JobRequest, master: Path, refs: list[Path]) -> None:
        """Execute the pipeline for one job and record the outcome. Never raises: every failure is converted
        into a failed job status plus a ``result.json``."""
        d = self.job_dir(job_id)
        tracer = Tracer(d, job_id)
        async with self._job_sem:                # queued jobs wait here until a slot frees up
            self.update(job_id, status="running", stage="ingesting master")
            # "single_prompt" mode runs the one-shot baseline graph instead of the full agent graph.
            graph = JOB_GRAPH if req.mode == "agent" else BASELINE_GRAPH
            try:
                # Per-job runtime context handed to every graph node.
                ctx = JobContext(
                    job_id=job_id, job_dir=d, tracer=tracer, browser=self.browser, page_sem=self._page_sem,
                    llm=make_llm(settings.model), vision_llm=make_llm(settings.vision_model),
                    model_name=settings.model, vision_model_name=settings.vision_model,
                    set_stage=lambda stage, size=None, size_status=None:
                        self.update(job_id, stage=stage, size=size, size_status=size_status),
                    landing_cache={},            # url -> HTTP status, shared by input checks and QC
                )
                final = await graph.ainvoke(initial_state(job_id, req, master, refs), context=ctx,
                                            config={"callbacks": [TraceCallback(tracer)], "recursion_limit": 100})
                self.update(job_id, status=final["job_status"], stage="done")
            except (JobInputError, LLMOutputError) as e:
                # Expected failures: unusable inputs, or the model could not produce valid output.
                code = getattr(e, "code", "LLM_OUTPUT_INVALID")
                await tracer.emit("error", code=code, message=str(e))
                self.update(job_id, status="failed", stage="stopped", error_code=code, error=str(e))
                self._write_failure(job_id, tracer)
            except Exception as e:
                # Unexpected bug: keep the full traceback in the trace for debugging.
                log.exception("job %s crashed", job_id)
                await tracer.emit("error", code="CRASH", message=repr(e), traceback=traceback.format_exc())
                self.update(job_id, status="failed", stage="crashed", error_code="CRASH", error=repr(e))
                self._write_failure(job_id, tracer)

    def _write_failure(self, job_id: str, tracer: Tracer | None) -> None:
        """Failed jobs still get a result.json, with the issues logged before the failure."""
        st = self.jobs[job_id]
        res = JobResult(job_id=job_id, status="failed", error_code=st.error_code, error=st.error,
                        issues=[Issue(**i) for i in tracer.issues()] if tracer else [],
                        usage=tracer.usage() if tracer else Usage())
        write_atomic(self.job_dir(job_id) / "result.json", res.model_dump_json(indent=2))

    async def shutdown(self) -> None:
        """Cancel all running job tasks and wait for them to finish unwinding."""
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
