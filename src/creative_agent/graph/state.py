"""State and runtime-context types for the LangGraph graphs.

* :class:`JobState`   – shared state of the job-level graph.
* :class:`SizeState`  – state of one per-size worker (copied from JobState by ``dispatch_sizes``).
* :class:`JobContext` – non-serialisable runtime objects (LLMs, browser, tracer) passed to every node.
"""
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.language_models import BaseChatModel
from playwright.async_api import Browser

from ..brand import BrandKit
from ..models import Element, Issue, JobRequest, LayoutNotes, Plan, QCReport, SizePlan, SizeResult
from ..trace import Tracer


def merge_dicts(a: dict, b: dict) -> dict:
    """Reducer that combines dict updates from parallel branches (later keys win)."""
    return {**a, **b}


class JobState(TypedDict, total=False):
    """Job-level graph state. ``total=False``: keys appear as the nodes that produce them have run."""

    job_id: str
    request: JobRequest
    master_path: str
    reference_paths: list[str]
    master_spec_path: str                        # JSON file written by ingest_master (see MasterSpec)
    layout_notes: list[LayoutNotes]
    brand: BrandKit
    content: list[Element]                       # final text elements after input checks / copy overrides
    issues: Annotated[list[Issue], operator.add]   # reducer: lists from different nodes are concatenated
    plan: Plan
    baseline_ads: dict[str, dict[str, Any]]      # baseline mode only
    results: Annotated[dict[str, SizeResult], merge_dicts]   # reducer: each size worker adds its own entry
    job_status: str


class SizeState(TypedDict, total=False):
    """State of one size's build/QC loop."""

    job_id: str
    size: str
    size_plan: SizePlan
    brand: BrandKit
    content: dict[str, Element]                  # by element id
    assets: dict[str, str]                       # asset id → crop path
    scene_assets: dict[str, str]                 # scene id → asset id
    layout_note: str
    click_url: str
    attempt: int                                 # build attempts made so far
    model_html_path: str                         # what the model wrote (reused on retries)
    html_path: str                               # what code finalised (what QC checks)
    contract_errors: list[str]
    budget_floor_hit: bool                       # file-size reduction reached its limit and still wasn't small enough
    qc: QCReport
    result: SizeResult


@dataclass
class JobContext:                                # LangGraph runtime context; never stored in state
    """Per-job runtime dependencies. Kept out of the state because they are not serialisable/checkpointable."""

    job_id: str
    job_dir: Path
    tracer: Tracer
    browser: Browser                             # shared headless Chromium
    page_sem: Any                                # asyncio.Semaphore limiting concurrently open pages
    llm: BaseChatModel
    vision_llm: BaseChatModel
    model_name: str                              # recorded in traces for cost lookup
    vision_model_name: str
    set_stage: Callable[..., None]               # (stage, size=None, size_status=None) -> updates the job status
    landing_cache: dict[str, int]                # url -> HTTP status, see net.url_status


def initial_state(job_id: str, req: JobRequest, master: Path, refs: list[Path]) -> JobState:
    """Starting state handed to the graph; accumulator keys are pre-seeded so their reducers have a base value."""
    return {"job_id": job_id, "request": req, "master_path": str(master),
            "reference_paths": [str(r) for r in refs], "issues": [], "results": {}, "layout_notes": []}
