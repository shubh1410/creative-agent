"""Pydantic data models shared across the pipeline: job requests, the parsed master creative, the layout plan,
QC reports and the final job result/status objects."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, HttpUrl

# Supported banner sizes ("WIDTHxHEIGHT" in pixels). ALL_SIZES is the runtime list with the same members.
Size = Literal["300x250", "728x90", "320x50", "160x600", "970x250"]
ALL_SIZES: list[str] = ["300x250", "728x90", "320x50", "160x600", "970x250"]
# What a layout element is: a piece of text, the logo, or a product image.
Role = Literal["headline", "subhead", "body", "cta", "legal", "highlight", "logo", "product_image"]
# Roles that carry text (everything except the image-like roles).
TEXT_ROLES = {"headline", "subhead", "body", "cta", "legal", "highlight"}


def dims(size: str) -> tuple[int, int]:
    """Split a ``"300x250"`` size string into ``(width, height)`` integers."""
    w, h = size.split("x")
    return int(w), int(h)


class BBox(BaseModel):
    """Axis-aligned rectangle in pixels: top-left corner ``(x, y)`` plus width ``w`` and height ``h``."""

    x: int
    y: int
    w: int
    h: int

    def inside(self, W: int, H: int) -> bool:
        """True if the box has positive area and lies fully within a ``W`` x ``H`` canvas."""
        return self.x >= 0 and self.y >= 0 and self.w > 0 and self.h > 0 and self.x + self.w <= W and self.y + self.h <= H

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Corner form ``(left, top, right, bottom)``, as used by PIL's ``crop``."""
        return self.x, self.y, self.x + self.w, self.y + self.h

    def scaled(self, f: float) -> "BBox":
        """New box with every coordinate multiplied by ``f`` and rounded (e.g. to map back to full resolution)."""
        return BBox(x=round(self.x * f), y=round(self.y * f), w=round(self.w * f), h=round(self.h * f))


class Element(BaseModel):
    """One text/logo/product element found in a master scene."""

    id: str                                  # "s2.headline"
    role: Role
    text: str | None = None
    highlights: list[str] = []               # substrings of ``text`` to emphasise (e.g. a price)
    bbox: BBox
    text_color: str | None = None            # palette name, set by check_inputs


class Scene(BaseModel):
    """One scene (video segment, or the single frame of an image master) and the elements on it."""

    id: str                                  # "s1" … "s5", "end"
    start_s: float | None = None             # time span in the video; None for image masters
    end_s: float | None = None
    frame_path: str                          # representative frame image on disk
    elements: list[Element]
    product_bbox: BBox | None = None
    product_asset: str | None = None         # key into MasterSpec.assets
    is_end_frame: bool = False               # the closing "logo + CTA" frame


class MasterSpec(BaseModel):
    """Structured description of the master creative, produced by the ingest stage."""

    source: str                              # path of the original master file
    kind: Literal["video", "image"]
    width: int
    height: int
    detected_brand: str | None
    scenes: list[Scene]
    transition_s: float | None = None        # cross-fade length between scenes, when detectable
    assets: dict[str, str] = {}              # asset id → crop path (shared crops appear once)
    cache_key: str                           # content hash used to reuse earlier ingest results

    def scene(self, sid: str) -> Scene:
        """Look up a scene by id. Raises ``StopIteration`` if it doesn't exist."""
        return next(s for s in self.scenes if s.id == sid)


class LayoutNotes(BaseModel):
    """Free-text layout guidance extracted from a reference creative, keyed by aspect ratio."""

    source: str = ""
    aspect: Literal["square", "landscape", "portrait"]
    notes: str


class Issue(BaseModel):
    """A problem or observation about the job inputs/outputs, surfaced to the user."""

    code: str
    severity: Literal["info", "warn", "high"]
    message: str
    fix: str | None = None                   # suggested remedy, if any


class CtaRule(BaseModel):
    """Constraint on the call-to-action button."""

    use_text: str                            # exact label the CTA must show


class JobRequest(BaseModel):
    """What the caller asks for: which brand, the brief, the click-through URL and the sizes to produce."""

    brand_id: str                            # folder name under settings.brands_dir
    brief: str
    click_url: HttpUrl
    sizes: list[Size] = ALL_SIZES            # type: ignore[assignment]
    cta: CtaRule | None = None
    # "agent" = full plan/build/QC graph; "single_prompt" = one-shot baseline used for comparison.
    mode: Literal["agent", "single_prompt"] = "agent"


class FramePlan(BaseModel):
    """Plan for one animation frame of a banner."""

    scene_id: str                            # which master scene this frame is based on
    keep: list[str]                          # element ids retained from that scene
    show_products: bool
    duration_ms: int


class SizePlan(BaseModel):
    """Plan for a single banner size: its ordered frames and which one doubles as the static backup image."""

    size: Size
    frames: list[FramePlan]
    backup_frame: int                        # index into ``frames``
    notes: str = ""


class Plan(BaseModel):
    """The planner's full output: one SizePlan per requested size."""

    sizes: list[SizePlan]


class CheckResult(BaseModel):
    """Outcome of one QC check."""

    name: str
    passed: bool
    fixable: bool                            # whether the builder can plausibly repair it on retry
    message: str = ""
    frame: int | None = None                 # frame index the check applied to, when frame-specific


class QCReport(BaseModel):
    """All QC checks for one size on one build attempt."""

    size: str
    attempt: int
    passed: bool
    checks: list[CheckResult]
    screenshots: list[str]


class SizeResult(BaseModel):
    """Final outcome for one size."""

    size: str
    status: Literal["approved", "failed"]
    attempts: int                            # build attempts used
    html_path: str | None = None
    backup_path: str | None = None
    screenshots: list[str] = []
    failed_checks: list[str] = []
    bytes: int | None = None                 # size of the final HTML bundle
    error: str | None = None


class Usage(BaseModel):
    """Aggregate LLM usage and cost for a job."""

    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0              # None when any call used an unpriced model
    by_node: dict[str, float | None] = {}     # cost per graph node


JobStatusValue = Literal["queued", "running", "completed", "completed_with_failures", "failed"]


class JobStatus(BaseModel):
    """Live status of a job, returned while polling."""

    job_id: str
    status: JobStatusValue
    stage: str = "queued"                    # current graph stage name
    sizes: dict[str, str] = {}               # size → per-size state
    error_code: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class JobResult(BaseModel):
    """Final result of a job: per-size outcomes, issues, usage and where to find the trace."""

    job_id: str
    status: JobStatusValue
    sizes: list[SizeResult] = []
    issues: list[Issue] = []
    usage: Usage = Usage()
    trace_path: str = "trace.jsonl"
    contact_sheet: str | None = None         # overview image of all screenshots
    error_code: str | None = None
    error: str | None = None
