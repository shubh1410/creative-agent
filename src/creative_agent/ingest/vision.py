"""Vision-model calls used during ingestion: locating frames in an image master, extracting scene elements, and
describing reference layouts. Each call validates the model's answer and asks for a repair if it is inconsistent."""
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image
from pydantic import BaseModel

from ..llm import ask_structured, image_part
from ..models import TEXT_ROLES, BBox, LayoutNotes, Role
from ..prompts import load_prompt

PROMPT_VERSION = "ingest-v1"           # bump when the extraction prompts change; invalidates the cache
VISION_MAX_SIDE = 1024                 # images are downscaled to this before sending; boxes are scaled back after


class ElementDraft(BaseModel):
    """One element as reported by the model (no id yet; ids are assigned in ``build_spec``)."""

    role: Role
    text: str | None = None
    highlights: list[str] = []
    bbox: BBox


class SceneDraft(BaseModel):
    """One scene as reported by the model."""

    scene_id: str
    is_end_frame: bool = False
    elements: list[ElementDraft]
    product_bbox: BBox | None = None


class ExtractionDraft(BaseModel):
    """The model's full reading of all scenes."""

    detected_brand: str | None
    scenes: list[SceneDraft]

    def rescaled(self, f: float) -> "ExtractionDraft":
        """Copy with every box multiplied by ``f`` (maps coordinates from the downscaled image back to original)."""
        d = self.model_copy(deep=True)
        for s in d.scenes:
            s.product_bbox = s.product_bbox.scaled(f) if s.product_bbox else None
            for e in s.elements:
                e.bbox = e.bbox.scaled(f)
        return d


class FrameBoxes(BaseModel):
    """Locations of the individual frames inside an image master."""

    frames: list[BBox]


class LayoutDraft(BaseModel):
    """Layout descriptions found in one reference image."""

    layouts: list[LayoutNotes]


def validate_extraction(d: ExtractionDraft, scene_ids: list[str], w: int, h: int) -> list[str]:
    """Sanity-check an extraction against what we know.

    Args:
        d: The model's answer.
        scene_ids: The scene ids it was asked about.
        w, h: Size of the images the model saw (boxes must lie inside).

    Returns:
        Human-readable problems, empty if the answer is consistent.
    """
    errors = []
    got = [s.scene_id for s in d.scenes]
    if sorted(got) != sorted(scene_ids):
        errors.append(f"scene ids must be exactly {scene_ids}, got {got}")
    if sum(s.is_end_frame for s in d.scenes) > 1:
        errors.append("at most one scene can be the end frame")
    for s in d.scenes:
        if not s.elements and not s.is_end_frame:
            errors.append(f"{s.scene_id}: no elements")
        for e in s.elements:
            if not e.bbox.inside(w, h):
                errors.append(f"{s.scene_id}: {e.role} box {e.bbox.as_tuple()} is outside the {w}x{h} image")
            if e.role in TEXT_ROLES and not (e.text or "").strip():
                errors.append(f"{s.scene_id}: {e.role} has no text")
            # Highlights are emphasised parts of the text, so each must literally occur in it.
            errors += [f"{s.scene_id}: highlight '{x}' is not part of '{e.text}'" for x in e.highlights
                       if x not in (e.text or "")]
        if s.product_bbox and not s.product_bbox.inside(w, h):
            errors.append(f"{s.scene_id}: product box is outside the image")
    return errors


def _scale(W: int, H: int) -> tuple[float, int, int]:
    """Factor that fits ``W``x``H`` within ``VISION_MAX_SIDE`` (never upscales), plus the scaled width/height."""
    f = min(1.0, VISION_MAX_SIDE / max(W, H))
    return f, round(W * f), round(H * f)


async def extract(llm, frames: dict[str, Path], W: int, H: int, *, meta: dict, config) -> ExtractionDraft:
    """Ask the vision model to read all frames at once.

    The model is told the downscaled pixel size so its boxes use that coordinate space; they are scaled back
    to ``W``x``H`` before returning.
    """
    f, w, h = _scale(W, H)
    content: list[dict] = [{"type": "text", "text": f"Each frame image is {w}x{h} pixels; give boxes in those "
                            f"pixels. Scene ids, in order: {', '.join(frames)}."}]
    for sid, p in frames.items():
        content += [{"type": "text", "text": f"Scene {sid}:"}, image_part(p, VISION_MAX_SIDE)]
    msgs = [SystemMessage(load_prompt("vision_extract")), HumanMessage(content=content)]
    draft = await ask_structured(llm, ExtractionDraft, msgs, meta=meta, config=config,
                                 validate=lambda d: validate_extraction(d, list(frames), w, h))
    return draft.rescaled(1 / f)


async def detect_frames(llm, path: Path, *, meta: dict, config) -> list[BBox]:
    """Find the individual frames inside an image master (e.g. a storyboard sheet); boxes are in original pixels."""
    with Image.open(path) as im:
        f, w, h = _scale(*im.size)
    msgs = [SystemMessage(load_prompt("vision_frames")),
            HumanMessage(content=[{"type": "text", "text": f"The image is {w}x{h} pixels."},
                                  image_part(path, VISION_MAX_SIDE)])]
    res = await ask_structured(llm, FrameBoxes, msgs, meta=meta, config=config,
                               validate=lambda r: [] if r.frames and all(b.inside(w, h) for b in r.frames)
                               else ["return at least one frame box, inside the image"])
    return [b.scaled(1 / f) for b in res.frames]


async def layout_notes(llm, path: Path, *, meta: dict, config) -> list[LayoutNotes]:
    """Describe the layout(s) in a reference creative; each note is tagged with the reference file name."""
    msgs = [SystemMessage(load_prompt("reference_layout")), HumanMessage(content=[image_part(path, VISION_MAX_SIDE)])]
    res = await ask_structured(llm, LayoutDraft, msgs, meta=meta, config=config,
                               validate=lambda r: [] if r.layouts else ["describe at least one layout"])
    return [n.model_copy(update={"source": path.name}) for n in res.layouts]
