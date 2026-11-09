"""Master ingestion: turn the uploaded master (video or image) into a structured :class:`MasterSpec`.

Videos are split into stable scenes with ffmpeg; image masters are split into frame boxes by a vision model.
A vision model then reads text/elements/product regions from each frame. Results are cached by input content.
"""
import asyncio
import json
import re
from collections import Counter
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from PIL import Image

from ...errors import JobInputError
from ...ingest import cache, crops, ocr, video, vision
from ...models import TEXT_ROLES, BBox, Element, Issue, MasterSpec, Scene
from ...trace import record_issues, traced
from ..state import JobContext, JobState

SAMPLE_FPS = 4                                   # scene detection samples per second of video
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm"}


@traced("ingest_master")
async def ingest_master_node(state: JobState, runtime: Runtime[JobContext], config: RunnableConfig):
    """Produce ``work/master_spec.json`` and ``work/layout_notes.json`` (from cache when possible).

    Returns the path of the spec, the layout notes from reference creatives, and any ingest issues.
    """
    ctx = runtime.context
    ctx.set_stage("ingesting master")
    master, refs = Path(state["master_path"]), [Path(p) for p in state["reference_paths"]]
    boxes_file = ctx.job_dir / "inputs" / "boxes.json"
    # Everything that influences the result goes into the key, so a changed file, prompt or model misses the cache.
    key_files = [master, *refs, *([boxes_file] if boxes_file.exists() else [])]
    key = cache.cache_key(key_files, vision.PROMPT_VERSION, ctx.vision_model_name)

    data = cache.load(key)
    if data:
        await ctx.tracer.emit("ingest_cache_hit", key=key)
    else:
        data = await _ingest(ctx, master, refs, boxes_file, key, config)
        cache.save(key, data)

    work = ctx.job_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "master_spec.json").write_text(data.spec.model_dump_json(indent=2))
    (work / "layout_notes.json").write_text(json.dumps([n.model_dump() for n in data.notes], indent=2))
    # Issues are recorded on every run (including cache hits) so each job's result lists them.
    await record_issues(ctx, data.issues)
    return {"master_spec_path": str(work / "master_spec.json"), "layout_notes": data.notes, "issues": data.issues}


async def _ingest(ctx, master: Path, refs: list[Path], boxes_file: Path, key: str, config) -> cache.CachedIngest:
    """The uncached ingest: extract frames, ask the vision model for elements, and assemble the spec.

    Raises:
        JobInputError: ``INGEST_FAILED`` if a video has no stable scene.
    """
    out = cache.cache_dir(key)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "crops").mkdir(exist_ok=True)
    meta = {"node": "ingest_master", "model": ctx.vision_model_name, "modality": "vision"}

    frames: dict[str, Path] = {}                 # scene id ("s1", "s2", ...) -> frame image
    timing: dict[str, video.Segment] = {}        # scene id -> time span in the video (video masters only)
    if master.suffix.lower() in VIDEO_EXT:
        # Blocking ffmpeg/numpy work runs in threads so the event loop (and other jobs) keep going.
        info = await asyncio.to_thread(video.probe, master)
        W, H = info["size"]
        gray = await asyncio.to_thread(video.sample_gray, master, SAMPLE_FPS, W, H)
        segments = video.stable_segments(gray, SAMPLE_FPS)
        if not segments:
            raise JobInputError("INGEST_FAILED", "no stable scenes found in the video")
        for i, seg in enumerate(segments, 1):
            p = out / "frames" / f"s{i}.png"
            await asyncio.to_thread(video.extract_frame, master, seg.mid, p)
            frames[f"s{i}"], timing[f"s{i}"] = p, seg
        kind, transition = "video", video.mean_gap(segments)
    else:
        # Image master (e.g. a storyboard sheet): the vision model finds where each frame is, then we cut them out.
        for i, b in enumerate(await vision.detect_frames(ctx.vision_llm, master, meta=meta, config=config), 1):
            frames[f"s{i}"] = crops.crop_box(master, b, out / "frames" / f"s{i}.png", pad=0)
        with Image.open(frames["s1"]) as im:
            W, H = im.size
        kind, transition = "image", None

    draft = await vision.extract(ctx.vision_llm, frames, W, H, meta=meta, config=config)
    # boxes.json lets the user hand-correct product regions per scene.
    overrides = ({k: BBox(**v) for k, v in json.loads(boxes_file.read_text()).items()}
                 if boxes_file.exists() else {})
    spec, issues = build_spec(draft, frames, timing, kind, W, H, out, master.name, key, transition, overrides)
    notes = [n for r in refs for n in await vision.layout_notes(ctx.vision_llm, r, meta=meta, config=config)]
    return cache.CachedIngest(spec=spec, notes=notes, issues=issues)


def build_spec(draft, frames, timing, kind, W, H, out: Path, source: str, key: str, transition, overrides):
    """Turn the vision model's draft into a final :class:`MasterSpec`, cutting out product crops on the way.

    Args:
        draft: The vision model's ``ExtractionDraft`` (coordinates already in full-resolution pixels).
        frames: ``scene id -> frame image``.
        timing: ``scene id -> Segment`` (empty for image masters).
        kind, W, H, source, key, transition: Copied into the spec.
        out: Cache folder; crops are written to ``out/crops``.
        overrides: ``final scene id -> BBox`` product-box overrides from boxes.json.

    Returns:
        ``(spec, issues)``.

    Notes:
        * The end frame's scene is renamed to ``"end"``.
        * Elements get ids ``<scene>.<role>`` with ``_2``, ``_3`` ... for repeated roles in a scene.
        * Identical product crops across scenes are stored once and shared (``assets``).
    """
    by_id = {s.scene_id: s for s in draft.scenes}
    scenes, assets, issues = [], {}, []
    for sid, frame in frames.items():
        d = by_id[sid]
        final_id = "end" if d.is_end_frame else sid
        counts: Counter[str] = Counter()
        elements = []
        for e in d.elements:
            counts[e.role] += 1
            suffix = f"_{counts[e.role]}" if counts[e.role] > 1 else ""
            elements.append(Element(id=f"{final_id}.{e.role}{suffix}", role=e.role, text=e.text,
                                    highlights=e.highlights, bbox=e.bbox))

        # Product crop: user override wins over the model's box; the end frame never has products.
        asset_id, box = None, overrides.get(final_id) or d.product_bbox
        if box and not d.is_end_frame:
            crop = crops.crop_box(frame, box, out / "crops" / f"{final_id}.png")
            # Reuse an existing asset if this crop looks the same (products often repeat across scenes).
            asset_id = next((a for a, p in assets.items() if crops.same_image(crop, Path(p))), None)
            if asset_id:
                crop.unlink()
            else:
                asset_id = f"products_{len(assets) + 1}"
                assets[asset_id] = str(crop)
                issues += crop_text_issues(final_id, crop, elements)
        issues += copy_check(final_id, frame, elements)

        seg = timing.get(sid)
        scenes.append(Scene(id=final_id, start_s=seg.start if seg else None, end_s=seg.end if seg else None,
                            frame_path=str(frame), elements=elements, product_bbox=box,
                            product_asset=asset_id, is_end_frame=d.is_end_frame))
    spec = MasterSpec(source=source, kind=kind, width=W, height=H, detected_brand=draft.detected_brand,
                      scenes=scenes, transition_s=transition, assets=assets, cache_key=key)
    return spec, issues


def copy_check(sid: str, frame: Path, elements: list[Element]) -> list[Issue]:
    """Compare the model-extracted copy with independent OCR of the frame; warn if they disagree (<90 %).

    Skipped (no issue) when OCR is unavailable or there is no extracted text.
    """
    text = ocr.ocr_text(frame)
    extracted = " ".join(e.text or "" for e in elements if e.role in TEXT_ROLES)
    if text is None or not extracted:
        return []
    score = ocr.similarity(extracted, text)
    if score >= 0.9:
        return []
    return [Issue(code="COPY_EXTRACTION_MISMATCH", severity="warn",
                  message=f"{sid}: OCR and the extracted copy agree only {score:.0%}",
                  fix="check work/master_spec.json; add copy.json if the text is wrong")]


def crop_text_issues(sid: str, crop: Path, elements: list[Element]) -> list[Issue]:
    """Warn if a product crop contains readable text that isn't the scene's legal text.

    That usually means the box is too wide and swallowed headline or body copy, which would then be duplicated
    inside the product image. Needs at least 3 unexplained words to trigger, to ignore packaging labels/noise.
    """
    text = ocr.ocr_text(crop)
    if not text:
        return []
    legal = " ".join(e.text or "" for e in elements if e.role == "legal").lower().split()
    extra = [w for w in re.findall(r"[A-Za-z]{3,}", text) if w.lower() not in legal]
    if len(extra) < 3:
        return []
    return [Issue(code="CROP_CONTAINS_TEXT", severity="warn",
                  message=f"{sid}: product crop contains text ({' '.join(extra[:6])})",
                  fix="tighten the box in boxes.json if this isn't part of the product")]
