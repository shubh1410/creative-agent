"""Input validation: deterministic checks and fix-ups on the ingested master before any planning or building.

Anything unusable stops the job with a :class:`JobInputError`; recoverable oddities become :class:`Issue` entries
that are reported back to the user in the job result.
"""
import json
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from PIL import Image

from ...brand import load_brand
from ...errors import JobInputError
from ...ingest.crops import sample_text_color
from ...models import TEXT_ROLES, Issue, dims
from ...net import url_status
from ...trace import record_issues, traced
from ..state import JobContext, JobState
from .common import load_spec, norm_text


def largest_image_width(sizes: list[str]) -> int:
    """Rough widest placement of a product image: 60% of wide sizes, full width otherwise."""
    return max(round(w * 0.6) if w / h >= 2 else w for w, h in map(dims, sizes))


@traced("check_inputs")
async def check_inputs_node(state: JobState, runtime: Runtime[JobContext], config: RunnableConfig):
    """Run all input checks (numbered below), then store ``work/content.json``.

    Returns:
        ``brand`` (loaded kit), ``content`` (non-logo elements, with copy overrides and text colours applied) and
        ``issues`` found.

    Raises:
        JobInputError: For problems that make the job impossible (brand mismatch, no CTA, impossible logo, dead URL, ...).
    """
    ctx, req = runtime.context, state["request"]
    ctx.set_stage("checking inputs")
    brand = load_brand(req.brand_id)
    spec = load_spec(state)
    issues: list[Issue] = []

    # 1. Brand match: stop before any planning or building.
    if not brand.matches_brand(spec.detected_brand):
        raise JobInputError("BRAND_MISMATCH", f"the master shows '{spec.detected_brand or 'no recognisable brand'}',"
                                              f" but the brand kit is '{brand.brand_name}'")

    # Logos are drawn by code from the brand kit, so only copy/product elements are carried forward.
    content = [e for s in spec.scenes for e in s.elements if e.role != "logo"]
    by_id = {e.id: e for e in content}
    frame_of = {e.id: Path(s.frame_path) for s in spec.scenes for e in s.elements}

    # 2. Copy override (copy.json).
    override = ctx.job_dir / "inputs" / "copy.json"
    if override.exists():
        for eid, text in json.loads(override.read_text()).items():
            if eid not in by_id:
                raise JobInputError("COPY_OVERRIDE_UNKNOWN_ID", f"copy.json names '{eid}', which isn't in master_spec.json")
            issues.append(Issue(code="COPY_OVERRIDDEN", severity="info",
                                message=f"{eid}: '{by_id[eid].text}' → '{text}'", fix="replaced from copy.json"))
            by_id[eid].text = text
            # Drop highlights that no longer occur in the new text.
            by_id[eid].highlights = [h for h in by_id[eid].highlights if h in text]

    # 3. Text colours: sampled, then snapped to the brand's text palette. Never used as brand colours.
    palette = brand.text_palette()
    for e in content:
        if e.role in TEXT_ROLES:
            e.text_color, dist = sample_text_color(frame_of[e.id], e.bbox, palette)
            if dist > brand.color_tolerance * 3:
                issues.append(Issue(code="COLOR_SNAPPED", severity="info",
                                    message=f"{e.id}: sampled colour is {dist:.0f} away from {e.text_color}",
                                    fix=f"using {e.text_color}"))

    # 4. CTA: reuse existing copy, never invent it.
    if not any(e.role == "cta" for e in content):
        if not req.cta:
            raise JobInputError("NO_CTA", "the master has no CTA and job.json sets no cta rule")
        match = next((e for e in content if norm_text(e.text) == norm_text(req.cta.use_text)), None)
        if not match:
            raise JobInputError("CTA_NOT_FOUND", f"no text in the master matches '{req.cta.use_text}'")
        match.role = "cta"
        issues.append(Issue(code="CTA_ADDED", severity="info",
                            message=f"the master has no CTA; {match.id} '{match.text}' becomes the CTA",
                            fix="styled as a CTA button; copy unchanged"))

    # 5. Legal text and footnote markers.
    for s in spec.scenes:
        legal = [e for e in s.elements if e.role == "legal"]
        if s.product_asset and not legal:
            issues.append(Issue(code="LEGAL_MISSING_IN_MASTER", severity="warn",
                                message=f"{s.id} shows products but has no legal text"))
        # A "^" or "*" in body copy must be explained by a legal line starting with the same marker.
        for e in s.elements:
            for marker in ("^", "*"):
                if e.role != "legal" and marker in (e.text or "") and \
                        not any((l.text or "").lstrip().startswith(marker) for l in legal):
                    issues.append(Issue(code="DISCLAIMER_MISSING", severity="high",
                                        message=f"{e.id} uses '{marker}' but {s.id} has no matching disclaimer"))

    # 6. Assets.
    if not brand.fonts.brand_files_available:
        issues.append(Issue(code="BRAND_FONT_SUBSTITUTED", severity="warn",
                            message=f"{brand.fonts.brand_family} font files aren't available",
                            fix=f"using {brand.fonts.active_family()}"))
    if brand.asset_provenance.get("logos") == "crop":
        issues.append(Issue(code="LOGO_FROM_CROP", severity="info",
                            message="logos are cropped from the inputs, not official files",
                            fix="provenance in brand_kit_notes.md"))
    need = largest_image_width(req.sizes) * 2                   # 2x for sharp screens
    for aid, path in spec.assets.items():
        with Image.open(path) as im:
            # Tolerate up to 1.5x upscaling before calling the image too small.
            if need / im.width > 1.5:
                issues.append(Issue(code="LOW_RESOLUTION", severity="warn",
                                    message=f"{aid} is {im.width}px wide; the largest placement needs about {need}px"))

    # 7. Logo feasibility per size.
    for size in req.sizes:
        W, H = dims(size)
        place = brand.logo.corner_for(size)
        if size in brand.logo.overrides:
            issues.append(Issue(code="LOGO_VARIANT_SWITCHED", severity="info",
                                message=f"{size}: {place.reason}", fix=f"{place.variant} logo, {place.height_px}px"))
        elif place.margin_top_px + place.height_px > H:
            raise JobInputError("LOGO_SPEC_IMPOSSIBLE", f"{size}: the corner logo doesn't fit and the kit has no override")
        if size not in brand.logo.spec_sizes:
            issues.append(Issue(code="LOGO_SPEC_EXTRAPOLATED", severity="info",
                                message=f"{size} isn't in the brand's display spec list", fix="same corner rule applied"))

    # 8. Master deviations (information only; banners follow the kit).
    end = next((s for s in spec.scenes if s.is_end_frame), None)
    logo = next((e for e in end.elements if e.role == "logo"), None) if end else None
    if logo:
        pct = 100 * logo.bbox.w / spec.width
        rule = brand.logo.end_frame
        want = rule.vertical_width_pct if spec.height > spec.width else rule.horizontal_width_pct
        if abs(pct - want) > 3:
            issues.append(Issue(code="MASTER_RULE_DEVIATION", severity="info",
                                message=f"the master's end-frame logo is {pct:.0f}% of the width; the kit says {want}%",
                                fix="banners follow the kit"))

    # 9. Click URL.
    status = await url_status(str(req.click_url), ctx.landing_cache)
    if status in (401, 403):
        issues.append(Issue(code="CLICK_URL_BLOCKS_BOTS", severity="warn",
                            message=f"{req.click_url} answers {status} to automated requests", fix="treated as reachable"))
    elif not 200 <= status < 400:
        raise JobInputError("CLICK_URL_UNREACHABLE", f"{req.click_url} returned {status or 'no response'}")

    work = ctx.job_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "content.json").write_text(json.dumps([e.model_dump() for e in content], indent=2))
    await record_issues(ctx, issues)
    return {"brand": brand, "content": content, "issues": issues}
