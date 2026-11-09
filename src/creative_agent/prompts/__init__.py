"""Prompt assembly: loads the ``*.md`` prompt templates in this folder and fills them with job data to produce
LangChain message lists for the planner, the HTML builder and the single-prompt baseline."""
from collections import defaultdict
from pathlib import Path
from string import Template

from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image

from ..models import dims

HERE = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Read ``<name>.md`` from this folder. System prompts use ``$PLACEHOLDER`` for :class:`string.Template`."""
    return (HERE / f"{name}.md").read_text()


def size_tips(sizes: list[str]) -> dict[str, str]:
    """Per-size layout advice parsed from ``layout_tips.md``.

    The file is split on ``## <size>`` headings; sizes without a section map to an empty string.
    """
    tips, current = {}, None
    for line in load_prompt("layout_tips").splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            tips[current] = ""
        elif current:
            tips[current] += line + "\n"
    return {s: tips.get(s, "").strip() for s in sizes}


def scene_summary(spec, content) -> str:
    """Plain-text outline of the master's scenes and their elements, for inclusion in prompts.

    Args:
        spec: The :class:`~creative_agent.models.MasterSpec`.
        content: Iterable of elements (with final text) whose ids look like ``"<scene_id>.<name>"``.
    """
    # Group elements by the scene prefix of their id ("s2.headline" -> "s2").
    by_scene = defaultdict(list)
    for e in content:
        by_scene[e.id.split(".")[0]].append(e)
    lines = []
    for s in spec.scenes:
        timing = f"{s.end_s - s.start_s:.1f}s stable in the master" if s.start_s is not None else "no timing"
        end = " (end frame: logo only)" if s.is_end_frame else ""
        lines.append(f"- {s.id}{end}; {timing}; product image: {'yes' if s.product_asset else 'no'}")
        for e in by_scene.get(s.id, []):
            bold = f" (bold: {', '.join(e.highlights)})" if e.highlights else ""
            lines.append(f'    - {e.id} [{e.role}] "{e.text}"{bold}')
    return "\n".join(lines)


def planner_messages(req, spec, content, notes, brand) -> list:
    """Messages asking the model to plan the frames of every requested size.

    Args:
        req: The job request (brief and sizes).
        spec: Master spec describing the scenes.
        content: Elements with their final copy.
        notes: Layout notes extracted from reference creatives (may be empty).
        brand: The brand kit (supplies the maximum animation length).
    """
    # OFFER = id of the first scene, which holds the offer; MAX_S = animation length cap.
    system = Template(load_prompt("planner_system")).substitute(OFFER=spec.scenes[0].id,
                                                                 MAX_S=brand.motion.max_duration_s)
    refs = "\n".join(f"- {n.aspect}: {n.notes}" for n in notes) or "- none"
    tips = "\n\n".join(f"## {s}\n{t}" for s, t in size_tips(list(req.sizes)).items())
    user = (f"Brief:\n{req.brief}\n\nScenes:\n{scene_summary(spec, content)}\n\nReference layouts:\n{refs}\n\n"
            f"Sizes to plan: {', '.join(req.sizes)}\n\nSize guidance:\n{tips}")
    return [SystemMessage(system), HumanMessage(user)]


def builder_messages(state, previous: str | None, failures: list[str]) -> list:
    """Messages asking the model to write the HTML for one size.

    Args:
        state: Per-size graph state (size, plan, brand, content, assets, layout note, ...).
        previous: HTML from the last attempt, if this is a retry.
        failures: QC failure descriptions to fix; only used together with ``previous``.
    """
    from ..render.finalize import corner_logo_box          # local import avoids a cycle
    size, plan, brand, content = state["size"], state["size_plan"], state["brand"], state["content"]
    W, H = dims(size)
    # Tell the model exactly which rectangle the logo (added later by code) will occupy so it keeps clear of it.
    lx, ly, lw, lh, _ = corner_logo_box(brand, size)
    w, t, z = brand.fonts.weights_by_role, brand.typography, brand.logo.adinfo_zone
    system = Template(load_prompt("builder_system")).substitute(
        W=W, H=H, BRAND=brand.brand_name, LOGO_X=lx, LOGO_X2=lx + lw, LOGO_Y=ly, LOGO_Y2=ly + lh,
        ZONE_W=z.width_px, ZONE_H=z.height_px,
        COLOR_VARS=", ".join(f"var(--brand-{n.replace('_', '-')})" for n in brand.colors),
        TEXT_COLORS=", ".join(c.replace("_", " ") for c in brand.text_colors),
        W_HEAD=w.get("headline", 600), W_CTA=w.get("cta", 600), W_HIGH=w.get("highlight", 700), W_BODY=w.get("body", 400),
        MAX_SIZES=t.max_font_sizes_per_frame, MAX_COLORS=t.max_text_colors_per_frame, HEAD_LINES=t.headline_max_lines,
        MIN_FONT=t.min_font_px, MIN_LEGAL=t.min_legal_font_px,
        VISUAL=brand.rules.horizontal_grid["visual_pct"], COPY=brand.rules.horizontal_grid["copy_pct"])

    # Describe each planned frame and the elements it must contain.
    lines = []
    for i, f in enumerate(plan.frames, 1):
        asset = state.get("scene_assets", {}).get(f.scene_id) if f.show_products else None
        lines.append(f'{i}. <div class="frame" data-scene="{f.scene_id}"> ({f.duration_ms} ms; product image: '
                     f'{"asset://" + asset if asset else "none"})')
        if f.scene_id == "end":
            lines.append("   - background only; code adds the centred logo")
        for eid in f.keep:
            e = content[eid]
            bold = f" (bold: {', '.join(e.highlights)})" if e.highlights else ""
            lines.append(f'   - data-el="{eid}" data-role="{e.role}": "{e.text}"{bold}')
    # List product images with their pixel sizes so the model can pick sensible aspect-correct boxes.
    assets = []
    for aid, p in state["assets"].items():
        with Image.open(p) as im:
            assets.append(f"- asset://{aid}: {im.width}x{im.height}px (aspect {im.width / im.height:.2f})")
    user = (f"Size: {size}\nSize guidance: {size_tips([size])[size]}\nReference layout: {state['layout_note']}\n\n"
            "Frames, in order:\n" + "\n".join(lines) + "\n\nProduct images:\n" + ("\n".join(assets) or "- none"))
    if previous:
        # Retry: show the last HTML and ask for minimal, targeted fixes to avoid regressions.
        user += (f"\n\n<previous_html>\n{previous}\n</previous_html>\n\n"
                 "Fix these QC failures and change nothing else:\n" + "\n".join(f"- {x}" for x in failures))
    return [SystemMessage(system), HumanMessage(user)]


def baseline_messages(req, spec, content, brand) -> list:
    """Messages for the single-prompt baseline: one call that must produce HTML for all sizes at once.

    The whole brand kit is passed as JSON because, unlike the agent, the baseline gets no per-size help.
    """
    system = Template(load_prompt("baseline_system")).substitute(SIZES=", ".join(req.sizes), BRAND=brand.brand_name)
    assets = "\n".join(f"- asset://{s.product_asset} shows the products of {s.id}" for s in spec.scenes if s.product_asset)
    cta = next((e.id for e in content if e.role == "cta"), "none")
    user = (f"Brief:\n{req.brief}\n\nBrand kit (JSON):\n{brand.model_dump_json(indent=1)}\n\n"
            f"Scenes:\n{scene_summary(spec, content)}\n\nProduct images:\n{assets or '- none'}\n\nCTA element: {cta}")
    return [SystemMessage(system), HumanMessage(user)]
