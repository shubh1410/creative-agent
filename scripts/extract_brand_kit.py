"""Draft brand_kit.json from a brand PDF with a vision model. A person reviews the draft (plan §5.1).

Usage:
  uv run python scripts/extract_brand_kit.py PDF --brand maxis --pages 9 18 19 20 23 24 25 27 39
  uv run python scripts/extract_brand_kit.py PDF --brand maxis      # no --pages: the model picks the pages"""
import argparse
import asyncio
import base64
import io
from pathlib import Path

import pypdfium2 as pdfium
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from creative_agent.brand import BrandKit
from creative_agent.config import settings
from creative_agent.llm import ask_structured, make_llm

SYSTEM = """You convert brand-guideline pages into a brand kit matching the schema.
Copy values exactly as printed: hex codes, pixel sizes, percentages. Set source_page on every field you fill.
If pages conflict, choose one value and describe the conflict in `conflicts`.
Never invent values. For asset paths you can't know, write "TODO"."""

SELECT_SYSTEM = """You are shown low-resolution thumbnails of pages from a brand-guidelines PDF, each labelled with
its page number. Pick the pages that contain rules or values needed to build digital ad banners: colour palette
(hex codes), logo variants, sizes, margins and clear space, typography (fonts, weights, sizes), layout, CTA/button
style, motion and legal-text rules. Skip covers, tables of contents, photography, tone of voice and other pages
that carry no such values. Return page numbers exactly as labelled."""

BATCH = 25          # thumbnails per selection request, to keep each request small
THUMB_SCALE = 0.6   # enough to recognise page content, cheap in tokens


class PageSelection(BaseModel):
    """Model's answer when choosing pages: the relevant page numbers (1-based)."""

    pages: list[int]


class Draft(BaseModel):
    """What the model must return: the draft kit plus notes on places where the PDF pages disagree."""

    kit: BrandKit
    conflicts: list[str]


def page_part(pdf: pdfium.PdfDocument, page_no: int, scale: float = 2) -> list[dict]:
    """Render one PDF page (1-based) and return it as message parts: a text label and a PNG data URL."""
    buf = io.BytesIO()
    pdf[page_no - 1].render(scale=scale).to_pil().save(buf, "PNG")
    data = base64.b64encode(buf.getvalue()).decode()
    return [{"type": "text", "text": f"Page {page_no}:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}]


async def select_pages(pdf: pdfium.PdfDocument) -> list[int]:
    """Let the model choose the relevant pages: show thumbnails of every page, ``BATCH`` at a time, and
    collect the pages it picks. Returns sorted 1-based page numbers. A page the model was not shown in that
    batch is reported back to it for correction (see ``ask_structured``)."""
    llm, total, chosen = make_llm(settings.vision_model), len(pdf), set()
    for start in range(1, total + 1, BATCH):
        batch = range(start, min(start + BATCH, total + 1))
        content = [part for p in batch for part in page_part(pdf, p, THUMB_SCALE)]
        sel = await ask_structured(llm, PageSelection, [SystemMessage(SELECT_SYSTEM), HumanMessage(content=content)],
                                   meta={"node": "extract_brand_kit.select_pages"},
                                   validate=lambda s, b=batch: [f"page {p} was not shown" for p in s.pages if p not in b])
        chosen.update(sel.pages)
    return sorted(chosen)


async def main() -> None:
    """Send the chosen pages to the vision model and write the draft kit and conflict notes into the brand's
    folder. The files are named ``*.draft.*`` so nothing is used before a person has reviewed it.
    If ``--pages`` is left out, the model first picks the pages itself (see :func:`select_pages`)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--brand", required=True)
    ap.add_argument("--pages", type=int, nargs="+", help="1-based pages to read; omit to let the model choose")
    args = ap.parse_args()

    pdf = pdfium.PdfDocument(str(args.pdf))
    pages = args.pages or await select_pages(pdf)
    if not pages:
        raise SystemExit("the model found no relevant pages; pass --pages explicitly")
    print(f"Reading pages: {' '.join(map(str, pages))}")
    content = [part for p in pages for part in page_part(pdf, p)]
    draft = await ask_structured(make_llm(settings.vision_model), Draft,
                                 [SystemMessage(SYSTEM), HumanMessage(content=content)],
                                 meta={"node": "extract_brand_kit"})
    out = settings.brands_dir / args.brand
    out.mkdir(parents=True, exist_ok=True)
    (out / "brand_kit.draft.json").write_text(draft.kit.model_dump_json(indent=2))
    (out / "brand_kit_notes.draft.md").write_text("## Conflicts found\n\n" + "\n".join(f"- {c}" for c in draft.conflicts))
    print(f"Wrote {out}/brand_kit.draft.json. Review every field against the PDF, then rename it to brand_kit.json.")


if __name__ == "__main__":
    asyncio.run(main())
