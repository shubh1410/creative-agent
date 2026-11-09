"""Fan-in step: gathers the per-size results, builds the contact sheet and writes ``result.json``."""
import asyncio
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from PIL import Image, ImageDraw

from ...models import JobResult, SizeResult
from ...trace import traced
from ..state import JobContext, JobState


def contact_sheet(results: list[SizeResult], out: Path) -> str | None:
    """Lay the backup images of all sizes side by side, each labelled with its size, and save as one PNG.

    Returns:
        The output file name, or ``None`` if no size has a backup image.
    """
    imgs = [(r.size, Image.open(r.backup_path)) for r in results if r.backup_path]
    if not imgs:
        return None
    pad, label = 24, 28                         # outer/inner spacing and height reserved for the caption
    W = sum(im.width for _, im in imgs) + pad * (len(imgs) + 1)
    H = max(im.height for _, im in imgs) + pad * 2 + label
    sheet = Image.new("RGB", (W, H), "white")
    draw, x = ImageDraw.Draw(sheet), pad
    for size, im in imgs:
        draw.text((x, pad), size, fill="black")
        sheet.paste(im, (x, pad + label))
        x += im.width + pad
    sheet.save(out)
    return out.name


@traced("collect")
async def collect_node(state: JobState, runtime: Runtime[JobContext], config: RunnableConfig):
    """Compute the overall job status and write the final :class:`JobResult`.

    Status: ``completed`` if every size was approved, ``completed_with_failures`` if at least one was,
    otherwise ``failed``.
    """
    ctx = runtime.context
    # Keep the order the caller requested rather than the order workers finished in.
    results = [state["results"][s] for s in state["request"].sizes if s in state["results"]]
    approved = sum(r.status == "approved" for r in results)
    status = "completed" if results and approved == len(results) else ("completed_with_failures" if approved else "failed")
    # Image work is blocking, so run it off the event loop.
    sheet = await asyncio.to_thread(contact_sheet, results, ctx.job_dir / "contact_sheet.png")
    result = JobResult(job_id=state["job_id"], status=status, sizes=results, issues=state.get("issues", []),
                       usage=ctx.tracer.usage(), contact_sheet=sheet)
    (ctx.job_dir / "result.json").write_text(result.model_dump_json(indent=2))
    return {"job_status": status}
