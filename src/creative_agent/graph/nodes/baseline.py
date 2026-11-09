"""Nodes of the single-prompt baseline graph: one LLM call writes all sizes, then QC runs once per size.

This deliberately has no planner and no repair loop; it exists to compare against the full agent.
"""
import asyncio

from bs4 import BeautifulSoup
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from pydantic import BaseModel

from ...llm import ask_structured
from ...models import FramePlan, MasterSpec, Size, SizePlan
from ...prompts import baseline_messages
from ...render.finalize import FinalizeInput, extract_html, finalize_html
from ...trace import traced
from ..state import JobContext, JobState, SizeState
from .common import load_spec
from .finish import finish_node
from .qc import qc_node


class SizedHtml(BaseModel):
    """The model's HTML for one size."""

    size: Size
    html: str


class AllSizes(BaseModel):
    """Structured answer of the baseline call: exactly one ad per requested size."""

    ads: list[SizedHtml]


def plan_from_html(size: str, html: str, content: dict, spec: MasterSpec) -> SizePlan:
    """Reconstruct a :class:`SizePlan` from HTML the model wrote without a plan.

    Downstream code (finalisation, QC) expects a plan, so it is inferred from the markup: each direct
    ``div.frame`` of ``#ad`` is a frame, ``data-el`` attributes list its elements and ``data-duration`` its
    length. Falls back to a single empty frame if the structure is missing.
    """
    ad = BeautifulSoup(extract_html(html), "html.parser").find(id="ad")
    frames = []
    for f in (ad.find_all("div", class_="frame", recursive=False) if ad else []):
        keep = [t["data-el"] for t in f.find_all(attrs={"data-el": True}) if t["data-el"] in content]
        frames.append(FramePlan(scene_id=f.get("data-scene", ""), keep=keep, show_products=f.find("img") is not None,
                                duration_ms=max(1000, int(f.get("data-duration", 3000)))))
    if not frames:
        frames = [FramePlan(scene_id=spec.scenes[0].id, keep=[], show_products=False, duration_ms=3000)]
    # Use the first non-end frame as the static backup.
    backup = next((i for i, f in enumerate(frames) if f.scene_id != "end"), 0)
    return SizePlan(size=size, frames=frames, backup_frame=backup, notes="read back from the single-prompt HTML")


@traced("single_prompt_build")
async def single_prompt_build_node(state: JobState, runtime: Runtime[JobContext], config: RunnableConfig):
    """One LLM call produces HTML for every requested size; each is then finalised like in the agent path.

    Returns ``baseline_ads``: per size, the inferred plan, finalised HTML path and contract errors.
    """
    ctx, req = runtime.context, state["request"]
    ctx.set_stage("building all sizes in one prompt")
    spec, content = load_spec(state), {e.id: e for e in state["content"]}
    want = sorted(req.sizes)
    out = await ask_structured(ctx.llm, AllSizes, baseline_messages(req, spec, state["content"], state["brand"]),
                               meta={"node": "single_prompt_build", "model": ctx.model_name}, config=config,
                               validate=lambda a: [] if sorted(x.size for x in a.ads) == want
                               else [f"return exactly one ad for each of {want}"])
    ads = {}
    for ad in out.ads:
        d = ctx.job_dir / "sizes" / ad.size
        d.mkdir(parents=True, exist_ok=True)
        plan = plan_from_html(ad.size, ad.html, content, spec)
        res = await finalize_html(FinalizeInput(model_html=ad.html, size=ad.size, brand=state["brand"], plan=plan,
                                                content=content, assets=dict(spec.assets),
                                                click_url=str(req.click_url), work_dir=d),
                                  ctx.browser, ctx.page_sem)
        (d / "attempt_1.html").write_text(res.html)
        ads[ad.size] = {"size_plan": plan, "html_path": str(d / "attempt_1.html"),
                        "contract_errors": res.contract_errors, "budget_floor_hit": res.budget_floor_hit}
    return {"baseline_ads": ads}


@traced("qc_all")
async def qc_all_node(state: JobState, runtime: Runtime[JobContext], config: RunnableConfig):
    """Run QC and packaging for all baseline ads concurrently, reusing the agent's ``qc_node``/``finish_node``
    by assembling a ``SizeState`` for each size. No retries: attempt is fixed at 1."""
    spec, content = load_spec(state), {e.id: e for e in state["content"]}

    async def one(size: str, ad: dict):
        s: SizeState = {"job_id": state["job_id"], "size": size, "size_plan": ad["size_plan"], "brand": state["brand"],
                        "content": content, "assets": dict(spec.assets), "click_url": str(state["request"].click_url),
                        "attempt": 1, "html_path": ad["html_path"], "contract_errors": ad["contract_errors"],
                        "budget_floor_hit": ad["budget_floor_hit"]}
        s.update(await qc_node(s, runtime, config))      # adds the "qc" report that finish_node reads
        return size, (await finish_node(s, runtime, config))["result"]

    pairs = await asyncio.gather(*(one(size, ad) for size, ad in state["baseline_ads"].items()))
    return {"results": dict(pairs)}
