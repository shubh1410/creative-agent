"""Planning step: the LLM decides, per size, which scenes/elements appear in which frame and for how long."""
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from ...llm import ask_structured
from ...models import MasterSpec, Plan
from ...prompts import planner_messages
from ...trace import traced
from ..state import JobContext, JobState
from .common import load_spec


def validate_plan(plan: Plan, spec: MasterSpec, content, brand, sizes: list[str]) -> list[str]:
    """Check a model-produced plan against the hard rules that the schema can't express.

    Args:
        plan: The plan to check.
        spec: Master spec (valid scene ids and product assets).
        content: All elements (valid element ids, roles).
        brand: Brand kit (maximum animation length).
        sizes: Sizes the job requested; the plan must cover exactly these.

    Returns:
        A list of human-readable problems; empty means the plan is valid. The list is fed back to the model
        by ``ask_structured`` for a repair attempt.
    """
    errors: list[str] = []
    by_id = {e.id: e for e in content}
    scene_ids = {s.id for s in spec.scenes}
    cta_ids = {e.id for e in content if e.role == "cta"}
    legal_ids = {e.id for e in content if e.role == "legal"}
    offer = spec.scenes[0].id                    # the first scene carries the offer and must never be dropped
    planned = [p.size for p in plan.sizes]
    if sorted(planned) != sorted(sizes):
        errors.append(f"the plan must cover exactly {sorted(sizes)}, got {sorted(planned)}")

    for sp in plan.sizes:
        tag = sp.size
        ids = [f.scene_id for f in sp.frames]
        if not ids:
            errors.append(f"{tag}: no frames")
            continue
        if offer not in ids:
            errors.append(f"{tag}: the offer scene {offer} must be included")
        if "end" in scene_ids and ids[-1] != "end":
            errors.append(f"{tag}: the end frame must be last")
        # Index of the last frame that carries a message (the end frame, if any, only shows the logo).
        last_message = len(ids) - (2 if ids[-1] == "end" else 1)
        if [i for i, f in enumerate(sp.frames) if set(f.keep) & cta_ids] != [last_message]:
            errors.append(f"{tag}: the CTA must appear once, on the last frame before the end frame")
        for i, f in enumerate(sp.frames):
            if f.scene_id not in scene_ids:
                errors.append(f"{tag}: unknown scene '{f.scene_id}'")
                continue
            for eid in f.keep:
                if eid not in by_id:
                    errors.append(f"{tag}: unknown element '{eid}'")
                elif eid.split(".")[0] != f.scene_id:
                    errors.append(f"{tag}: '{eid}' doesn't belong to scene {f.scene_id}")
            if f.show_products and not spec.scene(f.scene_id).product_asset:
                errors.append(f"{tag}: scene {f.scene_id} has no product image")
            # Frames with product images must carry the disclaimer text that goes with them.
            if f.show_products and not set(f.keep) & legal_ids:
                errors.append(f"{tag}: frame {i} shows products, so it must keep the legal text")
            if f.duration_ms < 1000:
                errors.append(f"{tag}: frame {i} is shorter than 1000 ms")
        if not any(set(f.keep) & legal_ids for f in sp.frames):
            errors.append(f"{tag}: legal text must appear at least once")
        total = sum(f.duration_ms for f in sp.frames)
        if total > brand.motion.max_duration_s * 1000:
            errors.append(f"{tag}: total duration {total} ms exceeds {brand.motion.max_duration_s} s")
        # The static backup must show real content, not just the closing logo frame.
        if not 0 <= sp.backup_frame < len(ids) or ids[sp.backup_frame] == "end":
            errors.append(f"{tag}: backup_frame must point to a frame other than the end frame")
    return errors


@traced("plan")
async def plan_node(state: JobState, runtime: Runtime[JobContext], config: RunnableConfig):
    """Ask the model for a :class:`Plan` (validated, with one repair attempt) and save it to ``work/plan.json``."""
    ctx, req = runtime.context, state["request"]
    ctx.set_stage("planning")
    spec, brand, content = load_spec(state), state["brand"], state["content"]
    plan = await ask_structured(
        ctx.llm, Plan, planner_messages(req, spec, content, state.get("layout_notes", []), brand),
        meta={"node": "plan", "model": ctx.model_name}, config=config,
        validate=lambda p: validate_plan(p, spec, content, brand, list(req.sizes)))
    work = ctx.job_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "plan.json").write_text(plan.model_dump_json(indent=2))
    return {"plan": plan}
