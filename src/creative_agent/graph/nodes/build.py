"""Build step of a size's loop: asks the model for HTML, then lets code finalise it."""
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from ...llm import ask_text
from ...prompts import builder_messages
from ...render.finalize import FinalizeInput, extract_html, finalize_html
from ...trace import traced
from ..state import JobContext, SizeState


def failure_lines(state: SizeState) -> list[str]:
    """Describe the failed, fixable QC checks of the last attempt as one line each (``name (frame N): message``).

    These are fed back to the model on retry. Unfixable failures are omitted since the model can't act on them.
    """
    qc = state.get("qc")
    if not qc:
        return []
    return [f"{c.name}" + (f" (frame {c.frame + 1})" if c.frame is not None else "") + f": {c.message}"
            for c in qc.checks if not c.passed and c.fixable]


@traced("build")
async def build_node(state: SizeState, runtime: Runtime[JobContext], config: RunnableConfig):
    """One build attempt.

    Pipeline: LLM writes the creative HTML -> ``finalize_html`` injects the brand-controlled parts (fonts,
    logo, images, click-through, size budget) -> both versions are saved. The model's own output
    (``attempt_N.model.html``) is kept separate from the finalised one (``attempt_N.html``) so retries can show
    the model its original HTML rather than one bloated with inlined assets.
    """
    ctx, size = runtime.context, state["size"]
    attempt = state.get("attempt", 0) + 1
    ctx.set_stage(f"building {size} (attempt {attempt})", size=size, size_status="building")
    d = ctx.job_dir / "sizes" / size
    d.mkdir(parents=True, exist_ok=True)

    # On a retry the model sees its own HTML (no inlined base64) plus the fixable QC failures.
    previous = Path(state["model_html_path"]).read_text() if attempt > 1 and state.get("model_html_path") else None
    raw = await ask_text(ctx.llm, builder_messages(state, previous, failure_lines(state) if previous else []),
                         meta={"node": "build", "model": ctx.model_name, "size": size, "attempt": attempt},
                         config=config)
    model_html = extract_html(raw)
    model_path = d / f"attempt_{attempt}.model.html"
    model_path.write_text(model_html)

    res = await finalize_html(FinalizeInput(model_html=model_html, size=size, brand=state["brand"],
                                            plan=state["size_plan"], content=state["content"], assets=state["assets"],
                                            click_url=state["click_url"], work_dir=d),
                              ctx.browser, ctx.page_sem)
    html_path = d / f"attempt_{attempt}.html"
    html_path.write_text(res.html)
    return {"attempt": attempt, "model_html_path": str(model_path), "html_path": str(html_path),
            "contract_errors": res.contract_errors, "budget_floor_hit": res.budget_floor_hit}
