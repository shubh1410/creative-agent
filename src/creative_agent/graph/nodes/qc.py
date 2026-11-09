"""QC step of a size's loop: runs all checks on the built HTML and decides whether to retry."""
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from ...config import settings
from ...models import CheckResult, QCReport
from ...qc.browser import QCContext, run_qc
from ...trace import traced
from ..state import JobContext, SizeState


@traced("qc")
async def qc_node(state: SizeState, runtime: Runtime[JobContext], config: RunnableConfig):
    """Run the code-contract check plus all browser-based QC checks and save the report.

    Writes ``sizes/<size>/qc_attempt_<n>.json`` and emits one ``qc_check`` trace event per check.
    """
    ctx, size, attempt = runtime.context, state["size"], state["attempt"]
    ctx.set_stage(f"checking {size} (attempt {attempt})", size=size, size_status="checking")
    d = ctx.job_dir / "sizes" / size
    d.mkdir(parents=True, exist_ok=True)
    # "contract" = structural errors found while finalising the model's HTML (always fixable by the model).
    errors = state.get("contract_errors", [])
    checks = [CheckResult(name="contract", passed=not errors, fixable=True, message="; ".join(errors[:6]))]
    shots: list[str] = []
    if Path(state["html_path"]).stat().st_size:          # empty when the model returned no #ad
        more, shots = await run_qc(ctx.browser, ctx.page_sem, Path(state["html_path"]), QCContext.from_state(state),
                                   ctx.landing_cache, d, attempt)
        checks += more
    report = QCReport(size=size, attempt=attempt, passed=all(c.passed for c in checks), checks=checks, screenshots=shots)
    (d / f"qc_attempt_{attempt}.json").write_text(report.model_dump_json(indent=2))
    for c in checks:
        await ctx.tracer.emit("qc_check", size=size, attempt=attempt, **c.model_dump())
    return {"qc": report}


def route_after_qc(state: SizeState) -> str:
    """Conditional edge after QC: ``"finish"`` or ``"build"`` (retry).

    Retry only if something failed that the model can fix and attempts remain; unfixable failures (e.g. an
    unreachable click URL) or exhausted attempts go straight to finish.
    """
    qc = state["qc"]
    if qc.passed:
        return "finish"
    fixable = any(c.fixable for c in qc.checks if not c.passed)
    return "build" if fixable and state["attempt"] < settings.max_attempts else "finish"
