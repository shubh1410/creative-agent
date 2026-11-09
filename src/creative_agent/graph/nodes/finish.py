"""Final step of a size's loop: package the deliverables (HTML, zip, screenshots, static backup image)."""
import shutil
import zipfile
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from PIL import Image

from ...models import SizeResult, dims
from ...trace import traced
from ..state import JobContext, SizeState


def make_backup(shot: Path, out: Path, wh: tuple[int, int], max_bytes: int) -> Path:
    """Create the static fallback JPEG shown when HTML5 can't play.

    Args:
        shot: Screenshot to convert (taken at device-pixel-ratio > 1, so it is resized down to the ad size).
        out: Destination ``.jpg`` path.
        wh: Target ``(width, height)`` in pixels.
        max_bytes: File-size budget; JPEG quality is lowered step by step until the file fits.

    Returns:
        ``out``. If even the lowest quality is over budget, the lowest-quality file is kept.
    """
    with Image.open(shot) as im:
        img = im.convert("RGB").resize(wh, Image.LANCZOS)
    for q in (90, 82, 74, 66, 58, 50):
        img.save(out, "JPEG", quality=q, optimize=True)
        if out.stat().st_size <= max_bytes:
            break
    return out


@traced("finish")
async def finish_node(state: SizeState, runtime: Runtime[JobContext], config: RunnableConfig):
    """Write ``sizes/<size>/`` outputs and return the size's :class:`SizeResult`.

    A size is "approved" only if its last QC report passed; failed sizes are still packaged so the user can
    inspect the best attempt.
    """
    ctx, size, qc = runtime.context, state["size"], state.get("qc")
    d = ctx.job_dir / "sizes" / size
    d.mkdir(parents=True, exist_ok=True)
    status = "approved" if qc and qc.passed else "failed"
    ctx.set_stage(f"{size} {status}", size=size, size_status=status)

    # Promote the last attempt to index.html and also ship it as a zip (the format ad servers upload).
    final = d / "index.html"
    shutil.copyfile(state["html_path"], final)
    with zipfile.ZipFile(d / f"{size}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(final, "index.html")
    # Keep the QC screenshots of the final attempt under stable names.
    shots = []
    for k, s in enumerate(qc.screenshots if qc else [], 1):
        shots.append(str(shutil.copyfile(s, d / f"final_frame_{k}.png")))
    backup = None
    if shots:
        # Clamp in case the plan's backup_frame is beyond the screenshots actually captured.
        idx = min(state["size_plan"].backup_frame, len(shots) - 1)
        backup = make_backup(Path(shots[idx]), d / "backup.jpg", dims(size), state["brand"].weight.max_kb * 1024)

    failed = sorted({c.name for c in qc.checks if not c.passed}) if qc else ["no_qc"]
    return {"result": SizeResult(size=size, status=status, attempts=state["attempt"], html_path=str(final),
                                 backup_path=str(backup) if backup else None, screenshots=shots,
                                 failed_checks=failed, bytes=final.stat().st_size)}
