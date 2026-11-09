"""Browser-driven QC: opens the finished banner in headless Chromium, captures each frame, runs the per-frame
checks from ``checks.py`` and adds whole-ad checks (timeline, file size, click-through, console, network)."""
import io
from pathlib import Path

from PIL import Image

from ..models import CheckResult
from ..net import url_status
from .checks import PER_FRAME
from .context import QCContext

# In-page script that measures every element (box, font, colour, ...); see measure.js.
MEASURE_JS = (Path(__file__).parent / "measure.js").read_text()
# Lists all running CSS animations with their timing, so we can verify total length and "plays once".
TIMELINE_JS = """() => document.getAnimations().map(a => { const t = a.effect.getComputedTiming();
  return {name: a.animationName || '', duration: t.duration, iterations: t.iterations, delay: t.delay || 0}; })"""
# Runs before the page's own scripts: replaces window.open with a recorder so a click's target URL can be read
# without actually navigating anywhere.
INIT_JS = "window.__opened=[];window.open=function(u){window.__opened.push(String(u));return null;};"


def merge_frames(results: list[CheckResult]) -> list[CheckResult]:
    """One result per check; per-frame messages are prefixed with their frame number."""
    merged: dict[str, CheckResult] = {}
    for r in results:
        if r.frame is None:
            merged[r.name] = r                     # whole-ad check: keep as is
            continue
        m = merged.setdefault(r.name, CheckResult(name=r.name, passed=True, fixable=r.fixable))
        m.passed = m.passed and r.passed           # fails if it failed on any frame
        if r.message:
            m.message = "; ".join(filter(None, [m.message, f"frame {r.frame + 1}: {r.message}"]))
    return list(merged.values())


def ad_checks(data: dict, ctx: QCContext, html_bytes: int, landing: int, frames_with_legal: int) -> list[CheckResult]:
    """Checks that concern the ad as a whole rather than a single frame.

    Args:
        data: Browser observations: ``animations``, ``opened`` (URLs passed to window.open), ``console``
            (errors) and ``external`` (blocked network requests).
        ctx: QC context (brand limits, expected click URL).
        html_bytes: Size of the final HTML file.
        landing: HTTP status of the click-through URL (0 = unreachable).
        frames_with_legal: Number of frames in which legal text was found.
    """
    motion, limit = ctx.brand.motion, ctx.brand.weight.max_kb * 1024
    anims = data["animations"]
    # End time of the latest-finishing animation = total playing time.
    total = max((a["delay"] + a["duration"] * (a["iterations"] or 1) for a in anims), default=0)
    wrong_plays = [a for a in anims if a["iterations"] != motion.plays]
    opened = data["opened"]
    if landing in (401, 403):
        # Many sites reject bots; that says nothing about whether real users can reach the page.
        landing_ok, landing_msg = True, f"warning: the landing page answers {landing} to automated checks"
    else:
        landing_ok, landing_msg = 200 <= landing < 400, f"landing page status {landing or 'no response'}"
    return [
        CheckResult(name="timeline", passed=total <= motion.max_duration_s * 1000 and not wrong_plays, fixable=False,
                    message=f"total {total / 1000:.1f}s" + (f"; {len(wrong_plays)} animations loop" if wrong_plays else "")),
        # fixable only while the size reducer still has room to shrink the file
        CheckResult(name="file_size", passed=html_bytes <= limit, fixable=not ctx.budget_floor_hit,
                    message=f"{html_bytes / 1024:.1f} KB of {ctx.brand.weight.max_kb} KB"),
        CheckResult(name="click_opens", passed=opened == [ctx.click_url], fixable=True,
                    message="" if opened == [ctx.click_url] else
                    f"click opened {opened or 'nothing'}; expected {ctx.click_url} (is something stopping clicks?)"),
        CheckResult(name="landing_url", passed=landing_ok, fixable=False, message=landing_msg),
        CheckResult(name="console_errors", passed=not data["console"], fixable=True, message="; ".join(data["console"][:3])),
        CheckResult(name="external_requests", passed=not data["external"], fixable=True,
                    message="; ".join(data["external"][:3])),
        CheckResult(name="legal_once", passed=frames_with_legal > 0, fixable=True,
                    message="" if frames_with_legal else "legal text never appears"),
    ]


async def run_qc(browser, page_sem, html_path: Path, ctx: QCContext, landing_cache: dict, out_dir: Path,
                 attempt: int) -> tuple[list[CheckResult], list[str]]:
    """Load the banner, step through every frame, and run all QC checks.

    Args:
        browser: Shared Playwright browser.
        page_sem: Semaphore bounding concurrently open pages across all jobs.
        html_path: The finalised banner HTML.
        ctx: QC context for this size.
        landing_cache: Per-job URL status cache (see ``net.url_status``).
        out_dir: Where screenshots are written.
        attempt: Build attempt number, used in screenshot file names.

    Returns:
        ``(checks, screenshot_paths)`` with one merged ``CheckResult`` per check name and one screenshot per
        frame (in frame order).
    """
    external: list[str] = []
    console: list[str] = []
    results: list[CheckResult] = []
    shots: list[str] = []
    frames_with_legal = 0
    async with page_sem:
        # device_scale_factor=2 gives crisper screenshots (the backup image is later resized down).
        context = await browser.new_context(viewport={"width": ctx.W, "height": ctx.H}, device_scale_factor=2)
        try:
            page = await context.new_page()

            async def route(r):
                """Banners must be self-contained: allow local/inline resources, record and block the rest."""
                if r.request.url.startswith(("file:", "data:")):
                    await r.continue_()
                else:
                    external.append(r.request.url)
                    await r.abort()

            await page.route("**/*", route)
            page.on("console", lambda m: console.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e: console.append(str(e)))
            await page.add_init_script(INIT_JS)
            await page.goto(html_path.resolve().as_uri())
            await page.evaluate("document.fonts.ready")          # wait for web fonts so measurements are accurate
            animations = await page.evaluate(TIMELINE_JS)       # read before show() stops the animations

            # Show each frame in turn (animations frozen), screenshot it and run the per-frame checks.
            for i in range(len(ctx.plan.frames)):
                await page.evaluate(f"window.__ad.show({i})")
                shot_path = out_dir / f"attempt_{attempt}_frame_{i + 1}.png"
                png = await page.screenshot(path=str(shot_path), clip={"x": 0, "y": 0, "width": ctx.W, "height": ctx.H})
                shots.append(str(shot_path))
                fm = await page.evaluate(MEASURE_JS)
                with Image.open(io.BytesIO(png)) as im:
                    # Back to CSS pixels so checks can map screenshot pixels to element boxes 1:1.
                    shot = im.convert("RGB").resize((ctx.W, ctx.H))
                fctx = ctx.at(i)
                results += [check(fm, fctx, shot) for check in PER_FRAME]
                frames_with_legal += any((e["role"] or e["roleCtx"]) == "legal" for e in fm["elements"])

            # Click test: click the centre of the first frame and see which URL the ad tried to open.
            await page.evaluate("window.__ad.show(0)")
            await page.mouse.click(ctx.W / 2, ctx.H / 2)
            opened = await page.evaluate("window.__opened")
        finally:
            await context.close()

    # Done outside the page semaphore: this is a network call, not browser work.
    landing = await url_status(ctx.click_url, landing_cache)
    data = {"animations": animations, "opened": opened, "console": console, "external": external}
    results += ad_checks(data, ctx, html_path.stat().st_size, landing, frames_with_legal)
    return merge_frames(results), shots
