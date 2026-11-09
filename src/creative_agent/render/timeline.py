"""Animation timeline: the JavaScript runtime and CSS keyframes that make the banner's frames play in sequence.

Everything is plain CSS animation (no libraries) to keep the file small and let QC read timings from the browser.
"""
import itertools

# Injected into every banner. Exposes ``window.__ad`` so QC can jump to a frame (``show(i)``) and take a
# screenshot with animations frozen. Also wires the click-through. ``__HAS_END__`` is replaced with true/false
# depending on whether the last frame is the logo end frame.
RUNTIME_JS = """(function(){
var ad=document.getElementById('ad'),frames=ad.querySelectorAll(':scope>.frame'),hasEnd=__HAS_END__;
window.__ad={n:frames.length,show:function(i){
  frames.forEach(function(f,j){f.style.animation='none';f.style.opacity='1';f.style.display=j===i?'block':'none';});
  var c=ad.querySelector('.logo-corner');
  if(c){c.style.animation='none';c.style.opacity='1';c.style.display=(hasEnd&&i===frames.length-1)?'none':'block';}
}};
ad.addEventListener('click',function(){window.open(window.clickTag);});
})();"""


def timeline_css(durations_ms: list[int], fade_ms: int = 500, hide_corner_on_last: bool = True) -> str:
    """Crossfades: around each boundary b, frame i fades out while frame i+1 fades in over fade_ms.
    Plays once and holds the last frame. The corner logo fades out as the end frame arrives.

    Args:
        durations_ms: How long each frame is shown.
        fade_ms: Length of each cross-fade, centred on the boundary between two frames.
        hide_corner_on_last: Fade out the corner logo when the final (end) frame appears, since that frame has
            its own large centred logo.

    Returns:
        CSS text with one ``@keyframes`` + rule pair per frame. Empty for a single frame (nothing to animate).
        Every frame uses the same total duration ``T`` and ``linear`` timing, with keyframe percentages
        positioning each fade, so one ``animation`` per frame covers the whole timeline.
    """
    if len(durations_ms) < 2:
        return ""
    T = sum(durations_ms)
    bounds = list(itertools.accumulate(durations_ms))[:-1]     # time of each boundary between frames
    h, last = fade_ms / 2, len(durations_ms) - 1

    def pct(ms: float) -> str:
        """Time in ms -> keyframe percentage of the total, clamped to 0-100%."""
        return f"{max(0.0, min(100.0, 100 * ms / T)):.3f}%"

    rules = []
    for i in range(last + 1):
        # Fade-in part: the first frame starts visible; others stay hidden until their incoming boundary.
        kf = ["0%{opacity:1}"] if i == 0 else \
            ["0%{opacity:0}", f"{pct(bounds[i-1] - h)}{{opacity:0}}", f"{pct(bounds[i-1] + h)}{{opacity:1}}"]
        # Fade-out part: the last frame stays visible; others fade at their outgoing boundary.
        kf += ["100%{opacity:1}"] if i == last else \
            [f"{pct(bounds[i] - h)}{{opacity:1}}", f"{pct(bounds[i] + h)}{{opacity:0}}", "100%{opacity:0}"]
        rules.append(f"@keyframes ad-f{i}{{{''.join(kf)}}}"
                     f"#ad>.frame[data-i='{i}']{{animation:ad-f{i} {T}ms linear 1 forwards}}")
    if hide_corner_on_last:
        b = bounds[-1]
        rules.append(f"@keyframes ad-logo{{0%{{opacity:1}}{pct(b - h)}{{opacity:1}}{pct(b + h)}{{opacity:0}}100%{{opacity:0}}}}"
                     f"#ad .logo-corner{{animation:ad-logo {T}ms linear 1 forwards}}")
    return "".join(rules)
