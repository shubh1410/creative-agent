import re

from creative_agent.render.timeline import RUNTIME_JS, timeline_css


def test_timeline_three_frames():
    css = timeline_css([2000, 2000, 2000])
    assert css.count("@keyframes ad-f") == 3 and "6000ms linear 1 forwards" in css
    assert "@keyframes ad-logo" in css                    # corner logo fades out before the end frame
    assert re.search(r"ad-f2\{0%\{opacity:0\}", css)       # last frame starts hidden


def test_single_frame_has_no_animation():
    assert timeline_css([3000]) == ""


def test_total_duration_is_sum_and_every_animation_plays_once_and_holds():
    css = timeline_css([2500, 2500, 2000])
    durations = re.findall(r"animation:ad-[a-z0-9]+ (\d+)ms linear (\S+) forwards", css)
    assert durations and all(d == "7000" and n == "1" for d, n in durations)
    assert "infinite" not in css


def test_first_frame_starts_visible_and_last_ends_visible():
    css = timeline_css([2000, 2000, 2000])
    assert re.search(r"@keyframes ad-f0\{0%\{opacity:1\}", css)
    assert re.search(r"@keyframes ad-f2\{.*?100%\{opacity:1\}\}", css)      # the last frame is held visible
    assert re.search(r"@keyframes ad-f0\{.*?100%\{opacity:0\}\}", css)      # earlier frames end hidden


def test_corner_logo_animation_can_be_omitted():
    assert "ad-logo" not in timeline_css([2000, 2000], hide_corner_on_last=False)


def test_runtime_js_exposes_show_and_click_handler():
    assert "window.__ad" in RUNTIME_JS and "show:function(i)" in RUNTIME_JS
    assert "window.open(window.clickTag)" in RUNTIME_JS
    assert "__HAS_END__" in RUNTIME_JS                     # the placeholder is filled in by finalize_html
