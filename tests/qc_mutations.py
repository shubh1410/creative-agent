"""Deliberately broken versions of the good ad (tests/fixtures/qc/make_good_ad.py), one rule broken each.

Each mutation maps to the QC check it must fail. `extra` lists any other check that necessarily fails with it, with a
one-line `reason`; the browser test asserts the failed set is exactly {expected} | extra, so nothing is loosened silently.
"""
import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Mutation:
    fn: Callable[[str], str]
    expected: str
    extra: frozenset[str] = field(default_factory=frozenset)
    reason: str = ""


def sub(html: str, old: str, new: str) -> str:
    """Replace exactly one occurrence; fail loudly if the good ad's markup changed under the mutation."""
    assert html.count(old) == 1, f"mutation anchor not found exactly once: {old[:60]!r}"
    return html.replace(old, new)


def off_palette_text(html: str) -> str:
    return sub(html, "font-weight:600;color:#000000}", "font-weight:600;color:#333333}")


def overflow(html: str) -> str:
    return sub(html, ".headline{position:absolute;", ".headline{height:20px;overflow:hidden;position:absolute;")


def logo_in_box(html: str) -> str:
    box = '<div style="position:absolute;left:255px;top:22px;width:40px;height:45px;background:#FFFFFF"></div>'
    return sub(html, '<img class="logo logo-corner"', box + '<img class="logo logo-corner"')


def text_in_adinfo(html: str) -> str:
    return sub(html, ".legal{position:absolute;left:14px;top:225px;", ".legal{position:absolute;right:2px;top:2px;")


def four_font_sizes(html: str) -> str:
    spans = ('<span style="font-size:11px">Get yours</span> <span style="font-size:12px">with Maxis</span> '
             '<span style="font-size:14px">today</span>')
    return sub(html, 'data-role="cta">Get yours with Maxis today</div>', f'data-role="cta">{spans}</div>')


def wrong_weight(html: str) -> str:
    return sub(html, "line-height:1.15;font-weight:600;", "line-height:1.15;font-weight:400;")


def unapproved_image(html: str) -> str:
    return sub(html, 'data-asset="products_1"', 'data-asset="stock_photo"')


def oversize(html: str) -> str:
    return sub(html, "<body>", "<body><!--" + "x" * (160 * 1024) + "-->")


def cover_click(html: str) -> str:
    cover = '<div class="cover" style="position:absolute;left:0;top:0;width:300px;height:250px"></div>'
    stopper = "<script>document.querySelector('.cover').addEventListener('click',function(e){e.stopPropagation();});</script>"
    return sub(sub(html, "</div><script>", cover + "</div><script>"), "</body>", stopper + "</body>")


def drop_legal(html: str) -> str:
    out, n = re.subn(r'<p class="legal"[^>]*>.*?</p>', "", html, flags=re.S)
    assert n == 1
    return out


def changed_copy(html: str) -> str:
    return sub(html, "</strong> Device Sale</p>", "</strong> Device Sales</p>")


def external_font(html: str) -> str:
    link = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Roboto">'
    return sub(html, "<style>", link + "<style>")


def console_error(html: str) -> str:
    return sub(html, "</body>", '<script>console.error("qc fixture error");</script></body>')


def move_logo(html: str) -> str:
    out, n = re.subn(r'(logo-corner"[^>]*style=")left:[^;]+;top:[^;]+;', r"\1left:230px;top:190px;", html, count=1)
    assert n == 1
    return out


def looping_animation(html: str) -> str:
    return sub(html, "ad-f0 7000ms linear 1 forwards", "ad-f0 7000ms linear infinite")


def stray_box(html: str) -> str:
    box = '<div style="position:absolute;left:200px;top:195px;width:80px;height:30px;background:#FFFFFF"></div>'
    return sub(html, '<div class="cta"', box + '<div class="cta"')


def image_over_cta(html: str) -> str:
    return sub(html, ".product{position:absolute;left:160px;top:70px;",
               ".product{position:absolute;z-index:5;left:10px;top:150px;")


def legal_on_cta(html: str) -> str:
    return sub(html, ".legal{position:absolute;left:14px;top:225px;", ".legal{position:absolute;left:20px;top:168px;")


MUTATIONS: dict[str, Mutation] = {
    "off_palette_text": Mutation(off_palette_text, "brand_colors"),
    "overflow": Mutation(overflow, "text_overflow"),
    "logo_in_box": Mutation(logo_in_box, "logo_no_container", frozenset({"empty_shape"}),
                            "the white box around the logo is a filled box with nothing in it"),
    "text_in_adinfo": Mutation(text_in_adinfo, "adinfo_zone"),
    "four_font_sizes": Mutation(four_font_sizes, "font_size_count"),
    "wrong_weight": Mutation(wrong_weight, "font_weight_by_role"),
    "unapproved_image": Mutation(unapproved_image, "assets_whitelist"),
    "oversize": Mutation(oversize, "file_size"),
    "cover_click": Mutation(cover_click, "click_opens"),
    "drop_legal": Mutation(drop_legal, "legal_present", frozenset({"copy_fidelity", "legal_once"}),
                           "s5.legal is a kept element (copy_fidelity) and the only legal text in the ad (legal_once)"),
    "changed_copy": Mutation(changed_copy, "copy_fidelity"),
    "external_font": Mutation(external_font, "external_requests", frozenset({"console_errors"}),
                              "run_qc aborts the external request and Chromium logs that as a 'Failed to load resource' error"),
    "console_error": Mutation(console_error, "console_errors"),
    "move_logo": Mutation(move_logo, "logo_spec"),
    "looping_animation": Mutation(looping_animation, "timeline"),
    "stray_box": Mutation(stray_box, "empty_shape"),
    "image_over_cta": Mutation(image_over_cta, "text_occluded"),
    "legal_on_cta": Mutation(legal_on_cta, "text_overlap"),
}
