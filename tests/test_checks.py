from PIL import Image

from creative_agent.qc import checks as C
from creative_agent.qc.checks import PER_FRAME
from creative_agent.render.finalize import corner_logo_box, end_logo_size


def el(**kw):
    base = dict(tag="p", id=None, cls="", role="body", roleCtx="body", el=None, insideEl=None, asset=None,
                variant=None, text=None, ownText="Hi", lines=None, bbox=dict(x=10, y=10, w=50, h=12),
                scroll=dict(w=50, h=12), client=dict(w=50, h=12), color="rgb(255, 255, 255)",
                bg="rgba(0, 0, 0, 0)", bgImage="none", boxShadow="none", borderColors=["rgb(0, 0, 0)"] * 4,
                borderWidths=[0, 0, 0, 0], fill="", stroke="", fontFamily="Figtree, sans-serif", fontSize=12,
                fontWeight=400, isImage=False, natural=None, idx=0, ancestors=[], hasContent=True, textRects=[],
                coveredBy=[])
    return {**base, **kw}


def bb(x, y, w, h):
    return dict(x=x, y=y, w=w, h=h)


def fm(*elements):
    return {"elements": list(elements)}


def logo(ctx, end=False, variant="green", **kw):
    """A measured logo element placed exactly where the brand says."""
    if end:
        _, w, h = end_logo_size(ctx.brand, ctx.size)
        box = bb((ctx.W - w) / 2, (ctx.H - h) / 2, w, h)
    else:
        x, y, w, h, _ = corner_logo_box(ctx.brand, ctx.size)
        box = bb(x, y, w, h)
    base = dict(tag="img", role="logo", roleCtx=None, ownText="", isImage=True, variant=variant, bbox=box,
                natural=dict(w=w * 2, h=h * 2))
    return el(**{**base, **kw})


def solid(color, size=(300, 250)):
    return Image.new("RGB", size, color)


# ---------- registry ----------

def test_per_frame_registry_has_all_21_checks():
    names = [c.__name__ for c in PER_FRAME]
    assert len(PER_FRAME) == 21
    assert len(set(names)) == 21
    assert set(names) == {
        "text_overlap", "text_occluded", "empty_shape", "image_resolution",
        "canvas_bounds", "text_overflow", "logo_spec", "logo_variant", "logo_no_container", "adinfo_zone",
        "brand_colors", "brand_fonts", "font_weight_by_role", "font_size_count", "text_color_count",
        "headline_lines", "min_font_size", "copy_fidelity", "legal_present", "cta_visible", "assets_whitelist"}


def test_result_carries_name_fixable_and_frame(qc_ctx):
    r = C.logo_spec(fm(), qc_ctx.at(1), None)
    assert (r.name, r.fixable, r.frame) == ("logo_spec", False, 1)
    assert C.canvas_bounds(fm(), qc_ctx, None).fixable is True


# ---------- layout ----------

def test_canvas_bounds_pass(qc_ctx):
    assert C.canvas_bounds(fm(el(bbox=bb(0, 0, 300, 250))), qc_ctx, None).passed


def test_canvas_bounds_fail(qc_ctx):
    r = C.canvas_bounds(fm(el(bbox=bb(280, 10, 40, 12))), qc_ctx, None)
    assert not r.passed and "300x250" in r.message


def test_text_overflow_pass(qc_ctx):
    assert C.text_overflow(fm(el()), qc_ctx, None).passed


def test_text_overflow_tolerates_a_few_px_but_not_more(qc_ctx):
    assert C.text_overflow(fm(el(scroll=dict(w=50, h=15))), qc_ctx, None).passed        # 3px over
    assert not C.text_overflow(fm(el(scroll=dict(w=50, h=17))), qc_ctx, None).passed    # 5px over


def test_text_overflow_fail_and_ignores_non_role_boxes(qc_ctx):
    assert not C.text_overflow(fm(el(scroll=dict(w=50, h=40))), qc_ctx, None).passed
    assert C.text_overflow(fm(el(role=None, roleCtx=None, scroll=dict(w=50, h=40))), qc_ctx, None).passed


# ---------- logo ----------

def test_logo_spec_corner_pass(qc_ctx):
    assert C.logo_spec(fm(logo(qc_ctx)), qc_ctx, None).passed


def test_logo_spec_corner_320x50_uses_horizontal_override(brand):
    from creative_agent.models import FramePlan, SizePlan
    from creative_agent.qc.context import QCContext
    plan = SizePlan(size="320x50", backup_frame=0, frames=[
        FramePlan(scene_id="s1", keep=[], show_products=False, duration_ms=1000)])
    ctx = QCContext(size="320x50", W=320, H=50, brand=brand, plan=plan, content={}, assets=set(), click_url="",
                    budget_floor_hit=False)
    x, y, w, h, place = corner_logo_box(brand, "320x50")
    assert place.variant == "horizontal" and h == 18 and x == 320 - 10 - w and y == 22
    assert C.logo_spec(fm(logo(ctx)), ctx, None).passed


def test_logo_spec_fail_wrong_position_missing_and_duplicate(qc_ctx):
    moved = logo(qc_ctx, bbox=bb(100, 100, 31, 40))
    assert not C.logo_spec(fm(moved), qc_ctx, None).passed
    assert "found 0" in C.logo_spec(fm(), qc_ctx, None).message
    assert "found 2" in C.logo_spec(fm(logo(qc_ctx), logo(qc_ctx)), qc_ctx, None).message


def test_logo_spec_fail_stretched(qc_ctx):
    r = C.logo_spec(fm(logo(qc_ctx, natural=dict(w=100, h=100))), qc_ctx, None)
    assert not r.passed and "stretched" in r.message


def test_logo_spec_end_frame_pass_and_fail(qc_ctx):
    end = qc_ctx.at(2)
    assert end.is_end
    assert C.logo_spec(fm(logo(end, end=True)), end, None).passed
    off = logo(end, end=True, bbox=bb(5, 5, 38, 49))
    r = C.logo_spec(fm(off), end, None)
    assert not r.passed and "centred" in r.message
    wrong = logo(end, end=True)
    wrong["bbox"] = bb(wrong["bbox"]["x"], wrong["bbox"]["y"], 80, 103)
    wrong["natural"] = dict(w=80, h=103)
    assert "expected" in C.logo_spec(fm(wrong), end, None).message


def test_logo_spec_end_frame_rejects_corner_position(qc_ctx):
    end = qc_ctx.at(2)
    assert not C.logo_spec(fm(logo(qc_ctx)), end, None).passed


def test_logo_variant_green_on_white_passes(qc_ctx):
    assert C.logo_variant(fm(logo(qc_ctx, variant="green")), qc_ctx, solid("white")).passed


def test_logo_variant_white_on_white_fails_and_white_on_black_passes(qc_ctx):
    r = C.logo_variant(fm(logo(qc_ctx, variant="white")), qc_ctx, solid("white"))
    assert not r.passed and "luminance" in r.message
    assert C.logo_variant(fm(logo(qc_ctx, variant="white")), qc_ctx, solid("black")).passed
    assert not C.logo_variant(fm(logo(qc_ctx, variant="green")), qc_ctx, solid("black")).passed


def test_logo_variant_skips_without_logo_or_screenshot(qc_ctx):
    assert C.logo_variant(fm(), qc_ctx, solid("white")).passed
    assert C.logo_variant(fm(logo(qc_ctx)), qc_ctx, None).passed


def test_logo_no_container_pass(qc_ctx):
    assert C.logo_no_container(fm(logo(qc_ctx), el(bbox=bb(0, 0, 300, 250), cls="frame")), qc_ctx, None).passed
    assert C.logo_no_container(fm(logo(qc_ctx)), qc_ctx, None).passed


def test_logo_no_container_fail_fill_border_shadow(qc_ctx):
    lg = logo(qc_ctx)
    pill = bb(lg["bbox"]["x"] - 4, lg["bbox"]["y"] - 4, 50, 56)
    assert not C.logo_no_container(fm(lg, el(bbox=pill, bg="rgb(255, 255, 255)")), qc_ctx, None).passed
    assert not C.logo_no_container(fm(lg, el(bbox=pill, borderWidths=[1, 1, 1, 1])), qc_ctx, None).passed
    assert not C.logo_no_container(fm(lg, el(bbox=pill, boxShadow="rgba(0, 0, 0, 0.3) 0px 2px 4px")),
                                   qc_ctx, None).passed


def test_adinfo_zone_pass(qc_ctx):
    assert C.adinfo_zone(fm(el(bbox=bb(10, 10, 100, 12)), logo(qc_ctx)), qc_ctx, None).passed


def test_adinfo_zone_fail_text_and_ignores_logo_and_background(qc_ctx):
    assert not C.adinfo_zone(fm(el(bbox=dict(x=280, y=2, w=18, h=10))), qc_ctx, None).passed
    assert C.adinfo_zone(fm(el(bbox=bb(0, 0, 300, 250), cls="frame", ownText="")), qc_ctx, None).passed


# ---------- colour and type ----------

def test_brand_colors_pass(qc_ctx):
    good = el(color="rgb(64, 199, 6)", bg="rgb(223, 226, 228)",
              bgImage="linear-gradient(rgb(64, 199, 6), rgb(255, 255, 255))",
              boxShadow="rgba(0, 0, 0, 0.3) 0px 4px 12px 0px", borderColors=["rgb(0, 127, 137)"] * 4,
              borderWidths=[1, 1, 1, 1])
    assert C.brand_colors(fm(good, el(bg="rgba(0, 0, 0, 0)")), qc_ctx, None).passed


def test_brand_colors_snaps_within_tolerance(qc_ctx):
    assert C.brand_colors(fm(el(color="rgb(66, 200, 8)")), qc_ctx, None).passed


def test_brand_colors_fail_text(qc_ctx):
    r = C.brand_colors(fm(el(color="rgb(255, 0, 0)")), qc_ctx, None)
    assert not r.passed and "text colour #FF0000" in r.message


def test_brand_colors_aqua_is_not_a_text_colour_but_is_a_fill(qc_ctx):
    assert not C.brand_colors(fm(el(color="rgb(0, 127, 137)")), qc_ctx, None).passed
    assert C.brand_colors(fm(el(ownText="", bg="rgb(0, 127, 137)")), qc_ctx, None).passed


def test_brand_colors_fail_background_border_gradient_shadow_svg(qc_ctx):
    assert not C.brand_colors(fm(el(ownText="", bg="rgb(200, 0, 0)")), qc_ctx, None).passed
    assert not C.brand_colors(fm(el(ownText="", borderColors=["rgb(200, 0, 0)"] * 4, borderWidths=[2] * 4)),
                              qc_ctx, None).passed
    assert not C.brand_colors(fm(el(ownText="", bgImage="linear-gradient(rgb(64, 199, 6), rgb(200, 0, 0))")),
                              qc_ctx, None).passed
    assert not C.brand_colors(fm(el(ownText="", boxShadow="rgba(200, 0, 0, 0.5) 0px 0px 5px")), qc_ctx, None).passed
    assert not C.brand_colors(fm(el(tag="path", ownText="", fill="rgb(200, 0, 0)")), qc_ctx, None).passed


def test_brand_colors_ignores_images_and_transparent(qc_ctx):
    assert C.brand_colors(fm(el(isImage=True, ownText="", bg="rgb(200, 0, 0)")), qc_ctx, None).passed
    assert C.brand_colors(fm(el(ownText="", bg="rgba(200, 0, 0, 0)")), qc_ctx, None).passed


def test_brand_fonts_pass(qc_ctx):
    assert C.brand_fonts(fm(el(fontFamily='"Figtree", sans-serif')), qc_ctx, None).passed


def test_brand_fonts_fail(qc_ctx):
    r = C.brand_fonts(fm(el(fontFamily="Arial, sans-serif")), qc_ctx, None)
    assert not r.passed and "Arial" in r.message


def test_font_weight_by_role_pass(qc_ctx):
    items = [el(role="headline", roleCtx="headline", fontWeight=600), el(role="body", fontWeight=400),
             el(role="legal", roleCtx="legal", fontWeight=400), el(role="cta", roleCtx="cta", fontWeight=600)]
    assert C.font_weight_by_role(fm(*items), qc_ctx, None).passed


def test_font_weight_by_role_fail_and_highlight_inside_headline(qc_ctx):
    assert not C.font_weight_by_role(fm(el(role="headline", roleCtx="headline", fontWeight=400)),
                                     qc_ctx, None).passed
    headline = el(role="headline", roleCtx="headline", fontWeight=600, ownText="Device Sale")
    hl_ok = el(tag="strong", role="highlight", roleCtx="highlight", fontWeight=700, ownText="RM100 Million")
    hl_bad = el(tag="strong", role="highlight", roleCtx="highlight", fontWeight=600, ownText="RM100 Million")
    assert C.font_weight_by_role(fm(headline, hl_ok), qc_ctx, None).passed
    r = C.font_weight_by_role(fm(headline, hl_bad), qc_ctx, None)
    assert not r.passed and "700 for highlight" in r.message


def test_font_size_count_pass(qc_ctx):
    assert C.font_size_count(fm(*[el(fontSize=s) for s in (10, 12, 18)]), qc_ctx, None).passed
    assert C.font_size_count(fm(*[el(fontSize=s) for s in (12, 12.2, 12.4)]), qc_ctx, None).passed


def test_font_size_count_four_sizes_fail(qc_ctx):
    r = C.font_size_count(fm(*[el(fontSize=s) for s in (10, 12, 14, 18)]), qc_ctx, None)
    assert not r.passed and "4 font sizes" in r.message


def test_text_color_count_pass(qc_ctx):
    white, green = "rgb(255, 255, 255)", "rgb(64, 199, 6)"
    assert C.text_color_count(fm(el(color=white), el(color=green)), qc_ctx, None).passed


def test_text_color_count_fail(qc_ctx):
    white, green, black = "rgb(255, 255, 255)", "rgb(64, 199, 6)", "rgb(0, 0, 0)"
    r = C.text_color_count(fm(el(color=white), el(color=green), el(color=black)), qc_ctx, None)
    assert not r.passed and "3 text colours" in r.message


def head(n):
    return el(role="headline", roleCtx="headline", lines=n)


def test_headline_lines_two_passes(qc_ctx):
    r = C.headline_lines(fm(head(2)), qc_ctx, None)
    assert r.passed and r.message == ""


def test_headline_lines_three_passes_with_warning(qc_ctx):
    r = C.headline_lines(fm(head(3)), qc_ctx, None)
    assert r.passed and r.message.startswith("warning:")


def test_headline_lines_four_fails(qc_ctx):
    r = C.headline_lines(fm(head(4)), qc_ctx, None)
    assert not r.passed and "4 lines" in r.message


def test_min_font_size_pass_legal_at_8(qc_ctx):
    assert C.min_font_size(fm(el(fontSize=10)), qc_ctx, None).passed
    assert C.min_font_size(fm(el(role="legal", roleCtx="legal", fontSize=8)), qc_ctx, None).passed


def test_min_font_size_fail_body_at_8_and_legal_at_7(qc_ctx):
    assert not C.min_font_size(fm(el(role="body", fontSize=8)), qc_ctx, None).passed
    assert not C.min_font_size(fm(el(role="legal", roleCtx="legal", fontSize=7)), qc_ctx, None).passed


# ---------- content ----------

def kept(eid, text, **kw):
    return el(el=eid, text=text, ownText=text, role="headline", roleCtx="headline", **kw)


def test_copy_fidelity_pass(qc_ctx):
    assert C.copy_fidelity(fm(kept("s1.headline", "RM100 Million Device Sale")), qc_ctx, None).passed


def test_copy_fidelity_normalises_quotes_case_and_whitespace(qc_ctx):
    assert C.copy_fidelity(fm(kept("s1.headline", "rm100  million\u00a0device sale")), qc_ctx, None).passed


def test_copy_fidelity_fail_changed_text(qc_ctx):
    r = C.copy_fidelity(fm(kept("s1.headline", "RM100 Million Sale")), qc_ctx, None)
    assert not r.passed and "expected 'RM100 Million Device Sale'" in r.message


def test_copy_fidelity_fail_missing_kept_element(qc_ctx):
    r = C.copy_fidelity(fm(), qc_ctx, None)
    assert not r.passed and "s1.headline is missing" in r.message


def test_copy_fidelity_fail_untracked_text_and_unknown_id(qc_ctx):
    ok = kept("s1.headline", "RM100 Million Device Sale")
    r = C.copy_fidelity(fm(ok, el(ownText="Free gift!")), qc_ctx, None)
    assert not r.passed and "isn't part of any content element" in r.message
    assert "unknown data-el" in C.copy_fidelity(fm(ok, kept("sX.nope", "x")), qc_ctx, None).message


def test_copy_fidelity_allows_text_inside_a_content_element(qc_ctx):
    ok = kept("s1.headline", "RM100 Million Device Sale")
    inner = el(tag="strong", role="highlight", ownText="RM100 Million", insideEl="s1.headline")
    assert C.copy_fidelity(fm(ok, inner), qc_ctx, None).passed


def test_legal_present_pass(qc_ctx):
    assert C.legal_present(fm(el(role="legal", roleCtx="legal")), qc_ctx.at(1), None).passed
    assert C.legal_present(fm(el(role="body")), qc_ctx, None).passed            # s1 shows no products


def test_legal_present_fail_on_product_frame(qc_ctx):
    r = C.legal_present(fm(el(role="body")), qc_ctx.at(1), None)
    assert not r.passed and "no legal text" in r.message


def test_cta_visible_pass(qc_ctx):
    cta = el(el="s5.headline", role="cta", bbox=bb(100, 200, 100, 30))
    assert C.cta_visible(fm(cta), qc_ctx.at(1), None).passed
    assert C.cta_visible(fm(), qc_ctx, None).passed                              # no CTA on s1


def test_cta_visible_fail_missing_or_off_canvas(qc_ctx):
    s5 = qc_ctx.at(1)
    assert not C.cta_visible(fm(), s5, None).passed
    off = el(el="s5.headline", role="cta", bbox=bb(250, 200, 100, 30))
    assert not C.cta_visible(fm(off), s5, None).passed


def test_assets_whitelist_pass(qc_ctx):
    assert C.assets_whitelist(fm(el(isImage=True, tag="img", asset="products_1")), qc_ctx, None).passed
    assert C.assets_whitelist(fm(logo(qc_ctx)), qc_ctx, None).passed
    assert C.assets_whitelist(fm(el(bgImage="linear-gradient(red, blue)")), qc_ctx, None).passed


def test_assets_whitelist_fail_unapproved_image(qc_ctx):
    bad = C.assets_whitelist(fm(el(isImage=True, tag="img", asset="evil")), qc_ctx, None)
    assert not bad.passed and "evil" in bad.message
    assert "without data-asset" in C.assets_whitelist(fm(el(isImage=True, tag="img")), qc_ctx, None).message


def test_assets_whitelist_fail_css_background_image(qc_ctx):
    css = C.assets_whitelist(fm(el(bgImage='url("data:image/png;base64,AAAA")')), qc_ctx, None)
    assert not css.passed and "gradients" in css.message


# ---------- overlap, occlusion, stray shapes, image resolution ----------

def txt(idx, x, y, w=80, h=12, **kw):
    return el(idx=idx, bbox=bb(x, y, w, h), textRects=[bb(x, y, w, h)], **kw)


def test_text_overlap_pass_when_separate(qc_ctx):
    assert C.text_overlap(fm(txt(1, 10, 10), txt(2, 10, 40)), qc_ctx, None).passed


def test_text_overlap_fail_text_on_text(qc_ctx):
    r = C.text_overlap(fm(txt(1, 10, 10), txt(2, 20, 14)), qc_ctx, None)
    assert not r.passed and "overlaps" in r.message


def test_text_overlap_fail_cta_pill_on_legal(qc_ctx):
    pill = txt(1, 10, 100, w=120, h=26, role="cta", roleCtx="cta", bg="rgb(64, 199, 6)")
    legal = txt(2, 12, 115, w=100, h=10, role="legal", roleCtx="legal")
    assert not C.text_overlap(fm(pill, legal), qc_ctx, None).passed


def test_text_overlap_ignores_nested_elements(qc_ctx):
    outer = txt(1, 10, 10, w=100, h=40)
    inner = txt(2, 20, 20, w=40, h=12, ancestors=[1])
    assert C.text_overlap(fm(outer, inner), qc_ctx, None).passed


def test_text_overlap_ignores_hairline_touch(qc_ctx):
    assert C.text_overlap(fm(txt(1, 10, 10), txt(2, 10, 21)), qc_ctx, None).passed


def test_text_occluded_pass_and_fail(qc_ctx):
    image = el(idx=9, tag="img", isImage=True, ownText="", bbox=bb(0, 0, 100, 100))
    clear = txt(1, 10, 10, coveredBy=[dict(idx=9, share=0.03)])
    hidden = txt(2, 10, 10, role="headline", roleCtx="headline", coveredBy=[dict(idx=9, share=0.6)])
    assert C.text_occluded(fm(clear, image), qc_ctx, None).passed
    r = C.text_occluded(fm(hidden, image), qc_ctx, None)
    assert not r.passed and "60% covered" in r.message


def test_text_occluded_ignores_ancestors(qc_ctx):
    bg = el(idx=5, ownText="", bbox=bb(0, 0, 300, 250), bg="rgb(0, 127, 137)")
    t = txt(1, 10, 10, ancestors=[5], coveredBy=[dict(idx=5, share=1.0)])
    assert C.text_occluded(fm(bg, t), qc_ctx, None).passed


def test_empty_shape_fails_filled_box_without_content(qc_ctx):
    ghost = el(ownText="", hasContent=False, bg="rgb(255, 255, 255)", bbox=bb(196, 137, 80, 34), tag="div")
    r = C.empty_shape(fm(ghost), qc_ctx, None)
    assert not r.passed and "80x34" in r.message


def test_empty_shape_allows_content_decoration_background_and_images(qc_ctx):
    full = dict(ownText="", hasContent=False, bg="rgb(255, 255, 255)", tag="div")
    items = [el(**{**full, "bbox": bb(10, 10, 80, 34), "hasContent": True}),             # a pill with text
             el(**{**full, "bbox": bb(10, 10, 4, 4)}),                                    # a decorative dot
             el(**{**full, "bbox": bb(0, 0, 300, 250), "cls": "frame"}),                  # the frame background
             el(**{**full, "bbox": bb(10, 60, 80, 34), "bg": "rgba(0, 0, 0, 0)"}),       # transparent layout box
             el(**{**full, "bbox": bb(10, 100, 80, 34), "isImage": True, "tag": "img"})]    # an image
    assert C.empty_shape(fm(*items), qc_ctx, None).passed


def test_image_resolution_pass_warn_fail(qc_ctx):
    def image(w, shown):
        return el(tag="img", isImage=True, ownText="", natural=dict(w=w, h=w), bbox=bb(0, 0, shown, shown))
    assert C.image_resolution(fm(image(200, 100)), qc_ctx, None).passed
    warn = C.image_resolution(fm(image(100, 125)), qc_ctx, None)
    assert warn.passed and warn.message.startswith("warning:")
    r = C.image_resolution(fm(image(100, 200)), qc_ctx, None)
    assert not r.passed and "2.0x" in r.message
    assert C.image_resolution(fm(logo(qc_ctx, natural=dict(w=10, h=13))), qc_ctx, None).passed
