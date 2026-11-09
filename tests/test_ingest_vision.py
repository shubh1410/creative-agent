from pathlib import Path

from fakes import FakeChatModel
from PIL import Image

from creative_agent.config import settings
from creative_agent.ingest import cache, crops, ocr
from creative_agent.ingest.vision import (ExtractionDraft, FrameBoxes, detect_frames, extract, validate_extraction)
from creative_agent.models import BBox, Issue, MasterSpec
from creative_agent.prompts import load_prompt

FIX = Path(__file__).parent / "fixtures" / "ingest"
META = {"node": "ingest_master", "model": "fake", "modality": "vision"}


def maxis() -> ExtractionDraft:
    return ExtractionDraft.model_validate_json((FIX / "maxis_extraction.json").read_text())


IDS = ["s1", "s2", "s3", "s4", "s5", "s6"]


def test_maxis_fixture_is_valid_and_copy_exact():
    d = maxis()
    assert validate_extraction(d, IDS, 576, 1024) == []
    s = {x.scene_id: x for x in d.scenes}
    assert s["s1"].elements[0].text == "RM100 Million Device Sale"
    assert s["s1"].elements[0].highlights == ["RM100 Million"]
    assert s["s5"].elements[0].highlights == ["yours", "today"]
    assert s["s6"].is_end_frame
    for sid in ("s2", "s3", "s4", "s5"):
        assert any(e.role == "legal" and e.text == "Terms & Conditions Apply" for e in s[sid].elements)
    assert all(any(e.role == "logo" for e in x.elements) for x in d.scenes)


def test_income_fixtures_valid():
    d = ExtractionDraft.model_validate_json((FIX / "income_extraction.json").read_text())
    assert d.detected_brand == "Income" and len(d.scenes) == 5
    frames = FrameBoxes.model_validate_json((FIX / "income_frames.json").read_text())
    assert len(frames.frames) == 5
    assert validate_extraction(d, ["s1", "s2", "s3", "s4", "s5"], 143, 286) == []


def test_validate_extraction_reports_each_error_type():
    d = maxis()
    d.scenes[0].scene_id = "x1"                                   # wrong ids
    d.scenes[1].is_end_frame = True                               # two end frames
    d.scenes[2].elements[0].bbox = BBox(x=500, y=10, w=200, h=10)  # outside
    d.scenes[3].elements[0].text = " "                            # text role without text
    d.scenes[4].elements[0].highlights = ["nope"]                 # not a substring
    d.scenes[4].product_bbox = BBox(x=0, y=0, w=9999, h=10)
    errs = "\n".join(validate_extraction(d, IDS, 576, 1024))
    for frag in ("scene ids must be exactly", "at most one scene", "is outside the 576x1024", "has no text",
                 "is not part of", "product box is outside"):
        assert frag in errs


def test_rescaled_scales_every_box():
    d = maxis()
    r = d.rescaled(2)
    for a, b in zip(d.scenes, r.scenes):
        assert (b.product_bbox is None) == (a.product_bbox is None)
        if a.product_bbox:
            assert b.product_bbox == a.product_bbox.scaled(2)
        for ea, eb in zip(a.elements, b.elements):
            assert eb.bbox == ea.bbox.scaled(2)


async def test_extract_scales_boxes_and_sends_six_images(tmp_path):
    frames = {}
    for sid in IDS:
        p = tmp_path / f"{sid}.png"
        Image.new("RGB", (1080, 1920), "green").save(p)
        frames[sid] = p
    llm = FakeChatModel(structured=[maxis()])
    out = await extract(llm, frames, 1080, 1920, meta=META, config=None)
    f = 1920 / 1024
    assert out.scenes[0].elements[0].bbox == maxis().scenes[0].elements[0].bbox.scaled(f)
    content = llm.calls[0].messages[1].content
    assert sum(1 for c in content if c["type"] == "image_url") == 6
    assert "576x1024" in content[0]["text"]
    assert llm.calls[0].messages[0].content == load_prompt("vision_extract")


async def test_detect_frames_scales_back(tmp_path):
    p = tmp_path / "wide.png"
    Image.new("RGB", (2048, 1024), "white").save(p)      # downscaled to 1024x512 for the model
    llm = FakeChatModel(structured=[FrameBoxes(frames=[BBox(x=10, y=20, w=100, h=200)])])
    boxes = await detect_frames(llm, p, meta=META, config=None)
    assert boxes == [BBox(x=20, y=40, w=200, h=400)]
    assert "1024x512" in llm.calls[0].messages[1].content[0]["text"]


def test_crop_box_clamps_at_edges(tmp_path):
    src = tmp_path / "a.png"
    Image.new("RGB", (100, 80), "red").save(src)
    out = crops.crop_box(src, BBox(x=5, y=5, w=50, h=30), tmp_path / "c.png", pad=12)
    with Image.open(out) as im:
        assert im.size == (5 + 50 + 12, 5 + 30 + 12)
    out = crops.crop_box(src, BBox(x=60, y=50, w=40, h=30), tmp_path / "d.png", pad=12)
    with Image.open(out) as im:
        assert im.size == (100 - 48, 80 - 38)


def _pattern(path: Path, seed: int, size=(200, 200)):
    import numpy as np
    rng = np.random.default_rng(seed)
    a = (rng.random((8, 8)) * 255).astype("uint8")
    Image.fromarray(a).resize(size, Image.NEAREST).convert("RGB").save(path)


def test_same_image(tmp_path):
    _pattern(tmp_path / "a.png", 1)
    _pattern(tmp_path / "b.png", 1, size=(220, 190))       # same region, slightly different crop size
    _pattern(tmp_path / "c.png", 2)
    assert crops.same_image(tmp_path / "a.png", tmp_path / "b.png")
    assert not crops.same_image(tmp_path / "a.png", tmp_path / "c.png")


def test_sample_text_color_white_on_green(tmp_path):
    from creative_agent.brand import load_brand
    im = Image.new("RGB", (200, 60), (64, 199, 6))
    for x in range(20, 180, 6):
        for y in range(20, 40):
            im.putpixel((x, y), (255, 255, 255))
            im.putpixel((x + 1, y), (255, 255, 255))
    p = tmp_path / "t.png"
    im.save(p)
    name, dist = crops.sample_text_color(p, BBox(x=0, y=0, w=200, h=60), load_brand("maxis").text_palette())
    assert name == "white" and dist < 8


def test_similarity():
    assert ocr.similarity("Get yours with Maxis today", "maxis Get yours with Maxis today T&C") == 1.0
    assert ocr.similarity("Get yours with Maxis today", "completely different words") < 0.5


def test_ocr_text_none_without_tesseract(tmp_path, monkeypatch):
    import sys
    p = tmp_path / "a.png"
    Image.new("RGB", (10, 10)).save(p)
    monkeypatch.setitem(sys.modules, "pytesseract", None)         # import fails
    assert ocr.ocr_text(p) is None


def test_ocr_text_with_fake_tesseract(tmp_path, monkeypatch):
    import sys
    import types
    fake = types.SimpleNamespace(image_to_string=lambda img: "hello world")
    monkeypatch.setitem(sys.modules, "pytesseract", fake)
    p = tmp_path / "a.png"
    Image.new("RGB", (10, 10)).save(p)
    assert ocr.ocr_text(p) == "hello world"


def test_cache_round_trip(data_dir):
    spec = MasterSpec(source="m.mp4", kind="video", width=1080, height=1920, detected_brand="Maxis",
                      scenes=[], cache_key="k1")
    data = cache.CachedIngest(spec=spec, notes=[], issues=[Issue(code="X", severity="info", message="m")])
    assert cache.load("k1") is None
    cache.save("k1", data)
    assert cache.load("k1") == data
    assert cache.cache_dir("k1").is_relative_to(settings.data_dir)
    f = data_dir / "f.bin"
    f.write_bytes(b"abc")
    assert cache.cache_key([f], "a") != cache.cache_key([f], "b")
    assert cache.cache_key([f], "a") == cache.cache_key([f], "a")
