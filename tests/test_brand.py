import json
import shutil
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from creative_agent.brand import BrandKit, load_brand, normalize_hex
from creative_agent.config import settings

REAL_KIT = Path(__file__).resolve().parent.parent / "config" / "brands" / "maxis" / "brand_kit.json"


@pytest.fixture
def brands_dir(tmp_path, monkeypatch):
    """A temporary brand folder with the real kit JSON and small synthetic RGBA logos (no dependency on T07)."""
    root = tmp_path / "brands"
    maxis = root / "maxis"
    (maxis / "logos").mkdir(parents=True)
    shutil.copy(REAL_KIT, maxis / "brand_kit.json")
    for name, size in {"stacked": (31, 40), "horizontal": (120, 36)}.items():
        for colour in ("green", "white"):
            Image.new("RGBA", size, (64, 199, 6, 255)).save(maxis / "logos" / f"{name}-{colour}.png")
    monkeypatch.setattr(settings, "brands_dir", root)
    return root


@pytest.fixture
def kit_data():
    return json.loads(REAL_KIT.read_text())


def test_kit_loads(brands_dir):
    kit = load_brand("maxis")
    assert kit.brand_id == "maxis"
    assert kit.fonts.active_family() == "Figtree"
    assert kit.fonts.active_files()[600] == "fonts/Figtree-SemiBold.woff2"
    assert kit.path("logos/x.png") == brands_dir / "maxis" / "logos" / "x.png"


def test_unknown_brand_raises(brands_dir):
    with pytest.raises(FileNotFoundError):
        load_brand("nope")


def test_normalize_hex():
    assert normalize_hex("#abc") == "#AABBCC"
    assert normalize_hex("40c706") == "#40C706"
    for bad in ("#12", "#gggggg", "", "#12345"):
        with pytest.raises(ValueError):
            normalize_hex(bad)


def test_invalid_hex_rejected_in_kit(kit_data):
    kit_data["colors"]["black"]["hex"] = "#zzz"
    with pytest.raises(ValidationError):
        BrandKit.model_validate(kit_data)


def test_text_color_missing_from_colors_rejected(kit_data):
    kit_data["text_colors"].append("purple")
    with pytest.raises(ValidationError, match="purple"):
        BrandKit.model_validate(kit_data)


def test_missing_logo_variant_rejected(kit_data):
    del kit_data["logo"]["variants"]["horizontal_white"]
    with pytest.raises(ValidationError, match="horizontal_white"):
        BrandKit.model_validate(kit_data)


def test_corner_for(brands_dir):
    logo = load_brand("maxis").logo
    assert logo.corner_for("320x50").variant == "horizontal"
    assert logo.corner_for("320x50").height_px == 18
    assert logo.corner_for("300x250").variant == "stacked"
    assert logo.corner_for("300x250").height_px == 40


def test_palette_and_text_palette(brands_dir):
    kit = load_brand("maxis")
    assert set(kit.palette()) == {"maxis_green", "aqua_blue", "light_grey", "black", "white"}
    assert kit.palette()["maxis_green"] == (64, 199, 6)
    assert set(kit.text_palette()) == {"black", "white", "maxis_green"}


def test_logo_aspect(brands_dir):
    kit = load_brand("maxis")
    assert kit.logo_aspect("stacked") == pytest.approx(0.775)
    assert kit.logo_aspect("horizontal") == pytest.approx(120 / 36)


@pytest.mark.parametrize("name", ["Maxis", "maxis", "Maxis Home Fibre", "  MAXIS  ", "Maxis Postpaid"])
def test_matches_brand_accepts(brands_dir, name):
    assert load_brand("maxis").matches_brand(name)


@pytest.mark.parametrize("name", ["Income", "", None, "Maxim"])
def test_matches_brand_rejects(brands_dir, name):
    assert not load_brand("maxis").matches_brand(name)


def test_every_real_kit_field_has_source_page(kit_data):
    for name, c in kit_data["colors"].items():
        assert c.get("source_page"), name
