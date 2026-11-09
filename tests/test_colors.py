import pytest

from creative_agent.colors import (
    colors_in,
    hex_to_rgb,
    nearest,
    parse_css_color,
    relative_luminance,
    rgb_to_hex,
)


def test_parse_rgb():
    assert parse_css_color("rgb(64, 199, 6)") == ((64, 199, 6), 1.0)


def test_parse_rgba():
    assert parse_css_color("rgba(0, 0, 0, 0)") == ((0, 0, 0), 0.0)
    assert parse_css_color("rgba(10, 20, 30, 0.5)") == ((10, 20, 30), 0.5)


def test_parse_modern_space_syntax():
    assert parse_css_color("rgb(0 0 0 / 50%)") == ((0, 0, 0), 0.5)
    assert parse_css_color("rgb(64 199 6)") == ((64, 199, 6), 1.0)


def test_parse_hex():
    assert parse_css_color("#40C706") == ((64, 199, 6), 1.0)
    assert parse_css_color("#fff") == ((255, 255, 255), 1.0)


@pytest.mark.parametrize("value", ["transparent", "", "none", "red", "#12", "linear-gradient(red, blue)"])
def test_parse_unparseable_returns_none(value):
    assert parse_css_color(value) is None


def test_hex_roundtrip():
    assert rgb_to_hex(hex_to_rgb("#40c706")) == "#40C706"


def test_nearest_on_palette():
    palette = {"green": (64, 199, 6), "white": (255, 255, 255), "black": (0, 0, 0)}
    name, d = nearest((73, 225, 12), palette)
    assert name == "green"
    assert d > 0
    assert nearest((250, 250, 250), palette)[0] == "white"
    assert nearest((1, 2, 3), palette) == ("black", pytest.approx((1 + 4 + 9) ** 0.5))


def test_relative_luminance():
    assert relative_luminance((255, 255, 255)) == pytest.approx(1.0)
    assert relative_luminance((0, 0, 0)) == pytest.approx(0.0)
    assert relative_luminance((64, 199, 6)) == pytest.approx(0.42, abs=0.01)


def test_colors_in_gradient():
    found = colors_in("linear-gradient(rgb(64, 199, 6), rgb(255, 255, 255))")
    assert len(found) == 2
    assert found[0][0] == (64, 199, 6)
    assert found[1][0] == (255, 255, 255)


def test_colors_in_box_shadow():
    found = colors_in("rgba(0, 0, 0, 0.3) 0px 4px 12px 0px")
    assert found == [((0, 0, 0), 0.3)]


def test_colors_in_handles_empty():
    assert colors_in("") == []
    assert colors_in("none") == []
