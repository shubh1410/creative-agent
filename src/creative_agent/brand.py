"""Brand kit schema and loader.

A brand kit (``config/brands/<brand_id>/brand_kit.json``) captures the brand guidelines that generated ads must
obey: colour palette, fonts, typography limits, logo placement and size rules, and file-weight / motion limits.
The models below validate that JSON and offer convenience accessors used by prompts, rendering and QC.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import BaseModel, PrivateAttr, field_validator, model_validator

from .colors import hex_to_rgb
from .config import settings

# Accepts "#abc", "abc", "#aabbcc" or "aabbcc"; group 1 is the digits without '#'.
HEX_RE = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def normalize_hex(value: str) -> str:
    """Return a canonical ``#RRGGBB`` upper-case form of a 3- or 6-digit hex colour.

    Raises:
        ValueError: If ``value`` isn't a valid hex colour.
    """
    m = HEX_RE.match(value.strip())
    if not m:
        raise ValueError(f"invalid hex colour: {value!r}")
    h = m.group(1)
    if len(h) == 3:
        h = "".join(c * 2 for c in h)          # expand shorthand: abc -> aabbcc
    return "#" + h.upper()


class ColorSpec(BaseModel):
    """One named brand colour."""

    hex: str                                    # normalised to #RRGGBB on load
    roles: list[str]                            # what the guidelines use it for (e.g. "background", "cta")
    source_page: int | None = None              # page of the brand PDF it was taken from (for traceability)

    @field_validator("hex")
    @classmethod
    def _hex(cls, v: str) -> str:
        return normalize_hex(v)


class SubstituteFont(BaseModel):
    """Open-licensed stand-in used when the proprietary brand font files aren't available."""

    family: str
    license: str
    files: dict[int, str]                       # weight → path relative to the brand folder


class FontsSpec(BaseModel):
    """Which font family to use, and which files provide it."""

    brand_family: str                           # the font named in the guidelines
    brand_files_available: bool = False         # True only if we actually hold licensed files for it
    brand_files: dict[int, str] = {}            # weight → path (when available)
    substitute: SubstituteFont | None = None
    weights_by_role: dict[str, int]             # e.g. {"headline": 700, "body": 400}
    source_page: int | None = None

    def active_family(self) -> str:
        """Family name to put in CSS: the brand font if its files exist, otherwise the substitute."""
        if self.brand_files_available:
            return self.brand_family
        if not self.substitute:
            raise ValueError("brand font files are missing and no substitute is declared")
        return self.substitute.family

    def active_files(self) -> dict[int, str]:
        """``weight -> file path`` map matching :meth:`active_family`."""
        return self.brand_files if self.brand_files_available else self.substitute.files  # type: ignore[union-attr]


class Typography(BaseModel):
    """Limits that QC enforces on text styling within a single frame."""

    max_font_sizes_per_frame: int = 3
    max_text_colors_per_frame: int = 2
    headline_max_lines: int = 2                 # preferred maximum
    headline_max_lines_hard: int = 3            # absolute maximum before QC fails it
    min_font_px: int = 10
    min_legal_font_px: int = 8                  # legal/disclaimer text may be smaller
    source_pages: list[int] = []


class LogoPlacement(BaseModel):
    """Where and how large the logo sits in the corner of a banner."""

    variant: Literal["stacked", "horizontal"]
    height_px: int
    width_px: int | None = None                 # None → from the asset's aspect ratio
    margin_right_px: int
    margin_top_px: int
    reason: str | None = None                   # why this size needs its own placement
    source_page: int | None = None


class EndFrameRule(BaseModel):
    """Sizing of the centred logo on the closing frame, as percentages of the canvas."""

    horizontal_width_pct: float                 # logo width for wide (landscape) canvases
    vertical_width_pct: float                   # logo width for tall / square canvases
    max_height_pct: float
    strip_variant: Literal["stacked", "horizontal"] = "horizontal"   # variant to use on thin strips
    strip_ratio: float = 4.0                    # width/height at or above this counts as a strip
    source_page: int | None = None


class Zone(BaseModel):
    """A rectangle (in px) reserved for something other than ad content."""

    width_px: int
    height_px: int


class LogoSpec(BaseModel):
    """All logo-related rules: assets, colour-on-background choice, corner placement and end-frame sizing."""

    variants: dict[str, str]                    # "stacked_green" → relative path
    light_bg_variant: Literal["green", "white"] = "green"   # logo colour to use on light backgrounds
    dark_bg_variant: Literal["green", "white"] = "white"    # logo colour to use on dark backgrounds
    luminance_threshold: float = 0.5            # background luminance below this counts as dark
    corner: LogoPlacement                       # default corner placement
    overrides: dict[str, LogoPlacement] = {}    # per-size placements that replace ``corner``
    end_frame: EndFrameRule
    adinfo_zone: Zone                           # corner area the ad platform overlays with its info icon
    spec_sizes: list[str] = []                  # sizes explicitly covered by the guidelines

    def corner_for(self, size: str) -> LogoPlacement:
        """Placement for ``size``: its override if one exists, otherwise the default corner placement."""
        return self.overrides.get(size, self.corner)


class Rules(BaseModel):
    """Miscellaneous brand rules."""

    mnemonic_allowed: bool = False              # whether the brand's sound/visual mnemonic may be used
    cta_shape: str = "pill"
    horizontal_grid: dict[str, int] = {"visual_pct": 60, "copy_pct": 40}   # split for wide layouts


class Motion(BaseModel):
    """Animation limits for HTML5 banners."""

    max_duration_s: float = 30
    plays: int = 1                              # how many times the animation may loop


class Weight(BaseModel):
    """File-size budget."""

    max_kb: int = 150


class BrandKit(BaseModel):
    """Complete, validated brand guideline set for one brand."""

    brand_id: str
    brand_name: str
    brand_aliases: list[str] = []               # other spellings seen in masters (used by matches_brand)
    sub_brand: str | None = None
    colors: dict[str, ColorSpec]                # name → colour
    text_colors: list[str]                      # names from ``colors`` that may be used for text
    color_tolerance: float = 8                  # max RGB distance for a colour to count as "on palette"
    fonts: FontsSpec
    typography: Typography
    logo: LogoSpec
    rules: Rules = Rules()
    motion: Motion = Motion()
    weight: Weight = Weight()
    asset_provenance: dict[str, str] = {}       # asset → where it came from
    _root: Path = PrivateAttr(default=Path("."))   # brand folder; set by load_brand, resolves relative paths

    @model_validator(mode="after")
    def _references(self) -> "BrandKit":
        """Cross-field checks: text colours must exist, and every logo placement needs both colour variants."""
        missing = [c for c in self.text_colors if c not in self.colors]
        if missing:
            raise ValueError(f"text_colors names unknown colours: {missing}")
        for size, place in {"default": self.logo.corner, **self.logo.overrides}.items():
            for colour in ("green", "white"):
                if f"{place.variant}_{colour}" not in self.logo.variants:
                    raise ValueError(f"logo placement '{size}' needs variant '{place.variant}_{colour}'")
        return self

    def path(self, rel: str) -> Path:
        """Resolve a path stored relative to the brand folder."""
        return self._root / rel

    def palette(self) -> dict[str, tuple[int, int, int]]:
        """All brand colours as ``name -> (r, g, b)``."""
        return {name: hex_to_rgb(c.hex) for name, c in self.colors.items()}

    def text_palette(self) -> dict[str, tuple[int, int, int]]:
        """The subset of :meth:`palette` that is allowed for text."""
        return {n: rgb for n, rgb in self.palette().items() if n in self.text_colors}

    def matches_brand(self, detected: str | None) -> bool:
        """Does a brand name detected in the master refer to this brand? Case-insensitive; checks aliases and
        also accepts the brand name appearing as a whole word inside a longer string."""
        if not detected:
            return False
        d = detected.strip().lower()
        names = {self.brand_name.lower(), *(a.lower() for a in self.brand_aliases)}
        return d in names or self.brand_name.lower() in d.split()

    def logo_asset(self, variant: str, colour: str) -> Path:
        """File path of a logo, e.g. ``logo_asset("stacked", "green")``."""
        return self.path(self.logo.variants[f"{variant}_{colour}"])

    def logo_aspect(self, variant: str) -> float:
        """Width / height ratio of a logo variant (colour variants share proportions, so green is used)."""
        with Image.open(self.logo_asset(variant, "green")) as im:
            return im.width / im.height


def load_brand(brand_id: str) -> BrandKit:
    """Load and validate ``<brands_dir>/<brand_id>/brand_kit.json``.

    Raises:
        FileNotFoundError: If the brand has no kit file.
        pydantic.ValidationError: If the kit is malformed or internally inconsistent.
    """
    root = settings.brands_dir / brand_id
    f = root / "brand_kit.json"
    if not f.exists():
        raise FileNotFoundError(f"unknown brand '{brand_id}' (no {f})")
    kit = BrandKit.model_validate_json(f.read_text())
    kit._root = root                            # remember where assets live
    return kit
