"""Make transparent logo PNGs in exact brand colours from the input crops (assumption A3).

The boxes are measured once by hand: the stacked green logo from the video's end frame, the
other three from the image on PDF page 9. Recolouring to the exact hex fixes the colour shift
caused by video compression (#49E10C measured vs #40C706).

Usage:
  uv run python scripts/prepare_logos.py                  # write the four logos
  uv run python scripts/prepare_logos.py --render-page 9  # save tmp/page_9.png (3x) to measure boxes
  uv run python scripts/prepare_logos.py --preview        # write tmp/logo_preview.png (white / grey / black)"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pypdfium2 as pdfium
from PIL import Image

OUT = Path("config/brands/maxis/logos")        # where the finished logo PNGs are written
TMP = Path("tmp")                              # scratch output: page renders and the preview sheet
GREEN, WHITE = (0x40, 0xC7, 0x06), (255, 255, 255)   # exact brand colours (RGB)
VIDEO = Path("inputs/maxis-merdeka/master.mp4")      # source of the stacked green logo
PDF = Path("config/brands/maxis/source/Maxis_Brand_CI_-_Jan2026.pdf")   # source of the other three

# name: (source, box (x0, y0, x1, y1) in source pixels, background, colour)
# pdf@9 boxes are in the 3x page render (2160x1215); measured with pixel analysis inside the panel borders.
# Each box has a few pixels of margin around the logo; key_logo() crops to the opaque pixels.
CROPS = {
    "stacked-green":    ("video@18.5", (380, 757, 697, 1161), "white", GREEN),
    "stacked-white":    ("pdf@9",      (1735, 427, 1891, 626), "black", WHITE),
    "horizontal-green": ("pdf@9",      (1102, 901, 1353, 983), "white", GREEN),
    "horizontal-white": ("pdf@9",      (1687, 902, 1938, 985), "black", WHITE),
}


def video_frame(t: float) -> Image.Image:
    """Grab the frame at ``t`` seconds from the master video with ffmpeg and return it as an RGB image."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "f.png"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-ss", str(t), "-i", str(VIDEO),
                        "-frames:v", "1", str(out), "-y"], check=True)
        return Image.open(out).convert("RGB")


def pdf_page(page_no: int, scale: float = 3.0) -> Image.Image:
    """Render a brand PDF page (1-based) at ``scale`` times its natural size. The ``CROPS`` boxes assume 3x."""
    return pdfium.PdfDocument(str(PDF))[page_no - 1].render(scale=scale).to_pil().convert("RGB")


def key_logo(img: Image.Image, background: str, colour: tuple[int, int, int]) -> Image.Image:
    """Logo on a plain white or black background → transparent PNG filled with one exact colour."""
    rgb = np.asarray(img).astype(np.float32)
    # distance from the background gives the opacity: darkest channel on white, brightest channel on black
    alpha = 255 - rgb.min(axis=2) if background == "white" else rgb.max(axis=2)
    alpha = np.clip((alpha - 20) * 255 / 215, 0, 255)          # drop compression noise near the background
    out = np.zeros((*alpha.shape, 4), np.uint8)
    out[..., :3] = colour
    out[..., 3] = alpha.astype(np.uint8)
    im = Image.fromarray(out, "RGBA")
    return im.crop(im.getbbox())


def make_logos() -> None:
    """Cut each box in ``CROPS`` out of its source, make the background transparent, recolour it to the exact
    brand colour and save it as ``OUT/<name>.png``. Boxes still set to ``(0, 0, 0, 0)`` are skipped."""
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (source, box, background, colour) in CROPS.items():
        if box == (0, 0, 0, 0):
            print(f"skip {name}: box not measured yet")
            continue
        kind, arg = source.split("@")
        src = video_frame(float(arg)) if kind == "video" else pdf_page(int(arg))
        key_logo(src.crop(box), background, colour).save(OUT / f"{name}.png")
        print(f"wrote {OUT / name}.png")


def render_page(n: int) -> None:
    """Save PDF page ``n`` as ``tmp/page_<n>.png`` so crop boxes can be measured in an image viewer."""
    TMP.mkdir(exist_ok=True)
    path = TMP / f"page_{n}.png"
    pdf_page(n).save(path)
    print(f"wrote {path}")


def preview() -> None:
    """Each logo over white, grey and black, so halos and rough edges are visible."""
    names = list(CROPS)
    logos = [Image.open(OUT / f"{n}.png").convert("RGBA") for n in names]
    pad, cell_h = 20, max(im.height for im in logos) + 40
    backgrounds = [(255, 255, 255), (128, 128, 128), (0, 0, 0)]
    width = sum(im.width + pad for im in logos) + pad
    sheet = Image.new("RGB", (width, cell_h * len(backgrounds)), (255, 255, 255))
    for row, bg in enumerate(backgrounds):
        sheet.paste(Image.new("RGB", (width, cell_h), bg), (0, row * cell_h))
        x = pad
        for im in logos:
            sheet.paste(im, (x, row * cell_h + (cell_h - im.height) // 2), im)
            x += im.width + pad
    TMP.mkdir(exist_ok=True)
    sheet.save(TMP / "logo_preview.png")
    print(f"wrote {TMP / 'logo_preview.png'} (logos left to right: {', '.join(names)})")


def main() -> None:
    """Command line entry point: render a page, write the preview sheet, or (by default) write the logos."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-page", type=int, metavar="N", help="save PDF page N at 3x to tmp/page_N.png")
    ap.add_argument("--preview", action="store_true", help="write tmp/logo_preview.png")
    args = ap.parse_args()
    if args.render_page:
        render_page(args.render_page)
    elif args.preview:
        preview()
    else:
        make_logos()


if __name__ == "__main__":
    main()
