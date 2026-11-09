"""Optional OCR cross-check used to catch mistakes in the LLM's text extraction."""
from pathlib import Path

from PIL import Image, ImageOps
from rapidfuzz import fuzz


def ocr_text(path: Path) -> str | None:
    """Best of two passes (light text made dark, then as-is). None if Tesseract isn't installed."""
    try:
        import pytesseract                    # imported lazily: Tesseract is an optional dependency
        gray = ImageOps.grayscale(Image.open(path))
        # Tesseract reads dark-on-light best, so the first pass turns bright pixels (white text) black and the
        # rest white; the second pass leaves the image alone for ordinary dark text.
        passes = [gray.point(lambda v: 0 if v > 200 else 255), gray]
        return max((pytesseract.image_to_string(p) for p in passes), key=len)   # longer output = better read
    except Exception:
        # Missing binary, unreadable image, ... OCR is only a sanity check, so never fail the job over it.
        return None


def similarity(extracted: str, ocr: str) -> float:
    """1.0 when every extracted word appears in the OCR text (extra OCR words don't count against)."""
    return fuzz.token_set_ratio(extracted.lower(), ocr.lower()) / 100
