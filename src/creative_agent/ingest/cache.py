"""On-disk cache for the (expensive, LLM-driven) master ingestion step.

Ingesting the same master/references/boxes with the same prompts and model always gives the same result, so it is
stored under ``<data_dir>/cache/ingest/<key>/`` and reused by later jobs.
"""
import hashlib
from pathlib import Path

from pydantic import BaseModel

from ..config import settings
from ..models import Issue, LayoutNotes, MasterSpec


class CachedIngest(BaseModel):
    """Everything the ingest step produces and later steps need."""

    spec: MasterSpec
    notes: list[LayoutNotes]                 # layout guidance from reference creatives
    issues: list[Issue]                      # problems found while ingesting (OCR mismatch, text in crops, ...)


def cache_key(files: list[Path], *parts: str) -> str:
    """Short hash identifying one ingest input set.

    Args:
        files: Input files; their *contents* are hashed (not names or timestamps).
        *parts: Extra strings that change the outcome, e.g. the prompt version and model name.
    """
    h = hashlib.sha256()
    for f in files:
        h.update(f.read_bytes())
    for p in parts:
        h.update(p.encode())
    return h.hexdigest()[:16]


def cache_dir(key: str) -> Path:
    """Folder for a cache entry (also holds the extracted frames and crops the spec refers to)."""
    return settings.data_dir / "cache" / "ingest" / key


def load(key: str) -> CachedIngest | None:
    """Return the cached ingest result, or ``None`` on a cache miss."""
    f = cache_dir(key) / "ingest.json"
    return CachedIngest.model_validate_json(f.read_text()) if f.exists() else None


def save(key: str, data: CachedIngest) -> None:
    """Store an ingest result under ``key``."""
    d = cache_dir(key)
    d.mkdir(parents=True, exist_ok=True)
    (d / "ingest.json").write_text(data.model_dump_json(indent=2))
