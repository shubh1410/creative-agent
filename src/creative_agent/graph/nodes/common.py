"""Helpers shared by several graph nodes."""
from pathlib import Path

from ...models import MasterSpec


def load_spec(state) -> MasterSpec:
    """Load the MasterSpec JSON that ``ingest_master`` wrote (the state stores its path, not the object)."""
    return MasterSpec.model_validate_json(Path(state["master_spec_path"]).read_text())


def norm_text(s: str | None) -> str:
    """Normalise text for comparison: straighten curly quotes, turn en-dashes/non-breaking spaces into plain
    ones, lower-case and collapse whitespace. ``None`` becomes an empty string."""
    s = (s or "").replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return " ".join(s.replace("\u2013", "-").replace("\u00a0", " ").lower().split())
