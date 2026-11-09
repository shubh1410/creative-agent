"""Fan-out step: turns the plan into one parallel ``size_worker`` task per banner size."""
from langgraph.types import Send

from ...models import dims
from ..state import JobState
from .common import load_spec


def aspect_of(size: str) -> str:
    """Classify a size by shape: 'landscape' (width >= 2x height), 'portrait' (width <= 0.6x height) else 'square'."""
    w, h = dims(size)
    return "landscape" if w / h >= 2 else ("portrait" if w / h <= 0.6 else "square")


def dispatch_sizes(state: JobState) -> list[Send]:
    """Conditional-edge function: returns one ``Send`` per planned size, each carrying only the data that
    size's worker needs (as a fresh ``SizeState``)."""
    spec = load_spec(state)
    content = {e.id: e for e in state["content"]}
    notes = {n.aspect: n.notes for n in state.get("layout_notes", [])}
    master_aspect = "portrait" if spec.height > spec.width else "landscape"
    sends = []
    for sp in state["plan"].sizes:
        aspect = aspect_of(sp.size)
        # Prefer a reference layout for this aspect; otherwise fall back to a generic hint.
        note = notes.get(aspect) or (
            "Follow the master's own composition: copy at the top, products in the lower band."
            if aspect == master_aspect else "No reference for this aspect; follow the size guidance.")
        # Only hand over product crops for scenes where this size actually shows products.
        used = {spec.scene(f.scene_id).product_asset for f in sp.frames if f.show_products} - {None}
        sends.append(Send("size_worker", {
            "job_id": state["job_id"], "size": sp.size, "size_plan": sp, "brand": state["brand"],
            "content": content, "assets": {a: spec.assets[a] for a in used}, "layout_note": note,
            "scene_assets": {s.id: s.product_asset for s in spec.scenes if s.product_asset},
            "click_url": str(state["request"].click_url), "attempt": 0,
        }))
    return sends
