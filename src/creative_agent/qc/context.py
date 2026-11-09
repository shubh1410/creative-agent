"""The data every QC check receives."""
import dataclasses
from dataclasses import dataclass

from ..brand import BrandKit
from ..models import Element, FramePlan, SizePlan, dims


@dataclass
class QCContext:
    """Immutable-ish bundle describing the ad under test. ``frame_index`` selects the frame currently being
    checked; use :meth:`at` to get a copy focused on another frame."""

    size: str
    W: int
    H: int
    brand: BrandKit
    plan: SizePlan
    content: dict[str, Element]               # element id -> element (expected copy)
    assets: set[str]                          # ids of product images that should appear
    click_url: str
    budget_floor_hit: bool                    # file-size shrinking already hit its floor (so the model can't fix it)
    frame_index: int = 0

    @classmethod
    def from_state(cls, s) -> "QCContext":
        """Build a context from a ``SizeState``."""
        W, H = dims(s["size"])
        return cls(size=s["size"], W=W, H=H, brand=s["brand"], plan=s["size_plan"], content=s["content"],
                   assets=set(s["assets"]), click_url=s["click_url"], budget_floor_hit=s.get("budget_floor_hit", False))

    def at(self, i: int) -> "QCContext":
        """Copy of this context pointing at frame ``i``."""
        return dataclasses.replace(self, frame_index=i)

    @property
    def frame(self) -> FramePlan:
        """Plan of the frame currently under test."""
        return self.plan.frames[self.frame_index]

    @property
    def is_end(self) -> bool:
        """True if the current frame is the closing logo frame."""
        return self.frame.scene_id == "end"
