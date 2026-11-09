"""Video helpers (ffmpeg): probing, cheap low-res sampling, scene segmentation and frame extraction."""
import subprocess
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import numpy as np

# Bundled ffmpeg binary, so no system installation is needed.
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


@dataclass
class Segment:
    """A time span (seconds) during which the picture is stable, i.e. one scene."""

    start: float
    end: float

    @property
    def mid(self) -> float:
        """Midpoint; the frame here is the most representative one (furthest from the cross-fades)."""
        return (self.start + self.end) / 2


def probe(path: Path) -> dict:
    """Read the video's metadata (``size``, ``fps``, ``duration``, ...) without decoding frames."""
    gen = imageio_ffmpeg.read_frames(str(path))
    meta = next(gen)                              # {"size": (w, h), "fps": ..., "duration": ...}
    gen.close()
    return meta


def sample_gray(path: Path, fps: int, W: int, H: int, width: int = 64) -> np.ndarray:
    """Decode the video to a small greyscale clip for scene-change detection.

    Args:
        path: Video file.
        fps: Samples per second to keep.
        W, H: Original frame size (used to keep the aspect ratio).
        width: Width of the downscaled frames in pixels.

    Returns:
        int16 array of shape ``(n_frames, h, width)``. int16 so later frame differences don't overflow.
    """
    h = max(2, round(width * H / W / 2) * 2)      # even height, as ffmpeg's scaler prefers
    cmd = [FFMPEG, "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale={width}:{h},format=gray",
           "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, width).astype(np.int16)


def stable_segments(frames: np.ndarray, fps: int, threshold: float = 4.0, min_len_s: float = 1.0) -> list[Segment]:
    """Stretches where consecutive samples barely change. Crossfades and motion break a stretch.

    Args:
        frames: Output of :func:`sample_gray`.
        fps: The sampling rate used to produce ``frames``.
        threshold: Mean absolute grey-level change between consecutive samples below which the picture counts
            as "still".
        min_len_s: Stretches shorter than this are ignored (transitions, flicker).
    """
    # diffs[i] = how much sample i differs from sample i-1 (the first has no predecessor, so 0).
    diffs = np.concatenate([[0.0], np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))])
    spans, start = [], None
    for i, d in enumerate(diffs):
        if d < threshold:
            start = i if start is None else start
        elif start is not None:
            spans.append((start, i))
            start = None
    if start is not None:                         # video ended while still stable
        spans.append((start, len(diffs)))
    return [Segment(a / fps, b / fps) for a, b in spans if (b - a) / fps >= min_len_s]


def mean_gap(segments: list[Segment]) -> float | None:
    """Average time between one scene ending and the next starting, i.e. the typical transition length.
    ``None`` if there is only one segment."""
    gaps = [b.start - a.end for a, b in zip(segments, segments[1:])]
    return round(sum(gaps) / len(gaps), 2) if gaps else None


def extract_frame(path: Path, t: float, out: Path) -> None:
    """Save the frame at ``t`` seconds as an image. ``-ss`` before ``-i`` makes ffmpeg seek quickly."""
    subprocess.run([FFMPEG, "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1", str(out), "-y"],
                   check=True)
