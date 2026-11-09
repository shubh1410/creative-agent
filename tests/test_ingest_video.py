from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from creative_agent.ingest import video

MASTER = Path("inputs/maxis-merdeka/master.mp4")


def test_stable_segments_synthetic():
    still = np.zeros((8, 10, 10), np.int16)              # 2 s still
    fade = np.linspace(0, 200, 6)[:, None, None] * np.ones((6, 10, 10))
    frames = np.concatenate([still, fade.astype(np.int16), still + 200])
    segs = video.stable_segments(frames, fps=4)
    assert len(segs) == 2 and segs[0].start == 0.0


def test_mean_gap():
    segs = [video.Segment(0, 2), video.Segment(3, 5), video.Segment(6, 8)]
    assert video.mean_gap(segs) == 1.0
    assert video.mean_gap(segs[:1]) is None


@pytest.mark.skipif(not MASTER.exists(), reason="master video not present")
def test_master_scenes(tmp_path):
    meta = video.probe(MASTER)
    segs = video.stable_segments(video.sample_gray(MASTER, 4, *meta["size"]), 4)
    starts = [s.start for s in segs]
    assert starts == pytest.approx([0.5, 3.75, 7.0, 10.25, 13.75, 17.25], abs=0.3)
    for i, s in enumerate(segs):
        out = tmp_path / f"s{i}.png"
        video.extract_frame(MASTER, s.mid, out)
        with Image.open(out) as im:
            assert im.size == (1080, 1920)
