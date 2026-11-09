import json
from pathlib import Path

import pytest

from creative_agent.models import ALL_SIZES, BBox, JobRequest, dims
from creative_agent.net import url_status

ROOT = Path(__file__).resolve().parent.parent
JOB_DIRS = ["maxis-merdeka", "income-storyboard"]


def load_job(name: str) -> JobRequest:
    folder = ROOT / "inputs" / name
    data = json.loads((folder / "job.json").read_text())
    data["brief"] = (folder / "brief.md").read_text()
    return JobRequest(**data)


def test_dims():
    assert dims("728x90") == (728, 90)
    assert dims("160x600") == (160, 600)


def test_bbox_inside():
    assert BBox(x=0, y=0, w=10, h=10).inside(10, 10)
    assert not BBox(x=1, y=0, w=10, h=10).inside(10, 10)
    assert not BBox(x=-1, y=0, w=5, h=5).inside(10, 10)
    assert not BBox(x=0, y=0, w=0, h=5).inside(10, 10)


def test_bbox_scaled_and_tuple():
    b = BBox(x=10, y=20, w=30, h=40).scaled(0.5)
    assert b == BBox(x=5, y=10, w=15, h=20)
    assert b.as_tuple() == (5, 10, 20, 30)


@pytest.mark.parametrize("name", JOB_DIRS)
def test_job_json_parses_as_job_request(name):
    job = load_job(name)
    assert job.brand_id == "maxis"
    assert job.sizes == ALL_SIZES
    assert job.mode == "agent"
    assert job.brief.strip()


def test_maxis_job_has_cta_and_income_does_not():
    assert load_job("maxis-merdeka").cta.use_text == "Get yours with Maxis today"
    assert load_job("income-storyboard").cta is None


def test_click_url_gets_trailing_slash():
    job = JobRequest(brand_id="maxis", brief="b", click_url="https://www.maxis.com.my")
    assert str(job.click_url).endswith("/")
    assert str(load_job("maxis-merdeka").click_url) == "https://www.maxis.com.my/"


async def test_url_status_uses_cache_without_network():
    cache = {"https://example.test/": 403}
    assert await url_status("https://example.test/", cache) == 403
