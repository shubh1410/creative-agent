
from fakes import FakeChatModel

from creative_agent.graph import graph as graph_mod
from creative_agent.graph.graph import BASELINE_GRAPH
from creative_agent.graph.nodes.baseline import AllSizes, SizedHtml, plan_from_html
from creative_agent.graph.state import initial_state
from creative_agent.models import ALL_SIZES, JobRequest, MasterSpec
from creative_agent.render.finalize import FinalizeResult
from test_graph import ScriptedQC, ok, spec_fixture

URL = "https://www.maxis.com.my/"

SAMPLE = """```html
<style>.x{color:red}</style>
<div id="ad">
  <div class="frame" data-scene="s1" data-duration="2200"><p data-el="s1.headline" data-role="headline">RM100 Million Device Sale</p></div>
  <div class="frame" data-scene="s5" data-duration="500"><img src="asset://products_1">
    <p data-el="s5.headline" data-role="cta">Get yours with Maxis today</p>
    <p data-el="s5.legal" data-role="legal">Terms &amp; Conditions Apply</p><p data-el="nope.thing">x</p></div>
  <div class="frame" data-scene="end"></div>
</div>
```"""


def load(tmp_path):
    spec = MasterSpec.model_validate_json(spec_fixture(tmp_path).read_text())
    content = {e.id: e for s in spec.scenes for e in s.elements}
    return spec, content


def test_plan_from_html(tmp_path):
    spec, content = load(tmp_path)
    p = plan_from_html("728x90", SAMPLE, content, spec)
    assert p.size == "728x90" and p.backup_frame == 0
    assert [f.scene_id for f in p.frames] == ["s1", "s5", "end"]
    assert [f.duration_ms for f in p.frames] == [2200, 1000, 3000]      # min 1000; default 3000
    assert p.frames[1].keep == ["s5.headline", "s5.legal"]              # unknown data-el ignored
    assert [f.show_products for f in p.frames] == [False, True, False]
    assert p.frames[0].keep == ["s1.headline"]


def test_plan_from_html_backup_skips_leading_end_and_handles_no_ad(tmp_path):
    spec, content = load(tmp_path)
    html = '<div id="ad"><div class="frame" data-scene="end"></div><div class="frame" data-scene="s1"></div></div>'
    assert plan_from_html("320x50", html, content, spec).backup_frame == 1
    empty = plan_from_html("320x50", "<p>nothing</p>", content, spec)
    assert [f.scene_id for f in empty.frames] == ["s1"] and empty.backup_frame == 0


async def test_baseline_graph_one_build_call_five_results(job_ctx, monkeypatch, data_dir, tmp_path):
    spec_path = spec_fixture(tmp_path)
    builds: list[str] = []

    async def fake_ingest(state, runtime, config):
        return {"master_spec_path": str(spec_path), "layout_notes": [], "issues": []}

    async def fake_finalize(inp, browser, page_sem):
        builds.append(inp.size)
        return FinalizeResult(html=f"<html>{inp.size}</html>")

    qc = ScriptedQC(tmp_path, {"*": [[ok()]]})
    monkeypatch.setattr(graph_mod, "ingest_master_node", fake_ingest)
    monkeypatch.setattr("creative_agent.graph.nodes.baseline.finalize_html", fake_finalize)
    monkeypatch.setattr("creative_agent.graph.nodes.qc.run_qc", qc)
    graph = graph_mod.build_baseline_graph()

    answer = AllSizes(ads=[SizedHtml(size=s, html=SAMPLE) for s in ALL_SIZES])
    llm = FakeChatModel(structured=[answer])
    ctx = job_ctx(llm=llm)
    req = JobRequest(brand_id="maxis", brief="Sell phones.", click_url=URL, mode="single_prompt")
    state = initial_state("j_test", req, tmp_path / "m.mp4", [])
    # check_inputs runs for real on the spec fixture; it needs the frame/crop files that spec_fixture wrote
    out = await graph.ainvoke(state, context=ctx)

    assert len(llm.calls) == 1 and llm.calls[0].kind == "structured" and llm.calls[0].schema is AllSizes
    assert sorted(out["results"]) == sorted(ALL_SIZES) and sorted(builds) == sorted(ALL_SIZES)
    assert all(r.status == "approved" and r.attempts == 1 for r in out["results"].values())
    assert out["job_status"] == "completed"
    assert all(n == 1 for n in qc.calls.values())                        # QC once per size, no retries
    for s in ALL_SIZES:
        d = ctx.job_dir / "sizes" / s
        assert (d / "index.html").exists() and (d / f"{s}.zip").exists() and (d / "backup.jpg").exists()
    assert (ctx.job_dir / "result.json").exists() and (ctx.job_dir / "contact_sheet.png").exists()
    prompt = llm.calls[0].messages[1].content
    assert "Brand kit (JSON)" in prompt and "CTA element: s5.headline" in prompt


async def test_baseline_wrong_size_set_is_repaired(job_ctx, monkeypatch, data_dir, tmp_path):
    spec_path = spec_fixture(tmp_path)

    async def fake_ingest(state, runtime, config):
        return {"master_spec_path": str(spec_path), "layout_notes": [], "issues": []}

    async def fake_finalize(inp, browser, page_sem):
        return FinalizeResult(html="<html></html>")

    monkeypatch.setattr(graph_mod, "ingest_master_node", fake_ingest)
    monkeypatch.setattr("creative_agent.graph.nodes.baseline.finalize_html", fake_finalize)
    monkeypatch.setattr("creative_agent.graph.nodes.qc.run_qc", ScriptedQC(tmp_path, {"*": [[ok()]]}))
    short = AllSizes(ads=[SizedHtml(size="300x250", html=SAMPLE)])
    full = AllSizes(ads=[SizedHtml(size=s, html=SAMPLE) for s in ALL_SIZES])
    llm = FakeChatModel(structured=[short, full])
    ctx = job_ctx(llm=llm)
    req = JobRequest(brand_id="maxis", brief="b", click_url=URL, mode="single_prompt")
    out = await graph_mod.build_baseline_graph().ainvoke(initial_state("j_test", req, tmp_path / "m.mp4", []), context=ctx)
    assert len(llm.calls) == 2 and len(out["results"]) == 5
    assert "return exactly one ad for each of" in llm.calls[1].messages[-1].content


def test_baseline_graph_nodes():
    m = BASELINE_GRAPH.get_graph().draw_mermaid()
    for n in ("ingest_master", "check_inputs", "single_prompt_build", "qc_all", "collect"):
        assert n in m
