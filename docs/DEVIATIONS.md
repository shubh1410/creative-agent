# Deviations from `creative-agent-plan.md`

Every change to model fields, cross-module signatures, API routes, MCP tool names, file layouts or issue codes, and every other departure from the plan, is recorded here.

| Task | Change | Reason |
|---|---|---|
| T01 | `pyproject.toml`: added `[build-system]` (hatchling) and `[tool.hatch.build.targets.wheel]`; added `pythonpath = ["tests"]` and `asyncio_default_fixture_loop_scope = "function"` to pytest options | The `src/` layout needs a build backend so `creative_agent` is importable and the `creative-agent`/`creative-mcp` scripts install; `pythonpath` lets tests do `from fakes import ...` |
| T08 | `PER_FRAME` has 17 checks, not the 18 stated in tasks.md T08 acceptance | The plan §12.3 code and T08's own check list both give 17 (`canvas_bounds`, `text_overflow`, `logo_spec`, `logo_variant`, `logo_no_container`, `adinfo_zone`, `brand_colors`, `brand_fonts`, `font_weight_by_role`, `font_size_count`, `text_color_count`, `headline_lines`, `min_font_size`, `copy_fidelity`, `legal_present`, `cta_visible`, `assets_whitelist`). "18" is a miscount; confirmed by the human. `tests/test_checks.py` asserts 17 |
| T11 | `ingest/cache.save` creates the cache directory itself (`mkdir(parents=True, exist_ok=True)`) | The plan's `save` assumed the directory already existed; a cache round-trip test calls it directly |
| T12 | `check_inputs_node` creates `work/` before writing `content.json` | The plan assumed `ingest_master` had already created it; the node is also called on its own in tests |
| T12 | `tests/conftest.py` `job_ctx` factory also exposes `ctx.stages` (recorded `set_stage` calls) and `ctx.runtime` (`SimpleNamespace(context=ctx)`) | Convenience for calling nodes directly, as tasks.md T12 specifies |
| T13 | `plan_node` creates `work/` before writing `plan.json` | Same reason as `check_inputs` |
| T14 | `finalize_html` raises `RuntimeError` if Pillow lacks WebP; `region_luminance` reads pixels through numpy instead of `Image.getdata()` | Required by tasks.md T14 (clear failure without WebP); `getdata()` is deprecated in the installed Pillow and warned on every call |
| T15 | `qc_node` and `finish_node` create `sizes/<size>/` if missing | Both can run without `build_node` having run first (baseline mode, direct tests) |
| T16 | `qc_node` and `finish_node` are reused unchanged by `qc_all_node`; `BASELINE_GRAPH` is built in `graph.py` | As planned; recorded because T15 left `BASELINE_GRAPH = None` |
| T17 | `create_job` names its optional upload parameter `copy_file` with `alias="copy"`; `JobManager._run` builds `JobContext` (and `make_llm`) inside the `try` | The form field is still `copy` (API unchanged), but a parameter called `copy` shadows a pydantic `BaseModel` attribute and warns; building the context inside `try` means a bad `MODEL` string fails the job with `CRASH` instead of leaving it `running` forever |
| T18 | Imports use `mcp.server.mcpserver.MCPServer` and `mcp_types.ToolAnnotations` (snake_case hints) instead of `mcp.server.fastmcp.FastMCP`; server tests also read the resource and call tools through `mcp.call_tool` | The installed mcp is 2.x, where FastMCP was renamed; `from mcp.server.fastmcp` raises ModuleNotFoundError |
| T19 | `make_llm` no longer passes `temperature=0.2` | The default model rejects non-default temperatures (`ValueError: temperature is not supported for claude-sonnet-5-5 at non-default values`), which crashed every job on its first run; the provider default is used instead |
| T19 | `is_transient` matches the exception class and its base classes by name, not only the exact class name | `AnthropicConnectionError` (a subclass of `APIConnectionError`) was not retried |
