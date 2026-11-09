.PHONY: setup agent mcp demo demo-mismatch test e2e compare

setup:
	uv sync
	uv run playwright install chromium
	test -f .env || cp .env.example .env

agent:
	uv run creative-agent

mcp:
	@uv run creative-mcp

demo:
	uv run python scripts/run_job.py inputs/maxis-merdeka

demo-mismatch:
	uv run python scripts/run_job.py inputs/income-storyboard --expect-error BRAND_MISMATCH

test:
	uv run pytest -q -m "not e2e"

e2e:
	uv run pytest -q -m e2e

compare:
	uv run python scripts/compare.py inputs/maxis-merdeka --runs 5
