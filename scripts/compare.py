"""Run the same job N times in each mode and report pass rates and cost per approved ad."""
import argparse
import asyncio
import csv
from pathlib import Path

from creative_agent.client import AgentClient


def summarise(mode: str, res: dict) -> dict:
    """Reduce one finished job result to the numbers the comparison needs.

    Args:
        mode: ``"single_prompt"`` or ``"agent"``.
        res: The job result dict returned by the agent server.

    Returns:
        One CSV row: job id and status, number of sizes, how many were approved, how many were approved on the
        first attempt, total cost in USD (``None`` if the model is unpriced) and the cost of the ingest step.
    """
    sizes = res.get("sizes", [])
    by_node = res.get("usage", {}).get("by_node", {})
    return {
        "mode": mode, "job_id": res["job_id"], "status": res["status"], "sizes": len(sizes),
        "approved": sum(s["status"] == "approved" for s in sizes),
        "first_try": sum(s["status"] == "approved" and s["attempts"] == 1 for s in sizes),
        "cost": res.get("usage", {}).get("cost_usd"),
        "ingest_cost": by_node.get("ingest_master") or 0.0,
    }


def table(rows: list[dict]) -> str:
    """Markdown table with one line per mode: first-try and final pass rates, and cost per approved ad
    (with and without the one-off ingest cost, which is the same for both modes)."""
    out = ["| Mode | First-try pass rate | Final pass rate | Cost per approved ad | Excl. ingest |",
           "|---|---|---|---|---|"]
    for mode in ("single_prompt", "agent"):
        r = [x for x in rows if x["mode"] == mode]
        if not r:
            continue
        sizes, approved = sum(x["sizes"] for x in r), sum(x["approved"] for x in r)
        first = sum(x["first_try"] for x in r)
        if any(x["cost"] is None for x in r):
            per, per_ex = "unpriced", "unpriced"
        else:
            total, ingest = sum(x["cost"] for x in r), sum(x["ingest_cost"] for x in r)
            per = f"${total / approved:.4f}" if approved else "n/a"
            per_ex = f"${(total - ingest) / approved:.4f}" if approved else "n/a"
        rate = lambda n: f"{n / sizes:.0%}" if sizes else "n/a"  # noqa: E731
        out.append(f"| {mode} | {rate(first)} | {rate(approved)} | {per} | {per_ex} |")
    return "\n".join(out)


async def main() -> None:
    """Run the job ``--runs`` times per mode, save every run to ``bench/results.csv`` and print the table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()
    c, rows = AgentClient(), []
    for mode in ("single_prompt", "agent"):
        for i in range(args.runs):
            job = await c.submit(args.input_dir, mode)
            await c.wait(job["job_id"], timeout_s=1800)
            rows.append(summarise(mode, await c.result(job["job_id"])))
            print(f"{mode} run {i + 1}: {rows[-1]['approved']}/{rows[-1]['sizes']} approved "
                  f"(cost {rows[-1]['cost']})")

    out = Path("bench/results.csv")
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n" + table(rows))


if __name__ == "__main__":
    asyncio.run(main())
