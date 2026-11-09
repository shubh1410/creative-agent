"""Submit an input folder to the agent server, print progress and the result."""
import argparse
import asyncio
import json
import sys

from creative_agent.client import AgentClient, AgentError, AgentUnavailable


async def main() -> int:
    """Submit the job, poll its status every 2 s until it ends, then print the result.

    Returns:
        The process exit code: 0 on success, 1 if the job did not finish as expected, 2 if the job could not
        be submitted. With ``--expect-error CODE`` the job is expected to fail with that error code, and the
        exit code is 0 only when it does (used to test the input checks).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("--mode", choices=["agent", "single_prompt"])
    ap.add_argument("--expect-error")
    args = ap.parse_args()

    c = AgentClient()
    try:
        job = await c.submit(args.input_dir, args.mode)
    except (AgentUnavailable, AgentError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"job {job['job_id']} started")
    last = ""
    while True:
        st = await c.status(job["job_id"])
        if st["stage"] != last:
            print(f"  {st['stage']}")
            last = st["stage"]
        if st["status"] in ("completed", "completed_with_failures", "failed"):
            break
        await asyncio.sleep(2)
    res = await c.result(job["job_id"])
    print(json.dumps({k: res[k] for k in ("status", "error_code", "error") if res.get(k)}, indent=2))
    for s in res.get("sizes", []):
        print(f"  {s['size']:>8}  {s['status']:<9} attempts={s['attempts']}  failed={s['failed_checks']}")
    for i in res.get("issues", []):
        print(f"  [{i['severity']}] {i['code']}: {i['message']}")
    if args.expect_error:
        return 0 if res.get("error_code") == args.expect_error else 1
    return 0 if res["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
