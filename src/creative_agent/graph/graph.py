"""LangGraph wiring for the pipeline.

Three graphs are built here:

* ``SIZE_GRAPH``     per-size loop:  build -> qc -> (build again | finish)
* ``JOB_GRAPH``      the full agent: ingest_master -> check_inputs -> plan -> [size_worker x N in parallel] -> collect
* ``BASELINE_GRAPH`` one-shot comparison: ingest_master -> check_inputs -> single_prompt_build -> qc_all -> collect
"""
import httpx
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy

from .nodes.baseline import qc_all_node, single_prompt_build_node
from .nodes.build import build_node
from .nodes.check_inputs import check_inputs_node
from .nodes.collect import collect_node
from .nodes.dispatch import dispatch_sizes
from .nodes.finish import finish_node
from .nodes.ingest_master import ingest_master_node
from .nodes.plan import plan_node
from .nodes.qc import qc_node, route_after_qc
from .state import JobContext, JobState, SizeState

# Class names of provider errors worth retrying (rate limits, timeouts, overloaded servers).
TRANSIENT = {"RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError", "OverloadedError"}


def is_transient(exc: Exception) -> bool:
    """True if ``exc`` is a temporary failure (rate limit, timeout, connection problem) that may succeed on retry.

    Matching is by class name over the whole MRO, so it works for any provider SDK without importing them.
    """
    names = {c.__name__ for c in type(exc).__mro__}          # providers subclass the SDK errors, e.g. AnthropicConnectionError
    return bool(names & TRANSIENT) or isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


# Applied to nodes that call an LLM: up to 3 tries with 2 s, then 4 s back-off. Non-transient errors fail at once.
LLM_RETRY = RetryPolicy(max_attempts=3, initial_interval=2.0, backoff_factor=2.0, retry_on=is_transient)


def build_size_graph():
    """Per-size subgraph: build the HTML, run QC, and loop back to build while QC fails and retries remain."""
    g = StateGraph(SizeState, context_schema=JobContext)
    g.add_node("build", build_node, retry_policy=LLM_RETRY)
    g.add_node("qc", qc_node)
    g.add_node("finish", finish_node)
    g.add_edge(START, "build")
    g.add_edge("build", "qc")
    # route_after_qc decides between another build attempt and finishing (pass, unfixable, or out of attempts).
    g.add_conditional_edges("qc", route_after_qc, {"build": "build", "finish": "finish"})
    g.add_edge("finish", END)
    return g.compile()


SIZE_GRAPH = build_size_graph()


async def size_worker_node(state: SizeState, runtime: Runtime[JobContext], config: RunnableConfig):
    """Runs the subgraph and returns only `results`. Adding the subgraph directly as a node would merge
    its whole final state back, and five parallel branches writing `brand` in one step is an error."""
    final = await SIZE_GRAPH.ainvoke(state, config=config, context=runtime.context)
    return {"results": {state["size"]: final["result"]}}


def build_job_graph():
    """The full agent graph. ``dispatch_sizes`` fans out one ``size_worker`` per planned size (map-reduce);
    ``collect`` runs once all workers have returned."""
    g = StateGraph(JobState, context_schema=JobContext)
    g.add_node("ingest_master", ingest_master_node, retry_policy=LLM_RETRY)
    g.add_node("check_inputs", check_inputs_node)
    g.add_node("plan", plan_node, retry_policy=LLM_RETRY)
    g.add_node("size_worker", size_worker_node)
    g.add_node("collect", collect_node)
    g.add_edge(START, "ingest_master")
    g.add_edge("ingest_master", "check_inputs")
    g.add_edge("check_inputs", "plan")
    g.add_conditional_edges("plan", dispatch_sizes, ["size_worker"])
    g.add_edge("size_worker", "collect")
    g.add_edge("collect", END)
    return g.compile()


def build_baseline_graph():
    """Single-prompt baseline: same ingest/input checks, but one LLM call writes every size and QC runs once,
    with no planning and no repair loop. Used to measure what the agent design adds."""
    g = StateGraph(JobState, context_schema=JobContext)
    g.add_node("ingest_master", ingest_master_node, retry_policy=LLM_RETRY)
    g.add_node("check_inputs", check_inputs_node)
    g.add_node("single_prompt_build", single_prompt_build_node, retry_policy=LLM_RETRY)
    g.add_node("qc_all", qc_all_node)
    g.add_node("collect", collect_node)
    for a, b in [(START, "ingest_master"), ("ingest_master", "check_inputs"), ("check_inputs", "single_prompt_build"),
                 ("single_prompt_build", "qc_all"), ("qc_all", "collect"), ("collect", END)]:
        g.add_edge(a, b)
    return g.compile()


JOB_GRAPH = build_job_graph()
BASELINE_GRAPH = build_baseline_graph()
