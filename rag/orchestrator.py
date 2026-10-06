"""LangGraph orchestration: every request flows through the same guarded graph.

        START → input_guard ─┬─(blocked)→ refuse ─────────────→ END
                             └─(ok)────→ run_pipeline → generate → output_guard → END
"""
from typing import Any, TypedDict

from django.conf import settings

from . import guardrails, llm
from .pipelines import PIPELINES, item, step


class RagState(TypedDict, total=False):
    question: str
    clean_question: str
    pipeline: str
    role: str
    blocked: bool
    reason: str
    answer: str
    steps: list
    sources: list
    groundedness: float
    findings: list


def input_guard(state: RagState) -> RagState:
    g = guardrails.check_input(state["question"])
    tone = "ok" if g.ok else "bad"
    detail = [item(g.reason or "passed", "ok" if g.ok else "blocked", tone)] + [item(f, "finding", "warn") for f in g.findings]
    return {"clean_question": g.text, "blocked": not g.ok, "reason": g.reason, "findings": g.findings,
            "steps": [step("Input guardrail (security)", detail)]}


def route_after_input(state: RagState) -> str:
    return "refuse" if state["blocked"] else "run_pipeline"


def refuse(state: RagState) -> RagState:
    return {"answer": f"Request blocked: {state['reason']}", "sources": [], "groundedness": 0.0}


def run_pipeline(state: RagState) -> RagState:
    result = PIPELINES[state["pipeline"]]["fn"](state["clean_question"], state.get("role", "public"))
    return {"answer": result["answer"], "sources": result["sources"], "steps": state["steps"] + result["steps"]}


def generate(state: RagState) -> RagState:
    if not (settings.USE_LLM and state["sources"]):
        return {}
    text = llm.generate(state["clean_question"], state["sources"])
    if not text:
        return {}
    return {"answer": text, "steps": state["steps"] + [step("LLM generation", [item(settings.LLM_MODEL, "llm")])]}


def output_guard(state: RagState) -> RagState:
    g, score = guardrails.check_output(state["answer"], state["sources"])
    tone = "ok" if g.ok else "bad"
    items = [item(f"groundedness = {score:.2f}", "pass" if g.ok else "blocked", tone)] + [item(f, "finding", "warn") for f in g.findings]
    return {"answer": g.text, "groundedness": score, "blocked": not g.ok, "reason": g.reason,
            "findings": state.get("findings", []) + g.findings,
            "steps": state["steps"] + [step("Output guardrail (hallucination check)", items,
                                           "Every answer sentence must be supported by the retrieved sources.")]}


def build_graph():
    from langgraph.graph import END, START, StateGraph
    g = StateGraph(RagState)
    for name, fn in [("input_guard", input_guard), ("refuse", refuse), ("run_pipeline", run_pipeline),
                     ("generate", generate), ("output_guard", output_guard)]:
        g.add_node(name, fn)
    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", route_after_input, {"refuse": "refuse", "run_pipeline": "run_pipeline"})
    g.add_edge("run_pipeline", "generate")
    g.add_edge("generate", "output_guard")
    g.add_edge("refuse", END)
    g.add_edge("output_guard", END)
    return g.compile()


_graph = None


def run(question: str, pipeline: str, role: str = "public") -> dict[str, Any]:
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph.invoke({"question": question, "pipeline": pipeline, "role": role, "steps": []})
