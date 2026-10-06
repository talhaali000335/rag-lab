import logging
import time

from . import orchestrator
from .models import QueryLog

log = logging.getLogger(__name__)


def ask(question, pipeline, role="public", save=True):
    start = time.perf_counter()
    state = orchestrator.run(question, pipeline, role)
    latency = int((time.perf_counter() - start) * 1000)
    state["latency_ms"] = latency
    sources = state.get("sources", [])
    if save:
        QueryLog.objects.create(
            pipeline=pipeline, role=role, question=state.get("clean_question", question)[:500], answer=state.get("answer", ""),
            blocked=state.get("blocked", False), findings=",".join(state.get("findings", []))[:200],
            groundedness=state.get("groundedness", 0), latency_ms=latency, source_ids=",".join(d.id for d in sources)[:100])
    log.info("rag pipeline=%s blocked=%s grounded=%.2f latency_ms=%d", pipeline, state.get("blocked"), state.get("groundedness", 0), latency)
    return state
