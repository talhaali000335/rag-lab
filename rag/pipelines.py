"""The six RAG architectures. Each function: (question, role) -> {"answer", "steps", "sources"}.

Every step is {"title", "items": [{"text", "badge", "tone"}], "note"} so one template renders them all.
"""
import re

from .kb import EDGES, ENTITIES, RELATION_HINTS
from .retrieval import KB_INDEX, WEB_INDEX, expand_query, extract_answer, rerank, sentences, tokenize

ICON = {"image": "🖼️", "audio": "🎙️", "text": "📄"}
NO_ANSWER = "I could not find that in the knowledge base."


def step(title, items=None, note=""):
    return {"title": title, "items": items or [], "note": note}


def item(text, badge="", tone=""):
    return {"text": text, "badge": badge, "tone": tone}


def doc_item(doc, score=None, tone=""):
    snippet = doc.text if len(doc.text) < 90 else doc.text[:90] + "…"
    return item(f"{ICON.get(doc.modality, '📄')} [{doc.id}] {doc.title}: {snippet}", f"{score:.2f}" if score is not None else doc.source, tone)


def build_answer(question, docs):
    picked = extract_answer(question, docs)
    if not picked:
        return NO_ANSWER, []
    used = {d for _, d in picked}
    return " ".join(s for s, _ in picked), [d for d in docs if d.id in used]


# 1 ─ Multimodal ───────────────────────────────────────────────────────────────
def multimodal(q, role):
    hits = KB_INDEX.search(q, role=role, k=6)
    by_mod = {m: [h for h in hits if h[0].modality == m] for m in ("text", "image", "audio")}
    steps = [step("Embed query once, search every modality",
                  [item(f"{ICON[m]} {m}: {len(v)} hit(s)", str(len(v)), "ok" if v else "warn") for m, v in by_mod.items()],
                  "Images/audio are stored as captions/transcripts here. In production use CLIP/SigLIP or Whisper+embeddings.")]
    top = hits[:4]
    steps.append(step("Fuse results across modalities", [doc_item(d, s) for d, s in top]))
    answer, used = build_answer(q, [d for d, _ in top])
    steps.append(step("Generate grounded answer", [item(f"Cites {', '.join(d.id for d in used) or 'nothing'}")]))
    return {"answer": answer, "steps": steps, "sources": used}


# 2 ─ Advanced (re-ranked / filtered) ──────────────────────────────────────────
def advanced(q, role):
    eq = expand_query(q)
    first = KB_INDEX.search(eq, role=role, k=8)
    steps = [step("Rewrite / expand query", [item(eq)]),
             step("First-pass retrieval (top 8, ACL-filtered)", [doc_item(d, s) for d, s in first])]
    ranked = rerank(q, first)
    steps.append(step("Re-rank by true relevance", [doc_item(d, s, "ok" if i < 3 else "") for i, (d, s) in enumerate(ranked)],
                      "A cross-encoder reads query+chunk together; here a heuristic stands in."))
    kept = [(d, s) for d, s in ranked if s >= 0.6][:3]
    steps.append(step("Filter below threshold, keep top 3", [doc_item(d, s, "ok") for d, s in kept]))
    answer, used = build_answer(q, [d for d, _ in kept])
    return {"answer": answer, "steps": steps, "sources": used}


# 3 ─ Adaptive ─────────────────────────────────────────────────────────────────
def classify(q):
    low = q.lower().strip()
    if re.match(r"^(hi|hello|hey|thanks|thank you)\b", low) or re.search(r"^what is (rag|an? llm)", low):
        return "none"
    hops = len(re.findall(r"\band\b|\bthen\b|\balso\b", low)) + len([e for e, w in ENTITIES.items() if any(x in low for x in w)])
    return "multi" if hops >= 2 else "simple"


def adaptive(q, role):
    route = classify(q)
    steps = [step("Classify query complexity", [item(f"route = {route}", route, "ok")],
                  "none → answer directly · simple → single retrieval · multi → decompose + retrieve per part")]
    if route == "none":
        steps.append(step("Skip retrieval", [item("No knowledge needed — saved latency and cost")]))
        return {"answer": "Hello! Ask me about the clinic: policies, hours, prices or doctors.", "steps": steps, "sources": []}
    if route == "simple":
        hits = KB_INDEX.search(q, role=role, k=3)
        steps.append(step("Single retrieval", [doc_item(d, s) for d, s in hits]))
        answer, used = build_answer(q, [d for d, _ in hits])
        return {"answer": answer, "steps": steps, "sources": used}
    parts = [p.strip() for p in re.split(r"\band\b|\bthen\b|\balso\b|[?,]", q) if len(tokenize(p)) >= 1]
    steps.append(step("Decompose into sub-queries", [item(p) for p in parts]))
    docs, seen = [], set()
    for p in parts:
        hits = KB_INDEX.search(p, role=role, k=2)
        steps.append(step(f"Retrieve for “{p}”", [doc_item(d, s) for d, s in hits]))
        for d, _ in hits:
            if d.id not in seen:
                seen.add(d.id); docs.append(d)
    answer, used = build_answer(q, docs)
    return {"answer": answer, "steps": steps, "sources": used}


# 4 ─ Corrective (CRAG) ────────────────────────────────────────────────────────
CORRECT, AMBIGUOUS = 2.5, 1.0


def corrective(q, role):
    hits = KB_INDEX.search(q, role=role, k=4)
    graded = [(d, s, "correct" if s >= CORRECT else "ambiguous" if s >= AMBIGUOUS else "incorrect") for d, s in hits]
    tone = {"correct": "ok", "ambiguous": "warn", "incorrect": "bad"}
    steps = [step("Retrieve from internal KB", [doc_item(d, s) for d, s in hits]),
             step("Grade each chunk (decision gate)", [item(f"[{d.id}] {g}", f"{s:.2f}", tone[g]) for d, s, g in graded])]
    good = [d for d, _, g in graded if g == "correct"]
    verdict = "correct" if good else "ambiguous" if any(g == "ambiguous" for *_, g in graded) else "incorrect"
    steps.append(step("Decision", [item(f"verdict = {verdict}", verdict, tone[verdict])],
                      "correct → use KB · ambiguous → KB + web · incorrect → web only"))
    docs = list(good)
    if verdict != "correct":
        if verdict == "ambiguous":
            docs += [d for d, _, g in graded if g == "ambiguous"]
        web = WEB_INDEX.search(q, k=2)
        steps.append(step("Fallback: external web search", [doc_item(d, s) for d, s in web] or [item("No web results", "", "bad")]))
        docs += [d for d, _ in web]
    answer, used = build_answer(q, docs)
    return {"answer": answer, "steps": steps, "sources": used}


# 5 ─ Graph RAG ────────────────────────────────────────────────────────────────
def graph(q, role):
    low = q.lower()
    seeds = [e for e, words in ENTITIES.items() if any(w in low for w in words)]
    steps = [step("Link entities in the question", [item(e, "entity", "ok") for e in seeds] or [item("none found", "", "bad")])]
    frontier, seen_edges = set(seeds), []
    for hop in (1, 2):
        new = [e for e in EDGES if (e[0] in frontier or e[2] in frontier) and e not in seen_edges]
        seen_edges += new
        frontier |= {e[0] for e in new} | {e[2] for e in new}
        steps.append(step(f"Traverse graph — hop {hop}", [item(f"{a} —{r}→ {b}") for a, r, b in new] or [item("no new edges")]))

    def relevance(edge):
        return 1 if re.search(RELATION_HINTS.get(edge[1], "$^"), low) else 0
    ranked = sorted(seen_edges, key=lambda e: -relevance(e))
    facts = [e for e in ranked if relevance(e)][:5] or ranked[:3]
    steps.append(step("Keep edges that match the question's relation", [item(f"{a} {r} {b}", "fact", "ok") for a, r, b in facts]))
    answer = ". ".join(f"{a} {r} {b}" for a, r, b in facts) + "." if facts else NO_ANSWER
    from .kb import Doc
    sources = [Doc("g1", "Knowledge graph", answer, source="graph")] if facts else []
    return {"answer": answer, "steps": steps, "sources": sources}


# 6 ─ Agentic ──────────────────────────────────────────────────────────────────
MAX_STEPS = 6


def agentic(q, role):
    plan = [p.strip() for p in re.split(r"\band\b|\?|,", q) if len(tokenize(p)) >= 1][:3]
    steps = [step("Plan", [item(f"{i + 1}. {p}") for i, p in enumerate(plan)], f"Budget: {MAX_STEPS} tool calls.")]
    calls, notes, docs, seen, prices = 0, [], [], set(), []
    for p in plan:
        if calls >= MAX_STEPS:
            break
        low = p.lower()
        tool = "graph_lookup" if any(w in low for ws in ENTITIES.values() for w in ws) and re.search(r"who|where|insur|accept|cost|price|room", low) else "kb_search"
        calls += 1
        if tool == "graph_lookup":
            r = graph(p, role)
            steps.append(step(f"Tool call {calls}: graph_lookup(“{p}”)", [item(r["answer"], "graph", "ok")]))
            notes.append(r["answer"]); docs += r["sources"]
            continue
        hits = KB_INDEX.search(p, role=role, k=2)
        if not hits or hits[0][1] < AMBIGUOUS:
            calls += 1
            web = WEB_INDEX.search(p, k=1)
            steps.append(step(f"Tool call {calls}: web_search(“{p}”) — KB was weak", [doc_item(d, s) for d, s in web] or [item("nothing found", "", "bad")]))
            hits = web
        else:
            steps.append(step(f"Tool call {calls}: kb_search(“{p}”)", [doc_item(d, s) for d, s in hits]))
        for d, _ in hits:
            if d.id not in seen:
                seen.add(d.id); docs.append(d)
        for d, _ in hits:  # collect "<item> N dollars" facts that match this sub-question
            for sent in sentences(d.text):
                for m in re.finditer(r"([A-Za-z][\w-]*) (\d+) dollars", sent):
                    if set(tokenize(m.group(1))) & set(tokenize(p)) and (m.group(1), int(m.group(2))) not in prices:
                        prices.append((m.group(1), int(m.group(2))))
        if hits:
            sents = extract_answer(p, [d for d, _ in hits], 1)
            if sents:
                notes.append(sents[0][0])
    answer = " ".join(dict.fromkeys(notes))
    if re.search(r"total|together|combined|both", q.lower()) and len(prices) >= 2:
        calls += 1
        nums = [n for _, n in prices]
        steps.append(step(f"Tool call {calls}: calculator({' + '.join(map(str, nums))})", [item(f"= {sum(nums)} dollars", "calc", "ok")]))
        answer = ". ".join(f"{w} {n} dollars" for w, n in prices) + f". Total {sum(nums)} dollars."
    steps.append(step("Reflect: did every sub-question get answered?", [item("yes" if notes else "no", "", "ok" if notes else "bad")]))
    return {"answer": answer or NO_ANSWER, "steps": steps, "sources": docs}


PIPELINES = {
    "multimodal": dict(name="Multimodal RAG", fn=multimodal, short="Text + image + audio",
        desc="Retrieves across text, images and audio, then grounds the answer in all of them.",
        flow=["Query", "Search all modalities", "Fuse", "Answer"],
        examples=["Where is the X-ray room?", "What did the patient say about parking?", "How much is whitening?"]),
    "advanced": dict(name="Advanced RAG", fn=advanced, short="Rewrite, re-rank, filter",
        desc="Adds query rewriting, a re-ranker and a relevance filter between retrieval and generation.",
        flow=["Rewrite", "Retrieve", "Re-rank", "Filter", "Answer"],
        examples=["Can I get my money back?", "What happens if I cancel late?", "What are your opening hours?"]),
    "adaptive": dict(name="Adaptive RAG", fn=adaptive, short="Routes by complexity",
        desc="Decides at runtime: skip retrieval, retrieve once, or decompose a multi-part question.",
        flow=["Classify", "Route", "Retrieve 0/1/N", "Answer"],
        examples=["Hello!", "What is the refund window?", "Who does checkups and which insurance do they accept?"]),
    "corrective": dict(name="Corrective RAG (CRAG)", fn=corrective, short="Grade, then web fallback",
        desc="Grades retrieved chunks and falls back to web search when the knowledge base is not good enough.",
        flow=["Retrieve", "Grade", "Decide", "Web fallback", "Answer"],
        examples=["How long does a root canal take?", "What should I avoid after whitening?", "What is the refund policy?"]),
    "graph": dict(name="Graph RAG", fn=graph, short="Entities & relationships",
        desc="Walks a knowledge graph of entities and relations to answer multi-hop questions.",
        flow=["Link entities", "Traverse hops", "Pick edges", "Answer"],
        examples=["Which insurance does Dr. Rivera accept?", "How much is the X-ray at Dr. Rivera's branch?", "Where does Dr. Rivera work?"]),
    "agentic": dict(name="Agentic RAG", fn=agentic, short="Plan, tools, reflect",
        desc="An agent plans, calls tools (KB, graph, web, calculator) in a bounded loop and reflects before answering.",
        flow=["Plan", "Tool loop", "Reflect", "Answer"],
        examples=["How much is a checkup and an X-ray in total?", "Who performs checkups and how long does a root canal take?"]),
}
