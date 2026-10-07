"""Retrieval building blocks: tokenizer, BM25 index, metadata filters, re-ranker, answer extraction.

BM25 stands in for embeddings so the project runs with zero API keys. To use vectors,
implement the same `search()` signature on top of pgvector (see guide, step 10).
"""
import math
import re
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar

from .kb import KB, QUERY_EXPANSIONS, WEB, Doc

STOP = set("the a an is are of to in on for and or with what how do does can i my me you it at be by from this that which who when where much many".split())
ROLE_LEVEL = {"public": 0, "staff": 1}


def stem(w):
    return w[:6] if len(w) > 6 else re.sub(r"s$", "", w)


def tokenize(text):
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


class BM25Index:
    def __init__(self, docs):
        self.docs = list(docs)
        self.tokens = [tokenize(d.title + " " + d.text) for d in self.docs]
        self.df = Counter(w for t in self.tokens for w in set(t))
        self.n = len(self.docs)
        self.avg = sum(len(t) for t in self.tokens) / max(self.n, 1)

    def search(self, query, role="public", modality=None, k=10):
        """Return [(Doc, score)] best-first. ACL + modality filters run BEFORE scoring results are returned."""
        q = set(tokenize(query))
        out = []
        for doc, toks in zip(self.docs, self.tokens):
            if ROLE_LEVEL.get(role, 0) < ROLE_LEVEL[doc.acl]:
                continue  # security: never retrieve what the caller may not see
            if modality and doc.modality != modality:
                continue
            tf = Counter(toks)
            score = 0.0
            for w in q:
                f = tf.get(w, 0)
                if not f:
                    continue
                n = self.df[w]
                idf = math.log(1 + (self.n - n + 0.5) / (n + 0.5))
                score += idf * f * 2.5 / (f + 1.5 * (0.25 + 0.75 * len(toks) / self.avg))
            if score > 0:
                out.append((doc, score))
        return sorted(out, key=lambda x: -x[1])[:k]


SCOPE = ContextVar("rag_scope", default="all")   # which data a search may use: all | demo | uploads


@contextmanager
def use_scope(scope):
    """Limit every KB search inside the `with` block to the demo documents, the uploaded files, or both."""
    token = SCOPE.set(scope if scope in ("all", "demo", "uploads") else "all")
    try:
        yield
    finally:
        SCOPE.reset(token)


class DynamicIndex:
    """Drop-in for the old static KB_INDEX: demo documents plus uploaded files, rebuilt when uploads change."""

    def __init__(self, static_docs):
        self.static = list(static_docs)
        self._version = None
        self._indexes = {}

    def _current_version(self):
        from django.db import DatabaseError
        from django.db.models import Count, Max
        from .models import Chunk
        try:
            agg = Chunk.objects.aggregate(n=Count("id"), m=Max("id"))
            return (agg["n"], agg["m"])
        except DatabaseError:  # tables not created yet (before migrate): demo documents only
            return (0, None)

    def _upload_docs(self):
        from django.db import DatabaseError
        from .models import Chunk
        try:
            return [Doc(f"u{c.id}", c.title, c.text, modality=c.upload.kind, acl=c.upload.acl, source="upload")
                    for c in Chunk.objects.select_related("upload").iterator()]
        except DatabaseError:
            return []

    def _get(self):
        from django.conf import settings
        version = (self._current_version(), settings.USE_DEMO_KB)
        if version != self._version:
            self._version, self._indexes = version, {}
        scope = SCOPE.get()
        if scope not in self._indexes:
            demo = list(self.static) if settings.USE_DEMO_KB and scope in ("all", "demo") else []
            uploads = self._upload_docs() if scope in ("all", "uploads") else []
            self._indexes[scope] = BM25Index(demo + uploads)
        return self._indexes[scope]

    def search(self, *args, **kwargs):
        return self._get().search(*args, **kwargs)


KB_INDEX = DynamicIndex(KB)
WEB_INDEX = BM25Index(WEB)


def expand_query(q):
    extra = [e for pat, e in QUERY_EXPANSIONS if re.search(pat, q.lower())]
    return (q + " " + " ".join(extra)).strip()


def rerank(query, hits):
    """Heuristic re-ranker (stand-in for a cross-encoder like bge-reranker or Cohere Rerank)."""
    q = set(tokenize(query))
    top = max((s for _, s in hits), default=1) or 1
    scored = []
    for doc, s in hits:
        title_overlap = len(q & set(tokenize(doc.title))) / max(len(q), 1)
        coverage = len(q & set(tokenize(doc.text + " " + doc.title))) / max(len(q), 1)
        scored.append((doc, s / top + 0.6 * title_overlap + 0.4 * coverage))
    return sorted(scored, key=lambda x: -x[1])


def sentences(text):
    protected = text.replace("Dr.", "Dr\u2024")  # don't split after the abbreviation "Dr."
    return [s.strip().replace("\u2024", ".") for s in re.findall(r"[^.!?]+[.!?]?", protected) if s.strip()]


def extract_answer(query, docs, max_sentences=3):
    """Extractive 'generator': pick the sentences that share the most words with the question."""
    q = set(tokenize(query))
    cands = []
    for i, d in enumerate(docs):
        for s in sentences(d.text):
            overlap = len(q & set(tokenize(s)))
            if overlap:
                cands.append((overlap, i, s, d.id))
    cands.sort(key=lambda c: (-c[0], c[1]))
    seen, picked = set(), []
    for _, _, s, did in cands:
        if s not in seen:
            seen.add(s)
            picked.append((s, did))
    return picked[:max_sentences]
