"""Guardrails: input checks (security) and output checks (hallucination + leakage).

Layered on purpose: regex is a fast first wall. In production add a classifier
(e.g. Llama Guard / AWS Bedrock Guardrails) as a second layer.
"""
import re
from dataclasses import dataclass, field

from django.conf import settings

from .retrieval import tokenize

INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above) (instructions|rules|prompts?)",
    r"disregard (the )?(system|previous) (prompt|instructions)",
    r"reveal (your |the )?(system prompt|instructions|hidden)",
    r"you are now (dan|in developer mode|unrestricted)",
    r"jailbreak|pretend (you have|there are) no (rules|restrictions)",
]
PII_PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "card": r"\b(?:\d[ -]?){13,16}\b",
    "phone": r"\+?\d[\d\s().-]{8,}\d",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
}
OFF_TOPIC = r"\b(write (me )?(code|malware)|exploit|bomb|hack into)\b"


@dataclass
class GuardResult:
    ok: bool = True
    text: str = ""
    reason: str = ""
    findings: list = field(default_factory=list)


def check_input(text):
    findings = []
    text = (text or "").strip()
    if not text:
        return GuardResult(False, text, "Empty question.", ["empty"])
    if len(text) > settings.MAX_QUESTION_CHARS:
        return GuardResult(False, text[:80], f"Question longer than {settings.MAX_QUESTION_CHARS} characters.", ["too_long"])
    low = text.lower()
    for p in INJECTION_PATTERNS:
        if re.search(p, low):
            return GuardResult(False, text, "Prompt-injection attempt detected.", ["prompt_injection"])
    if re.search(OFF_TOPIC, low):
        return GuardResult(False, text, "Out of scope for this assistant.", ["off_topic"])
    for name, p in PII_PATTERNS.items():  # redact instead of block: keep the user moving, keep PII out of logs
        if re.search(p, text):
            text = re.sub(p, f"[{name} removed]", text)
            findings.append(f"pii_{name}")
    return GuardResult(True, text, "", findings)


def groundedness(answer_sentences, source_docs):
    """Share of answer sentences whose words are found in the retrieved sources (0..1)."""
    if not answer_sentences:
        return 0.0
    src = set()
    for d in source_docs:
        src |= set(tokenize(d.title + " " + d.text))
    good = 0
    for s in answer_sentences:
        t = set(tokenize(s))
        if t and len(t & src) / len(t) >= 0.6:
            good += 1
    return good / len(answer_sentences)


def check_output(answer, source_docs, threshold=0.5):
    from .retrieval import sentences
    sents = sentences(answer)
    score = groundedness(sents, source_docs)
    findings = []
    for name, p in PII_PATTERNS.items():
        if re.search(p, answer):
            answer = re.sub(p, f"[{name} removed]", answer)
            findings.append(f"pii_{name}")
    if not source_docs:
        return GuardResult(True, answer, "No sources: answer is a fallback message.", findings), 1.0
    if score < threshold:
        return GuardResult(False, "I can't verify that from the knowledge base, so I won't answer.", "Low groundedness.", findings + ["ungrounded"]), score
    return GuardResult(True, answer, "", findings), score
