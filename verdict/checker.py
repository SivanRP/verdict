"""Claim-level grounding checks.

For each claim in an AI-generated answer, find the best supporting passage in
the source documents, then run a set of targeted checks. Each check targets a
specific way LLMs get legal text wrong:

    coverage     the claim talks about things the sources never mention
    numbers      a deadline, amount, or percentage that isn't in the source
    negation     the claim flips "does not" into "does", or the reverse
    modality     "must" turned into "may", or "may" into "must"
    exclusivity  the claim adds "only" or "solely" where the source doesn't
    citation     a cited case that doesn't exist

The checks are deliberately simple and explainable: every verdict comes with
the passage it was judged against and the exact reasons it failed.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from .citations import CITE_RE, CitationCheck, CitationVerifier, extract_citations
from .text import (EXCLUSIVE, content_words, modality, negated_words, negation_count,
                   quantities, same_word, split_sentences, tokens)


@dataclass
class Passage:
    doc: str
    text: str


@dataclass
class ClaimResult:
    claim: str
    verdict: str                     # "supported" | "unsupported"
    score: float                     # 0..1 share of the claim's key terms found in the evidence
    evidence: Optional[Passage]
    reasons: list[str] = field(default_factory=list)
    citations: list[CitationCheck] = field(default_factory=list)
    judge: Optional[dict] = None     # filled in when an LLM judge is used

    @property
    def supported(self) -> bool:
        return self.verdict == "supported"


@dataclass
class Report:
    results: list[ClaimResult]

    @property
    def unsupported(self) -> list[ClaimResult]:
        return [r for r in self.results if not r.supported]

    @property
    def grounding_rate(self) -> float:
        return sum(r.supported for r in self.results) / len(self.results) if self.results else 1.0


class Index:
    """BM25 over sentence windows. Windows of 1 and 2 sentences let a claim match
    a rule plus the citation that follows it in the next sentence."""

    def __init__(self, docs: dict[str, str], k1: float = 1.4, b: float = 0.75):
        self.passages: list[Passage] = []
        for name, text in docs.items():
            sents = split_sentences(text)
            for i, s in enumerate(sents):
                self.passages.append(Passage(name, s))
                if i + 1 < len(sents):
                    self.passages.append(Passage(name, s + " " + sents[i + 1]))
        self.terms = [content_words(p.text) for p in self.passages]
        self.avg = sum(map(len, self.terms)) / max(1, len(self.terms))
        df: Counter = Counter()
        for t in self.terms:
            df.update(set(t))
        n = len(self.terms)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, k: int = 8) -> list[Passage]:
        q = content_words(query)
        scores = []
        for i, terms in enumerate(self.terms):
            tf = Counter(terms)
            s = 0.0
            for w in q:
                f = sum(c for t, c in tf.items() if same_word(w, t))
                if f:
                    idf = self.idf.get(w, max(self.idf.values(), default=1.0))
                    s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(terms) / self.avg))
            scores.append((s, i))
        scores.sort(reverse=True)
        return [self.passages[i] for s, i in scores[:k] if s > 0]


HARD_PREFIXES = ("not in source:", "citation could not be verified", "no related passage")


def is_hard(reason: str) -> bool:
    """Hard failures are factual mismatches no paraphrase can explain."""
    return reason.startswith(HARD_PREFIXES)


def coverage(claim: str, passage: str) -> float:
    cw = list(dict.fromkeys(content_words(claim)))
    if not cw:
        return 1.0
    pw = content_words(passage)
    return sum(any(same_word(w, p) for p in pw) for w in cw) / len(cw)


def check_against(claim: str, passage: str) -> list[str]:
    """Return the reasons a claim is NOT supported by this passage (empty = fine)."""
    reasons = []
    ct, pt = tokens(claim), tokens(passage)

    # Citation numbers (volume, page, year) are verified separately, so they're
    # removed before comparing quantities like deadlines and dollar amounts.
    pq = quantities(CITE_RE.sub(" ", passage))
    missing = []
    for value, unit in sorted(quantities(CITE_RE.sub(" ", claim))):
        if not any(v == value and (not unit or not u or unit == u or same_word(unit, u))
                   for v, u in pq):
            missing.append(f"${value:,.0f}" if unit == "$" else f"{value:g}{unit}" if unit == "%"
                           else f"{value:g} {unit}".strip())
    if missing:
        reasons.append(f"not in source: {', '.join(missing)}")

    # Negation is checked sentence by sentence. Sentences with two negations
    # ("No person may operate ... without a permit") are double negatives that
    # state a requirement, so they're skipped rather than misread.
    src_neg: set[str] = set()
    for sent in split_sentences(passage) or [passage]:
        st = tokens(sent)
        if negation_count(st) == 1:
            src_neg |= negated_words(st)
    claim_neg = negated_words(ct) if negation_count(ct) == 1 else set()
    claim_plain = [w for w in ct if w not in claim_neg]
    passage_plain = [w for w in pt if w not in src_neg]
    if any(any(same_word(w, c) for c in claim_plain) for w in src_neg):
        reasons.append("source negates this; the claim states it as true")
    elif any(any(same_word(w, p) for p in passage_plain) for w in claim_neg - src_neg):
        reasons.append("claim adds a negation the source doesn't have")

    cm, pm = modality(ct), modality(pt)
    if cm and pm and cm != pm:
        reasons.append(f"source states a{'n ' if pm == 'obligation' else ' '}{pm}, "
                       f"the claim states a{'n ' if cm == 'obligation' else ' '}{cm}")

    if any(t in EXCLUSIVE for t in ct) and not any(t in EXCLUSIVE for t in pt):
        reasons.append("claim says 'only' or 'solely'; the source has no such limit")
    return reasons


class GroundingChecker:
    def __init__(self, sources: dict[str, str], citation_verifier: Optional[CitationVerifier] = None,
                 min_coverage: float = 0.7, judge=None):
        self.index = Index(sources)
        self.verifier = citation_verifier
        self.min_coverage = min_coverage
        self.judge = judge

    def check_claim(self, claim: str) -> ClaimResult:
        candidates = self.index.search(claim)
        best: Optional[Passage] = None
        best_cov, best_reasons = 0.0, ["no related passage in the sources"]
        # Rank passages by: talks about the same thing (enough coverage), then passes
        # every check, then coverage. So a claim with a wrong deadline is judged
        # against the clause about that deadline, not some unrelated clause that
        # happens to contain the same number.
        def rank(cov: float, reasons: list[str]) -> tuple:
            return (cov >= self.min_coverage, not reasons, cov)

        for p in candidates:
            cov = coverage(claim, p.text)
            reasons = check_against(claim, p.text)
            if best is None or rank(cov, reasons) > rank(best_cov, best_reasons):
                best, best_cov, best_reasons = p, cov, reasons

        reasons = list(best_reasons) if best else ["no related passage in the sources"]
        if best and best_cov < self.min_coverage:
            reasons.insert(0, f"only {best_cov:.0%} of the claim's key terms appear in the closest source passage")

        cites = extract_citations(claim)
        checks = self.verifier.verify(cites) if (self.verifier and cites) else []
        for c in checks:
            if c.status == "not_found":
                reasons.append(f"citation could not be verified: {c.citation.text} ({c.note})")

        result = ClaimResult(
            claim=claim,
            verdict="supported" if not reasons else "unsupported",
            score=round(best_cov, 3),
            evidence=best,
            reasons=reasons,
            citations=checks,
        )
        if self.judge and candidates and (result.supported or not any(map(is_hard, reasons))):
            # The judge reviews claims that passed (to catch subtle errors) and claims
            # that failed only soft checks (to clear faithful paraphrases). Hard
            # failures, like a missing number or a fake case, are never overridden.
            verdict = self.judge.judge(claim, [p.text for p in candidates[:3]])
            result.judge = verdict
            if verdict.get("label") == "unsupported" and result.supported:
                result.verdict = "unsupported"
                result.reasons.append(f"judge: {verdict.get('reason', 'not supported')}")
            elif verdict.get("label") == "supported" and not result.supported:
                result.verdict = "supported"
                result.reasons = [f"cleared by judge: {verdict.get('reason', '')}".strip()]
        return result

    def check_claims(self, claims: list[str]) -> Report:
        return Report([self.check_claim(c) for c in claims])

    def check_answer(self, answer: str) -> Report:
        return self.check_claims(split_claims(answer))


FRAMING = re.compile(r"^(here (is|are)|below (is|are)|the following|in summary|to summarize|"
                     r"overall|let me|i('| a)m|this (summary|answer|memo))\b", re.I)


def split_claims(answer: str) -> list[str]:
    """One claim per sentence. A sentence that is only a citation
    ("Celotex Corp. v. Catrett, 477 U.S. 317 (1986).") is attached to the claim before it.
    Framing sentences ("Here is a summary of...") aren't claims and are skipped."""
    claims: list[str] = []
    for s in split_sentences(answer):
        s = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", "", s)
        if FRAMING.match(s) or s.endswith(":"):
            continue
        is_bare_cite = extract_citations(s) and len(content_words(re.sub(r"\(.*?\)", "", s))) <= 6 \
            and not re.search(r"\b(held|holds|states|requires|provides)\b", s, re.I)
        if claims and is_bare_cite:
            claims[-1] = f"{claims[-1]} {s}"
        elif s:
            claims.append(s)
    return claims
