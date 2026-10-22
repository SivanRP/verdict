"""Find case citations in text and check that the cases actually exist.

Fabricated citations are the most dangerous kind of legal AI hallucination:
they look exactly like real ones. This module pulls citations like
"550 U.S. 544 (2007)" out of text and verifies them against CourtListener,
a free public database of US case law.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Protocol

import httpx

REPORTERS = [
    r"U\.\s?S\.", r"S\.\s?Ct\.", r"L\.\s?Ed\.\s?2d", r"L\.\s?Ed\.",
    r"F\.\s?4th", r"F\.\s?3d", r"F\.\s?2d", r"F\.",
    r"F\.\s?Supp\.\s?3d", r"F\.\s?Supp\.\s?2d", r"F\.\s?Supp\.",
    r"Cal\.\s?(?:App\.\s?)?(?:4th|5th|3d|2d)?", r"N\.\s?E\.\s?(?:2d|3d)?",
    r"N\.\s?Y\.\s?(?:2d|3d)?", r"A\.\s?(?:2d|3d)?", r"P\.\s?(?:2d|3d)?",
    r"So\.\s?(?:2d|3d)?", r"S\.\s?W\.\s?(?:2d|3d)?", r"N\.\s?W\.\s?2d",
]
_REPORTER = "|".join(sorted(REPORTERS, key=len, reverse=True))
CITE_RE = re.compile(
    rf"(?:(?P<name>[A-Z][\w.&'\- ]{{1,80}}? v\. [A-Z][\w.&'\- ]{{1,80}}?),\s+)?"
    rf"(?P<vol>\d{{1,4}})\s+(?P<rep>{_REPORTER})\s+(?P<page>\d{{1,5}})"
    rf"(?:,\s*\d{{1,5}}(?:-\d{{1,5}})?)?"
    rf"(?:\s*\((?P<court>[^()]*?)\s*(?P<year>\d{{4}})\))?"
)


@dataclass(frozen=True)
class Citation:
    text: str
    volume: int
    reporter: str
    page: int
    case_name: Optional[str] = None
    year: Optional[int] = None

    @property
    def key(self) -> str:
        return f"{self.volume} {self.reporter} {self.page}"


def extract_citations(text: str) -> list[Citation]:
    out = []
    for m in CITE_RE.finditer(text):
        rep = re.sub(r"\s+", "", m.group("rep"))
        name = m.group("name")
        if name:
            name = re.sub(r"^(?:In|See|Under|Accord|Cf\.)\s+", "", name.strip())
        out.append(Citation(
            text=m.group(0).strip(),
            volume=int(m.group("vol")),
            reporter=rep,
            page=int(m.group("page")),
            case_name=name,
            year=int(m.group("year")) if m.group("year") else None,
        ))
    return out


@dataclass
class CitationCheck:
    citation: Citation
    status: str              # "found" | "not_found" | "unknown"
    matched_name: Optional[str] = None
    note: str = ""


class CitationVerifier(Protocol):
    def verify(self, cites: list[Citation]) -> list[CitationCheck]: ...


class LocalVerifier:
    """Offline verifier backed by a small table of known cases.
    Used in tests and demos so the project runs with no network or API key."""

    def __init__(self, known: dict[str, str]):
        self.known = {re.sub(r"\s+", "", k): v for k, v in known.items()}

    def verify(self, cites: list[Citation]) -> list[CitationCheck]:
        out = []
        for c in cites:
            name = self.known.get(f"{c.volume}{c.reporter}{c.page}")
            if name is None:
                out.append(CitationCheck(c, "not_found", note="no case at this volume/reporter/page"))
            elif c.case_name and not _names_agree(c.case_name, name):
                out.append(CitationCheck(c, "not_found", matched_name=name,
                                         note=f"citation points to {name}, not {c.case_name}"))
            else:
                out.append(CitationCheck(c, "found", matched_name=name))
        return out


class CourtListenerVerifier:
    """Checks citations with CourtListener's citation lookup API, which was built
    specifically to catch hallucinated citations. Needs a free API token from
    courtlistener.com (set COURTLISTENER_TOKEN)."""

    URL = "https://www.courtlistener.com/api/rest/v4/citation-lookup/"

    def __init__(self, token: str, timeout: float = 20.0):
        self.client = httpx.Client(headers={"Authorization": f"Token {token}"}, timeout=timeout)

    def verify(self, cites: list[Citation]) -> list[CitationCheck]:
        if not cites:
            return []
        resp = self.client.post(self.URL, data={"text": "\n".join(c.text for c in cites)})
        resp.raise_for_status()
        by_key: dict[str, dict] = {}
        for item in resp.json():
            raw = re.sub(r"\s+", "", item.get("citation", ""))
            by_key[raw] = item
        out = []
        for c in cites:
            item = by_key.get(f"{c.volume}{c.reporter}{c.page}")
            if item is None:
                out.append(CitationCheck(c, "unknown", note="not returned by CourtListener"))
                continue
            status = item.get("status")
            clusters = item.get("clusters") or []
            name = clusters[0].get("case_name") if clusters else None
            if status in (200, 300) and clusters:
                if c.case_name and name and not _names_agree(c.case_name, name):
                    out.append(CitationCheck(c, "not_found", matched_name=name,
                                             note=f"citation points to {name}, not {c.case_name}"))
                else:
                    out.append(CitationCheck(c, "found", matched_name=name))
            elif status == 404:
                out.append(CitationCheck(c, "not_found", note="no such citation"))
            else:
                out.append(CitationCheck(c, "unknown", note=f"lookup status {status}"))
        return out


def _names_agree(claimed: str, actual: str) -> bool:
    """True if the first party's distinctive word appears in the real case name."""
    first = re.split(r"\s+v\.\s+", claimed)[0]
    words = [w for w in re.findall(r"[A-Za-z]{3,}", first)
             if w.lower() not in {"inc", "corp", "llc", "the", "company", "united", "states"}]
    return not words or any(w.lower() in actual.lower() for w in words)
