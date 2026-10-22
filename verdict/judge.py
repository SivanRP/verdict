"""Optional LLM judge for the cases fast checks can't reason about.

The rule-based checks are fast, free, and explainable, but they can't follow
an exception clause or tell that "required before terminating" paraphrases
"upon prior written notice." The judge gets a second look in two situations:

  1. A claim passed every check. The judge can still catch subtle errors
     (ignored exceptions, numbers borrowed from the wrong clause).
  2. A claim failed only soft checks (word overlap, modality, negation wording).
     The judge can clear paraphrases that are actually faithful.

Hard failures, like a number that appears nowhere in the source or a case
citation that doesn't exist, are never overridden. Those are facts, not
judgment calls.
"""
from __future__ import annotations

import json
import re
from typing import Protocol

import httpx

PROMPT = """You check whether a legal claim is fully supported by source text.

Source passages:
{passages}

Claim:
{claim}

Rules:
- "supported" only if the passages state everything the claim says, including numbers, parties, exceptions, and whether something is required or optional.
- Paraphrase is fine. Adding facts, dropping an exception, or changing who/what/how much is not.
- Quote the exact source sentence you relied on.

Reply with JSON only: {{"label": "supported" | "unsupported", "reason": "<one sentence>", "quote": "<source text>"}}"""


class Judge(Protocol):
    def judge(self, claim: str, passages: list[str]) -> dict: ...


class TogetherJudge:
    URL = "https://api.together.xyz/v1/chat/completions"

    def __init__(self, api_key: str, model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                 timeout: float = 60.0):
        self.model = model
        self.client = httpx.Client(headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)

    def judge(self, claim: str, passages: list[str]) -> dict:
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 300,
            "messages": [{"role": "user", "content": PROMPT.format(
                passages="\n".join(f"- {p}" for p in passages), claim=claim)}],
        }
        resp = self.client.post(self.URL, json=body)
        resp.raise_for_status()
        return parse_verdict(resp.json()["choices"][0]["message"]["content"])


def parse_verdict(text: str) -> dict:
    """Pull the JSON object out of a model reply, tolerating code fences or stray prose."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"label": "unknown", "reason": "judge reply had no JSON", "quote": ""}
    try:
        data = json.loads(m.group())
    except json.JSONDecodeError:
        return {"label": "unknown", "reason": "judge reply was not valid JSON", "quote": ""}
    label = str(data.get("label", "")).lower()
    data["label"] = label if label in {"supported", "unsupported"} else "unknown"
    return data
