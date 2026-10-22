"""Score the checker on the labeled benchmark.

    python evals/run_eval.py
    python evals/run_eval.py --judge together     # add the LLM judge (needs TOGETHER_API_KEY)

"Positive" means unsupported: the thing we want to catch. Precision answers
"when it raises a flag, is it right?" Recall answers "of the bad claims, how
many did it catch?"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from verdict.checker import GroundingChecker  # noqa: E402
from verdict.citations import LocalVerifier  # noqa: E402

# Real Supreme Court citations used by the memo, for offline citation checks.
KNOWN_CASES = {
    "550 U.S. 544": "Bell Atlantic Corp. v. Twombly",
    "556 U.S. 662": "Ashcroft v. Iqbal",
    "477 U.S. 317": "Celotex Corp. v. Catrett",
    "477 U.S. 242": "Anderson v. Liberty Lobby, Inc.",
}


SETS = {"dev": "benchmark.jsonl", "holdout": "holdout.jsonl"}


def load(which: str = "dev"):
    sources = {p.stem: p.read_text() for p in sorted((ROOT / "sources").glob("*.txt"))}
    rows = [json.loads(line) for line in (ROOT / SETS[which]).read_text().splitlines() if line]
    return sources, rows


def evaluate(judge=None, which: str = "dev") -> dict:
    sources, rows = load(which)
    checker = GroundingChecker(sources, LocalVerifier(KNOWN_CASES), judge=judge)
    tp = fp = fn = tn = 0
    by_type: dict[str, list[bool]] = defaultdict(list)
    misses = []
    for r in rows:
        res = checker.check_claim(r["claim"])
        flagged, bad = not res.supported, r["label"] == "unsupported"
        tp += flagged and bad
        fp += flagged and not bad
        fn += (not flagged) and bad
        tn += (not flagged) and not bad
        if bad:
            by_type[r["error_type"]].append(flagged)
        if flagged != bad:
            misses.append((r, res))
    p = tp / (tp + fp) if tp + fp else 1.0
    rc = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    out = {
        "claims": len(rows), "unsupported": tp + fn,
        "precision": round(p, 3), "recall": round(rc, 3), "f1": round(f1, 3),
        "accuracy": round((tp + tn) / len(rows), 3),
        "recall_by_error_type": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_type.items())},
        "mistakes": [
            {"id": r["id"], "claim": r["claim"], "label": r["label"],
             "error_type": r["error_type"], "reasons": res.reasons}
            for r, res in misses
        ],
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", choices=["together"], default=None)
    ap.add_argument("--set", choices=list(SETS), default="dev",
                    help="dev = used while building the checks; holdout = written afterwards, never tuned on")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    judge = None
    if a.judge == "together":
        from verdict.judge import TogetherJudge
        judge = TogetherJudge(os.environ["TOGETHER_API_KEY"])
    out = evaluate(judge, a.set)
    if a.json:
        print(json.dumps(out, indent=2))
        return
    print(f"[{a.set}] {out['claims']} claims, {out['unsupported']} unsupported\n")
    print(f"precision {out['precision']:.0%}   recall {out['recall']:.0%}   "
          f"F1 {out['f1']:.2f}   accuracy {out['accuracy']:.0%}\n")
    print("caught, by error type:")
    for k, v in out["recall_by_error_type"].items():
        print(f"  {k:28} {v}")
    if out["mistakes"]:
        print("\nmistakes:")
        for m in out["mistakes"]:
            kind = "missed" if m["label"] == "unsupported" else "false alarm"
            print(f"  #{m['id']} {kind} ({m['error_type'] or 'supported claim'}): {m['claim']}")
            for why in m["reasons"]:
                print(f"      - {why}")


if __name__ == "__main__":
    main()
