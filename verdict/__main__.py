"""Command line.

    python -m verdict check --answer answer.txt --sources docs/ --html report.html
    python -m verdict check --answer answer.txt --sources docs/ --judge together --courtlistener
    python -m verdict eval --set holdout
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .checker import GroundingChecker
from .citations import CourtListenerVerifier, LocalVerifier
from .report import to_html, to_markdown


def load_sources(path: str) -> dict[str, str]:
    p = Path(path)
    files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.suffix in {".txt", ".md"})
    if not files:
        sys.exit(f"No .txt or .md files found in {path}")
    return {f.stem: f.read_text() for f in files}


def cmd_check(a) -> int:
    sources = load_sources(a.sources)
    verifier = None
    if a.courtlistener:
        token = os.getenv("COURTLISTENER_TOKEN")
        if not token:
            sys.exit("--courtlistener needs COURTLISTENER_TOKEN (free at courtlistener.com)")
        verifier = CourtListenerVerifier(token)
    elif a.known_cases:
        import json
        verifier = LocalVerifier(json.loads(Path(a.known_cases).read_text()))
    judge = None
    if a.judge == "together":
        from .judge import TogetherJudge
        key = os.getenv("TOGETHER_API_KEY")
        if not key:
            sys.exit("--judge together needs TOGETHER_API_KEY")
        judge = TogetherJudge(key)

    checker = GroundingChecker(sources, verifier, min_coverage=a.min_coverage, judge=judge)
    report = checker.check_answer(Path(a.answer).read_text())
    print(to_markdown(report))
    if a.html:
        Path(a.html).write_text(to_html(report, title=a.title))
        print(f"\nHTML report written to {a.html}")
    return 1 if report.unsupported and a.fail_on_unsupported else 0


def main() -> None:
    ap = argparse.ArgumentParser(prog="verdict", description="Check AI-written legal text against its sources.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="check an answer against source documents")
    c.add_argument("--answer", required=True, help="file with the AI-generated answer")
    c.add_argument("--sources", required=True, help="a source file or a folder of .txt/.md files")
    c.add_argument("--html", help="also write an annotated HTML report here")
    c.add_argument("--title", default="Grounding report")
    c.add_argument("--min-coverage", type=float, default=0.7)
    c.add_argument("--judge", choices=["together"], help="second opinion from an LLM on soft cases")
    c.add_argument("--courtlistener", action="store_true", help="verify case citations against CourtListener")
    c.add_argument("--known-cases", help="JSON file of known citations for offline checks")
    c.add_argument("--fail-on-unsupported", action="store_true", help="exit 1 if anything is flagged (for CI)")

    e = sub.add_parser("eval", help="score the checker on the labeled benchmark")
    e.add_argument("--set", choices=["dev", "holdout"], default="dev")

    a = ap.parse_args()
    if a.cmd == "check":
        sys.exit(cmd_check(a))
    sys.argv = ["run_eval", "--set", a.set]
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evals.run_eval import main as run_eval
    run_eval()


if __name__ == "__main__":
    main()
