"""Render a Report as Markdown (terminal) or a standalone HTML page.

The HTML version reads like a marked-up brief: the AI's answer with every
claim underlined, unsupported ones struck in red, and margin notes saying
exactly what's wrong and which source passage was checked.
"""
from __future__ import annotations

import html

from .checker import Report


def to_markdown(report: Report) -> str:
    lines = [f"**{len(report.unsupported)} of {len(report.results)} claims unsupported** "
             f"(grounding rate {report.grounding_rate:.0%})", ""]
    for i, r in enumerate(report.results, 1):
        mark = "OK " if r.supported else "FLAG"
        lines.append(f"{i}. [{mark}] {r.claim}")
        for why in r.reasons:
            lines.append(f"     - {why}")
        if r.evidence and not r.supported:
            lines.append(f"     source ({r.evidence.doc}): \"{r.evidence.text[:220]}\"")
    return "\n".join(lines)


def to_html(report: Report, title: str = "Grounding report") -> str:
    e = html.escape
    body, notes = [], []
    for i, r in enumerate(report.results, 1):
        cls = "ok" if r.supported else "bad"
        body.append(f'<span class="claim {cls}" id="c{i}"><a href="#n{i}">{e(r.claim)}</a><sup>{i}</sup></span> ')
        reasons = "".join(f"<li>{e(x)}</li>" for x in r.reasons) or "<li>Every check passed.</li>"
        cites = "".join(
            f'<li class="cite {c.status}">{e(c.citation.text)}: '
            f'{"found" if c.status == "found" else e(c.note or c.status)}'
            f'{" (" + e(c.matched_name) + ")" if c.matched_name and c.status == "found" else ""}</li>'
            for c in r.citations)
        ev = (f'<blockquote><cite>{e(r.evidence.doc.replace("_", " "))}</cite>{e(r.evidence.text)}</blockquote>'
              if r.evidence else "")
        notes.append(
            f'<li class="note {cls}" id="n{i}"><div class="head"><b>{i}</b>'
            f'<span>{"Supported" if r.supported else "Unsupported"}</span>'
            f'<span class="score">{r.score:.0%} key terms matched</span></div>'
            f'<ul class="why">{reasons}{cites}</ul>{ev}</li>')

    n_bad = len(report.unsupported)
    summary = ("Every claim is backed by the sources." if not n_bad else
               f"{n_bad} of {len(report.results)} claims are not backed by the sources.")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)}</title>
<style>
:root {{ color-scheme: light dark; --bg:#f5f6f8; --paper:#fff; --ink:#1d2330; --muted:#5d6676;
  --rule:#dfe3ea; --ok:#2f7d55; --bad:#b3261e; --bad-bg:#fbe9e7; --ok-bg:#e8f3ec; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141920; --paper:#1c222c; --ink:#e6eaf0;
  --muted:#9aa4b4; --rule:#2c3440; --ok:#5cc28b; --bad:#ff7a6e; --bad-bg:#3a2224; --ok-bg:#1f3328; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width:1120px; margin:0 auto; padding:32px 20px 56px; }}
h1 {{ font-size:24px; margin:0 0 4px; }}
.summary {{ color:var(--muted); margin:0 0 24px; }}
.grid {{ display:grid; grid-template-columns:minmax(0,1.1fr) minmax(0,1fr); gap:24px; align-items:start; }}
@media (max-width:820px) {{ .grid {{ grid-template-columns:1fr; }} }}
.answer {{ background:var(--paper); border:1px solid var(--rule); border-radius:10px; padding:28px 30px;
  font:17px/1.85 Charter, "Iowan Old Style", Georgia, serif; position:sticky; top:16px; }}
.claim a {{ color:inherit; text-decoration:none; }}
.claim.ok a {{ border-bottom:2px solid var(--ok); }}
.claim.bad a {{ background:var(--bad-bg); text-decoration:line-through; text-decoration-color:var(--bad);
  text-decoration-thickness:2px; }}
.claim sup {{ font:600 11px ui-sans-serif, system-ui, sans-serif; color:var(--muted); margin-left:1px; }}
.claim.bad sup {{ color:var(--bad); }}
.claim a:focus-visible {{ outline:2px solid var(--ink); outline-offset:2px; }}
ol.notes {{ list-style:none; margin:0; padding:0; display:grid; gap:12px; }}
.note {{ background:var(--paper); border:1px solid var(--rule); border-left:4px solid var(--ok);
  border-radius:8px; padding:14px 16px; }}
.note.bad {{ border-left-color:var(--bad); }}
.note:target {{ box-shadow:0 0 0 2px var(--ink); }}
.head {{ display:flex; gap:10px; align-items:baseline; font-size:14px; }}
.head span:first-of-type {{ font-weight:600; }}
.note.bad .head span:first-of-type {{ color:var(--bad); }}
.note.ok .head span:first-of-type {{ color:var(--ok); }}
.score {{ margin-left:auto; color:var(--muted); font-size:13px; }}
.why {{ margin:8px 0 0; padding-left:18px; font-size:14px; }}
.cite.not_found {{ color:var(--bad); font-weight:600; }}
blockquote {{ margin:10px 0 0; padding:8px 12px; border-left:2px solid var(--rule); color:var(--muted);
  font:14px/1.6 Charter, Georgia, serif; }}
cite {{ display:block; font:12px ui-sans-serif, system-ui, sans-serif; margin-bottom:2px; text-transform:capitalize; }}
</style></head>
<body><main>
<h1>{e(title)}</h1>
<p class="summary">{summary} Select a claim to see what it was checked against.</p>
<div class="grid">
<article class="answer">{''.join(body)}</article>
<ol class="notes">{''.join(notes)}</ol>
</div>
</main></body></html>"""
