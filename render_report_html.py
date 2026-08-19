#!/usr/bin/env python3
"""Render a report markdown file into a standalone, readable HTML page.

The reports are numeric: dozens of result tables that get scanned, not read. So the
craft goes into information design rather than decoration -- tabular figures so digits
line up, numeric columns right-aligned, sticky table headers, each wide table scrolling
inside its own box so the page never scrolls sideways, and status colour reserved for
the distinction the report actually turns on (confirmed / unverified / do-not-cite).

No external assets: the artifact CSP blocks font and script CDNs, so type comes from
system stacks (which is also the honest choice for Korean text) and everything is inline.

Usage:
    python render_report_html.py results/REPORT_V2.md [-o results/REPORT_V2.html]
"""

import argparse
import html
import re
from pathlib import Path
from typing import List

CSS = """
*,*::before,*::after{box-sizing:border-box}
:root{
  --ground:#F5F7F7; --surface:#FFFFFF; --surface-2:#EFF3F3;
  --ink:#121A1D; --ink-2:#3A4A50; --muted:#617076; --rule:#D9E1E3;
  --accent:#0E6F6B; --accent-soft:#0E6F6B14;
  --ok:#1C7A4F; --warn:#8A5F0B; --bad:#A03030;
  --mono:ui-monospace,"SF Mono",SFMono-Regular,"JetBrains Mono",Menlo,Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Pretendard","Apple SD Gothic Neo",
         "Malgun Gothic","Noto Sans KR",system-ui,"Segoe UI",Roboto,sans-serif;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0C1214; --surface:#131B1E; --surface-2:#182226;
    --ink:#E4ECEE; --ink-2:#BCCACE; --muted:#8B9BA1; --rule:#243135;
    --accent:#57C6BA; --accent-soft:#57C6BA1F;
    --ok:#5DC08C; --warn:#D2A247; --bad:#E08585;
  }
}
:root[data-theme="dark"]{
  --ground:#0C1214; --surface:#131B1E; --surface-2:#182226;
  --ink:#E4ECEE; --ink-2:#BCCACE; --muted:#8B9BA1; --rule:#243135;
  --accent:#57C6BA; --accent-soft:#57C6BA1F;
  --ok:#5DC08C; --warn:#D2A247; --bad:#E08585;
}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:var(--sans); font-size:16px; line-height:1.75;
  font-variant-numeric:tabular-nums;
}
.wrap{max-width:1180px;margin:0 auto;padding:clamp(28px,5vw,72px) clamp(18px,4vw,44px) 96px}
.prose{max-width:74ch}
h1,h2,h3,h4{text-wrap:balance;line-height:1.25;margin:0}
h1{font-size:clamp(1.75rem,3.4vw,2.6rem);font-weight:700;letter-spacing:-.022em}
h2{font-size:clamp(1.25rem,2.2vw,1.65rem);font-weight:660;letter-spacing:-.014em;
   margin:3.4em 0 .2em;padding-top:1.5em;border-top:1px solid var(--rule)}
h2:first-of-type{border-top:0;padding-top:0}
h3{font-size:1.06rem;font-weight:650;margin:2.2em 0 .1em;color:var(--ink)}
h4{font-size:.94rem;font-weight:650;margin:1.6em 0 .1em;color:var(--ink-2)}
p{margin:.85em 0;max-width:74ch}
a{color:var(--accent);text-underline-offset:3px;text-decoration-thickness:1px}
strong{font-weight:660;color:var(--ink)}
hr{border:0;border-top:1px solid var(--rule);margin:2.6em 0}
ul,ol{margin:.7em 0;padding-left:1.35em;max-width:74ch}
li{margin:.32em 0}
li::marker{color:var(--muted)}
code{font-family:var(--mono);font-size:.855em;background:var(--surface-2);
     border:1px solid var(--rule);border-radius:4px;padding:.08em .34em}
pre{background:var(--surface);border:1px solid var(--rule);border-radius:8px;
    padding:16px 18px;overflow-x:auto;margin:1.2em 0;line-height:1.62}
pre code{background:none;border:0;padding:0;font-size:.845rem}
blockquote{margin:1.4em 0;padding:.9em 1.15em;background:var(--accent-soft);
  border-left:2.5px solid var(--accent);border-radius:0 6px 6px 0;color:var(--ink-2)}
blockquote p{margin:.3em 0}
.mathish{font-family:var(--mono);font-size:.9em;color:var(--ink-2);white-space:nowrap}
.tbl{overflow-x:auto;margin:1.3em 0;border:1px solid var(--rule);
     border-radius:8px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:.855rem;line-height:1.5}
thead th{position:sticky;top:0;background:var(--surface-2);z-index:1;
  font-weight:640;font-size:.76rem;letter-spacing:.045em;text-transform:uppercase;
  color:var(--muted);text-align:left;padding:9px 13px;border-bottom:1px solid var(--rule);
  white-space:nowrap}
tbody td{padding:8px 13px;border-bottom:1px solid var(--rule);vertical-align:top}
tbody tr:last-child td{border-bottom:0}
tbody tr:hover{background:var(--accent-soft)}
td.num{text-align:right;font-family:var(--mono);font-size:.83rem;white-space:nowrap}
td.lead{font-weight:560;white-space:nowrap}
details{margin:1.1em 0;border:1px solid var(--rule);border-radius:8px;
  background:var(--surface);overflow:hidden}
summary{cursor:pointer;padding:10px 14px;font-size:.82rem;font-weight:600;
  color:var(--muted);letter-spacing:.03em;list-style:none}
summary::-webkit-details-marker{display:none}
summary::before{content:"▸ ";color:var(--accent)}
details[open] summary{border-bottom:1px solid var(--rule)}
details[open] summary::before{content:"▾ "}
details>*:not(summary){margin-left:14px;margin-right:14px}
details .tbl{border:0;border-radius:0;margin:0 0 12px}
.masthead{margin-bottom:2.6em}
.masthead .kicker{font-family:var(--mono);font-size:.7rem;letter-spacing:.16em;
  text-transform:uppercase;color:var(--accent);margin-bottom:.9em}
.masthead .sub{color:var(--muted);font-size:.93rem;margin-top:.7em;max-width:74ch}
:focus-visible{outline:2px solid var(--accent);outline-offset:3px;border-radius:3px}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
@media print{.tbl,pre{overflow:visible}thead th{position:static}}
"""

NUM_RE = re.compile(r"^[\s*_]*[-+]?[\d.,]+(?:[eE][-+]?\d+)?\s*(?:%|%p|×|x|pp)?[\s*_]*$")

# --- LaTeX -> MathML ------------------------------------------------------------
# The reports use a small, closed subset of LaTeX.  MathML is native in every current
# browser, so this typesets properly with no CDN and no font payload -- which is what
# ruled out KaTeX here.  Anything the grammar below does not cover raises _TexError and
# falls back to the old monospace presentation, so an unsupported macro degrades to
# what it looked like before rather than emitting broken markup.

_GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
          "theta": "θ", "kappa": "κ", "lambda": "λ", "mu": "μ", "rho": "ρ",
          "sigma": "σ", "tau": "τ", "phi": "φ", "omega": "ω",
          "Delta": "Δ", "Sigma": "Σ", "Omega": "Ω", "Lambda": "Λ"}
_BINOPS = {"approx": "≈", "to": "→", "cdot": "⋅", "top": "⊤", "times": "×",
           "le": "≤", "ge": "≥", "ll": "≪", "gg": "≫", "pm": "±", "ne": "≠",
           "in": "∈", "propto": "∝", "mid": "∣", "cup": "∪", "cap": "∩",
           "subset": "⊂", "setminus": "∖", "equiv": "≡", "sim": "∼"}
_BIG = {"sum": "∑", "prod": "∏", "int": "∫"}
_SYM = {"infty": "∞", "partial": "∂", "nabla": "∇", "ldots": "…", "cdots": "⋯"}
_FUNCS = {"log", "ln", "exp", "min", "max", "det", "arg", "sin", "cos", "tan"}
_SPACE = {",": "0.167em", ";": "0.278em", ":": "0.222em", "!": "-0.167em",
          "quad": "1em", "qquad": "2em"}
_VARIANT = {"mathbb": "double-struck", "mathbf": "bold", "mathrm": "normal",
            "mathcal": "script", "mathsf": "sans-serif", "text": "normal"}

_TOK = re.compile(r"\\[a-zA-Z]+|\\[,;:!]|[{}_^]|\d*\.?\d+|[a-zA-Z]|\s+|.")


class _TexError(Exception):
    """The span is not in the supported subset -- caller falls back to monospace."""


class _Tex:
    def __init__(self, src: str) -> None:
        self.toks = [t for t in _TOK.findall(src) if t.strip()]
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self) -> str:
        t = self.peek()
        if t is None:
            raise _TexError("unexpected end of input")
        self.i += 1
        return t

    def row(self, closing: bool = False) -> str:
        out = []
        while True:
            t = self.peek()
            if t is None:
                if closing:
                    raise _TexError("unclosed group")
                break
            if t == "}":
                if not closing:
                    raise _TexError("stray }")
                self.take()
                break
            out.append(self.script())
        return "".join(out)

    def script(self) -> str:
        base, sub, sup = self.atom(), None, None
        while self.peek() in ("_", "^"):
            kind = self.take()
            arg = self.atom()
            if kind == "_":
                if sub is not None:
                    raise _TexError("double subscript")
                sub = arg
            else:
                if sup is not None:
                    raise _TexError("double superscript")
                sup = arg
        if sub is not None and sup is not None:
            return f"<msubsup>{base}{sub}{sup}</msubsup>"
        if sub is not None:
            return f"<msub>{base}{sub}</msub>"
        if sup is not None:
            return f"<msup>{base}{sup}</msup>"
        return base

    def atom(self) -> str:
        t = self.take()
        if t == "{":
            return f"<mrow>{self.row(closing=True)}</mrow>"
        if t in ("}", "_", "^"):
            raise _TexError(f"unexpected {t}")
        if t.startswith("\\"):
            return self.command(t[1:] if len(t) > 1 else t)
        if t[0].isdigit() or t[0] == ".":
            return f"<mn>{t}</mn>"
        if t.isalpha():
            return f"<mi>{t}</mi>"
        if t in "()[]":
            return f'<mo stretchy="false">{html.escape(t)}</mo>'
        return f"<mo>{html.escape(t)}</mo>"

    def command(self, name: str) -> str:
        if name in _SPACE:
            return f'<mspace width="{_SPACE[name]}"/>'
        if name in _GREEK:
            return f"<mi>{_GREEK[name]}</mi>"
        if name in _SYM:
            return f"<mi>{_SYM[name]}</mi>"
        if name in _BINOPS:
            return f"<mo>{_BINOPS[name]}</mo>"
        if name in _BIG:
            return f'<mo movablelimits="true">{_BIG[name]}</mo>'
        if name in _FUNCS:
            # mrow-wrapped so a following _ or ^ attaches to the whole function name
            return (f'<mrow><mi mathvariant="normal">{name}</mi>'
                    f'<mspace width="0.167em"/></mrow>')
        if name in ("bar", "overline"):
            return f'<mover accent="true">{self.atom()}<mo>‾</mo></mover>'
        if name == "hat":
            return f'<mover accent="true">{self.atom()}<mo>^</mo></mover>'
        if name == "tilde":
            return f'<mover accent="true">{self.atom()}<mo>~</mo></mover>'
        if name == "vec":
            return f'<mover accent="true">{self.atom()}<mo>→</mo></mover>'
        if name in _VARIANT:
            return f'<mi mathvariant="{_VARIANT[name]}">{self.word()}</mi>'
        if name == "frac":
            return f"<mfrac>{self.atom()}{self.atom()}</mfrac>"
        if name == "sqrt":
            return f"<msqrt>{self.atom()}</msqrt>"
        if name in ("left", "right"):
            return self.atom()          # drop the sizing, keep the delimiter
        raise _TexError(f"unsupported command \\{name}")

    def word(self) -> str:
        """Argument of \\mathrm-style macros: a braced run of plain characters, or one token."""
        t = self.take()
        if t != "{":
            if t.startswith("\\") or t in "}_^":
                raise _TexError("bad font-macro argument")
            return html.escape(t)
        buf = []
        while True:
            t = self.take()
            if t == "}":
                break
            if t.startswith("\\") or t in "{_^":
                raise _TexError("nested markup in font macro")
            buf.append(t)
        if not buf:
            raise _TexError("empty font-macro argument")
        return html.escape("".join(buf))


def tex_to_mathml(tex: str) -> str:
    """Inline LaTeX -> MathML.  Raises _TexError outside the supported subset."""
    body = _Tex(html.unescape(tex)).row()
    if not body:
        raise _TexError("empty span")
    return (f'<math xmlns="http://www.w3.org/1998/Math/MathML" display="inline">'
            f"<mrow>{body}</mrow></math>")


def _render_math(m: "re.Match[str]") -> str:
    try:
        return tex_to_mathml(m.group(1))
    except _TexError:
        return f'<span class="mathish">{m.group(1)}</span>'


def inline(s: str) -> str:
    """Inline markdown -> HTML.  Order matters: code first so nothing rewrites inside it."""
    out, parts = [], re.split(r"(`[^`]+`)", s)
    for part in parts:
        if part.startswith("`") and part.endswith("`") and len(part) > 1:
            out.append(f"<code>{html.escape(part[1:-1])}</code>")
            continue
        t = html.escape(part)
        t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
        t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", t)
        # $...$ is LaTeX; typeset via MathML, falling back to monospace if unsupported
        t = re.sub(r"\$([^$]+)\$", _render_math, t)
        out.append(t)
    return "".join(out)


def render_table(block: List[str]) -> str:
    rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in block]
    head, body = rows[0], rows[2:]
    h = "".join(f"<th>{inline(c)}</th>" for c in head)
    trs = []
    for r in body:
        tds = []
        for i, c in enumerate(r):
            cls = "num" if (i and NUM_RE.match(c)) else ("lead" if i == 0 else "")
            tds.append(f'<td class="{cls}">{inline(c)}</td>' if cls else f"<td>{inline(c)}</td>")
        trs.append("<tr>" + "".join(tds) + "</tr>")
    return (f'<div class="tbl"><table><thead><tr>{h}</tr></thead>'
            f'<tbody>{"".join(trs)}</tbody></table></div>')


def md_to_html(md: str) -> str:
    lines = md.replace("\r\n", "\n").split("\n")
    out: List[str] = []
    i, n = 0, len(lines)
    list_stack: List[str] = []

    def close_lists() -> None:
        while list_stack:
            out.append(f"</{list_stack.pop()}>")

    while i < n:
        ln = lines[i]

        if ln.startswith("<!--"):
            i += 1
            continue
        if ln.strip().startswith(("<details", "</details", "<summary", "</summary")):
            close_lists()
            out.append(ln.strip())
            i += 1
            continue
        if ln.startswith("```"):
            close_lists()
            i += 1
            buf = []
            while i < n and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            out.append(f"<pre><code>{html.escape(chr(10).join(buf))}</code></pre>")
            continue
        if re.match(r"^\s*\|.*\|\s*$", ln) and i + 1 < n and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            close_lists()
            blk = []
            while i < n and re.match(r"^\s*\|.*\|\s*$", lines[i]):
                blk.append(lines[i])
                i += 1
            out.append(render_table(blk))
            continue
        if re.match(r"^\s*(---|___|\*\*\*)\s*$", ln):
            close_lists()
            out.append("<hr>")
            i += 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            close_lists()
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>")
            i += 1
            continue
        if ln.startswith(">"):
            close_lists()
            buf = []
            while i < n and lines[i].startswith(">"):
                buf.append(lines[i].lstrip("> ").rstrip())
                i += 1
            out.append(f"<blockquote><p>{inline(' '.join(buf))}</p></blockquote>")
            continue
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", ln)
        if m:
            tag = "ul" if m.group(2) in "-*" else "ol"
            if not list_stack:
                list_stack.append(tag)
                out.append(f"<{tag}>")
            out.append(f"<li>{inline(m.group(3))}</li>")
            i += 1
            continue
        if not ln.strip():
            close_lists()
            i += 1
            continue
        buf = [ln]
        i += 1
        while i < n and lines[i].strip() and not re.match(
                r"^\s*(\||#{1,6}\s|```|>|[-*]\s|\d+\.\s|---|<details|</details|<summary)", lines[i]):
            buf.append(lines[i])
            i += 1
        close_lists()
        out.append(f"<p>{inline(' '.join(buf))}</p>")

    close_lists()
    return "\n".join(out)


def build_page(md: str, title: str) -> str:
    body = md_to_html(md)
    # promote the first h1 into a masthead with its lead paragraphs
    m = re.match(r"^(<h1>.*?</h1>)(.*?)(?=<h2|<hr>)", body, re.DOTALL)
    if m:
        head = (f'<header class="masthead"><div class="kicker">'
                f'sparse-attention-hub · experiment report</div>'
                f'{m.group(1)}<div class="sub">{m.group(2)}</div></header>')
        body = head + body[m.end():]
    return (f"<title>{html.escape(title)}</title><style>{CSS}</style>"
            f'<div class="wrap"><div class="prose-root">{body}</div></div>')


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a report markdown to standalone HTML")
    ap.add_argument("source")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()

    src = Path(a.source)
    md = src.read_text()
    title = a.title or next((l[2:].strip() for l in md.split("\n") if l.startswith("# ")),
                            src.stem)
    out = Path(a.out) if a.out else src.with_suffix(".html")
    out.write_text(build_page(md, title))
    print(f"wrote {out}  ({len(out.read_text()):,} bytes)")


if __name__ == "__main__":
    main()
