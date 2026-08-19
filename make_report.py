#!/usr/bin/env python3
"""Regenerate the cumulative experiment report at results/REPORT.md.

The report has two kinds of content:

  * AUTO blocks, delimited by  <!-- AUTO:<name> START/END -->, which this script
    rewrites from the raw result directories on every run.  Never hand-edit them.
  * everything else -- conclusions, invalidation notes, open questions -- which is
    hand-written and is PRESERVED across runs.  Edit those freely.

So the workflow is: run experiments -> `python make_report.py` -> tables refresh,
narrative stays.  New result directories are picked up automatically.

Numbers come from  <result_dir>/<method>/<subset>/metrics.json  and the settings
that make a row comparable come from the sibling config.json, not from the
directory name -- so a mislabelled run cannot silently pass as comparable.

Usage:
    python make_report.py                 # refresh AUTO blocks
    python make_report.py --check         # exit 1 if the report is stale
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parent
DEFAULT_REPORT = REPO / "results" / "REPORT.md"
REPORT = DEFAULT_REPORT

SUBSETS = ["qa_1", "qa_2", "vt", "fwe",
           "niah_multikey_2", "niah_multikey_3", "niah_multivalue"]
SHORT = {"niah_multikey_2": "nk_2", "niah_multikey_3": "nk_3", "niah_multivalue": "nmv"}

# Result directories to include, in report order, with a one-line purpose.
TRACKED = [
    ("results_v2", "V2 ablation ladder @5%, HAT selector (dense → ours)"),
    ("results_multibin_hat", "Multi-bin UTA vs vAttention (5%/10%, 두 셀렉터)"),
    ("results_density_sweep", "Density sweep 1/2/3% (oracle-top-k 셀렉터)"),
    ("results_covuta", "이전 3/5/10% 실행 — 프로토콜 일치, baseline으로 재사용"),
    ("results_full_comparison", "이전 5/10% 비교 (CV_UTA 포함)"),
    ("results_uta_low_density", "초기 3/5% 탐색 (50샘플 — 검정력 부족)"),
    ("results_damped_jensen_full", "Damped Jensen (alpha=0.25) vs vAttention @10%"),
    ("results_jensen_diagnostic", "Jensen alpha 스윕 진단"),
    ("results_covuta_v2", "CovUTA 분자 보정 (폐기됨)"),
]


# --------------------------------------------------------------------- data --
def read_score(path: Path, subset: str) -> Optional[float]:
    try:
        m = json.loads(path.read_text())
    except Exception:
        return None
    v = m.get("task_scores", {}).get(subset, {}).get("string_match")
    return round(v, 2) if isinstance(v, (int, float)) else None


def fingerprint(cfg_path: Path) -> Dict[str, Any]:
    """Settings that decide whether two rows may be compared."""
    fp: Dict[str, Any] = {}
    try:
        sa = json.loads(cfg_path.read_text()).get("sparse_attention_config")
    except Exception:
        return fp
    if not sa:                      # dense baseline: no sparse config at all
        return {"selector": "—", "density": "100%"}

    density = 0.0
    for m in sa.get("masker_configs", []):
        for k in ("sink_size", "window_size", "heavy_size", "base_rate_sampling"):
            if k in m and isinstance(m[k], (int, float)):
                density += m[k]
        if "hat_bits" in m:
            fp["selector"] = "HAT"
        elif "heavy_size" in m and "selector" not in fp:
            fp["selector"] = "oracle-top-k"
        if "heavy_size" in m:
            fp["heavy"] = m["heavy_size"]
        if "base_rate_sampling" in m:
            fp["sampling"] = m["base_rate_sampling"]
            fp["eps/delta"] = f"{m.get('epsilon')}/{m.get('delta')}"
    if density:
        fp["density"] = f"{density * 100:.3g}%"
    for k in ("bin_mode", "num_bins", "bin_size", "kappa_mode"):
        if k in sa:
            fp[k] = sa[k]
    if fp.get("bin_mode") in ("fixed", "equalcount"):
        fp.pop("num_bins", None)
    elif "bin_mode" in fp:
        fp.pop("bin_size", None)
    return fp


def collect(result_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for method_dir in sorted(p for p in result_dir.rglob("*") if p.is_dir()):
        hits = {s: method_dir / s / "metrics.json" for s in SUBSETS}
        hits = {s: p for s, p in hits.items() if p.exists()}
        if not hits:
            continue
        label = str(method_dir.relative_to(result_dir)).replace("_at_", "@")
        scores = {s: read_score(p, s) for s, p in hits.items()}
        got = [v for v in scores.values() if v is not None]
        cfg = next((method_dir / s / "config.json" for s in hits
                    if (method_dir / s / "config.json").exists()), None)
        mtimes = [p.stat().st_mtime for p in hits.values()]
        rows.append({
            "label": label,
            "scores": scores,
            "n": len(got),
            "avg": round(sum(got) / len(got), 2) if len(got) == len(SUBSETS) else None,
            "partial_avg": round(sum(got) / len(got), 2) if got else None,
            "fp": fingerprint(cfg) if cfg else {},
            "updated": datetime.fromtimestamp(max(mtimes)).strftime("%m-%d %H:%M"),
        })
    return rows


# ------------------------------------------------------------------ render --
def md_table(rows: List[Dict[str, Any]]) -> str:
    head = ["방법", "평균", "완료"] + [SHORT.get(s, s) for s in SUBSETS]
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join(["---"] * len(head)) + "|"]
    for r in sorted(rows, key=lambda x: (-(x["avg"] or -1), x["label"])):
        avg = f"**{r['avg']}**" if r["avg"] is not None else f"_{r['partial_avg']}_*"
        cells = [r["label"], avg, f"{r['n']}/7"]
        cells += [("—" if r["scores"].get(s) is None else str(r["scores"][s])) for s in SUBSETS]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def md_fingerprints(rows: List[Dict[str, Any]]) -> str:
    keys, seen = [], set()
    for r in rows:
        for k in r["fp"]:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    if not keys:
        return "_config.json 없음_"
    out = ["| 방법 | " + " | ".join(keys) + " |",
           "|" + "|".join(["---"] * (len(keys) + 1)) + "|"]
    for r in sorted(rows, key=lambda x: x["label"]):
        out.append("| " + r["label"] + " | "
                   + " | ".join(str(r["fp"].get(k, "—")) for k in keys) + " |")
    return "\n".join(out)


def build_results_block() -> str:
    parts = []
    for name, purpose in TRACKED:
        d = REPO / name
        if not d.is_dir():
            continue
        rows = collect(d)
        if not rows:
            continue
        complete = sum(1 for r in rows if r["avg"] is not None)
        parts.append(f"### `{name}`\n\n{purpose}  \n"
                     f"_{complete}/{len(rows)}개 완료(7/7 subset), "
                     f"최종 기록 {max(r['updated'] for r in rows)}_\n\n"
                     + md_table(rows)
                     + "\n\n<details><summary>config.json에서 검증한 설정</summary>\n\n"
                     + md_fingerprints(rows) + "\n\n</details>")
    body = "\n\n".join(parts) if parts else "_결과 없음_"
    return (f"_{datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M %Z')} "
            f"`make_report.py` 자동 생성. `_숫자_*` 표기는 7개 미만 subset의 부분 평균이며 "
            f"7/7 평균과 비교 불가._\n\n" + body)


def build_inventory_block() -> str:
    out = ["| 결과 디렉터리 | 실행 수 | 완료 | 완료 subset 합 | 최종 기록 |",
           "|---|---|---|---|---|"]
    for name, _ in TRACKED:
        d = REPO / name
        if not d.is_dir():
            continue
        rows = collect(d)
        if not rows:
            continue
        out.append(f"| `{name}` | {len(rows)} | {sum(1 for r in rows if r['avg'] is not None)} "
                   f"| {sum(r['n'] for r in rows)} | {max(r['updated'] for r in rows)} |")
    return "\n".join(out)


BLOCKS = {"results": build_results_block, "inventory": build_inventory_block}


def splice(text: str, name: str, content: str) -> str:
    start, end = f"<!-- AUTO:{name} START -->", f"<!-- AUTO:{name} END -->"
    # tolerate an empty block (the two markers on consecutive lines)
    pat = re.compile(rf"{re.escape(start)}.*?{re.escape(end)}", re.DOTALL)
    if not pat.search(text):
        raise SystemExit(f"marker AUTO:{name} not found in {REPORT}")
    return pat.sub(lambda _: f"{start}\n{content}\n{end}", text, count=1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate results/REPORT.md AUTO blocks")
    ap.add_argument("--only", default=None,
                    help="comma-separated result dirs to include (default: all tracked)")
    ap.add_argument("--report", default=None,
                    help="path of the report to refresh (default results/REPORT.md)")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if regenerating would change the file")
    args = ap.parse_args()

    global REPORT, TRACKED
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        TRACKED = [t for t in TRACKED if t[0] in keep]
    if args.report:
        REPORT = Path(args.report)
    if not REPORT.exists():
        raise SystemExit(f"{REPORT} does not exist — create it with the AUTO markers first")

    old = REPORT.read_text()
    new = old
    for name, fn in BLOCKS.items():
        new = splice(new, name, fn())

    if args.check:
        if new != old:
            print(f"{REPORT} is stale — regenerate it")
            sys.exit(1)
        print(f"{REPORT} is up to date")
        return

    REPORT.write_text(new)
    print(f"wrote {REPORT}" + ("" if new != old else "  (no change)"))


if __name__ == "__main__":
    main()
