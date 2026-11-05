"""Compare two benchmark runs, print a table in the terminal and write an HTML page.

Both groups use this script:

- Group 1: run A = current model on the phone, run B = new model on the phone.
  Uses `<run>.tsv` and, if present, `<run>.summary.tsv` (speed and memory).
- Group 2: run A = phone transcript, run B = transcript improved by Claude.
  Uses the `latency_ms`, token and `status` columns of run B, if present.

Every run is a TSV file with the columns `file` and `hypothesis` (written by
device_benchmark.py, stt_benchmark.py or the Claude script). WER is calculated
against transcripts.tsv with the shared normalization (wer_report.py).

Decision rule (all thresholds can be changed with options):
  B is better if its WER is at least --min-gain percentage points lower than A,
  and B passes every limit that applies to it (RTF, memory, load time, LLM
  latency, failed LLM requests).
  A is better if B has a higher WER by more than --min-gain, or B fails a limit.
  Otherwise there is no clear difference.

Example:

    uv run --no-project --python 3.12 --with jiwer \
        python docs/stt-model-replacement/tools/compare_runs.py \
        docs/stt-model-replacement/results/group1/baseline.tsv \
        docs/stt-model-replacement/results/group1/nemo-480ms.tsv
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import jiwer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wer_report import load_tsv, normalize


@dataclass
class FileResult:
    name: str
    reference: str
    hypothesis: str
    errors: int
    words: int
    substitutions: int
    deletions: int
    insertions: int
    alignment: list = field(default_factory=list)

    @property
    def wer(self) -> float:
        return self.errors / self.words if self.words else 0.0


@dataclass
class Run:
    label: str
    path: Path
    files: dict[str, FileResult]
    summary: dict[str, str]
    rows: dict[str, dict[str, str]]

    @property
    def errors(self) -> int:
        return sum(f.errors for f in self.files.values())

    @property
    def words(self) -> int:
        return sum(f.words for f in self.files.values())

    @property
    def wer(self) -> float:
        return self.errors / self.words if self.words else 0.0

    def total(self, kind: str) -> int:
        return sum(getattr(f, kind) for f in self.files.values())

    def number(self, key: str) -> float | None:
        try:
            return float(self.summary[key])
        except (KeyError, ValueError):
            return None

    def column(self, *names: str) -> list[float]:
        for name in names:
            values = []
            for row in self.rows.values():
                try:
                    values.append(float(row[name]))
                except (KeyError, ValueError, TypeError):
                    pass
            if values:
                return values
        return []

    def failed_requests(self) -> int | None:
        statuses = [row.get("status") for row in self.rows.values()]
        if all(status is None for status in statuses):
            return None
        return sum(1 for status in statuses if status not in (None, "", "ok"))


def score(name: str, reference: str, hypothesis: str) -> FileResult:
    reference_n, hypothesis_n = normalize(reference), normalize(hypothesis)
    words = len(reference_n.split())
    if not hypothesis_n:
        return FileResult(name, reference_n, "", words, words, 0, words, 0)
    output = jiwer.process_words(reference_n, hypothesis_n)
    return FileResult(
        name,
        reference_n,
        hypothesis_n,
        output.substitutions + output.deletions + output.insertions,
        words,
        output.substitutions,
        output.deletions,
        output.insertions,
        output.alignments[0],
    )


def load_run(path: Path, label: str, references: dict[str, str]) -> Run:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = {row["file"]: row for row in csv.DictReader(handle, delimiter="\t")}
    hypotheses = load_tsv(path)
    files = {
        name: score(name, reference, hypotheses[name])
        for name, reference in references.items()
        if name in hypotheses
    }
    if not files:
        raise SystemExit(f"{path}: no file matches transcripts.tsv")
    summary_path = path.with_name(path.name[: -len(".tsv")] + ".summary.tsv")
    summary: dict[str, str] = {}
    if summary_path.exists():
        with summary_path.open(encoding="utf-8-sig", newline="") as handle:
            summary = {
                row["key"]: row["value"]
                for row in csv.DictReader(handle, delimiter="\t")
            }
    return Run(label, path, files, summary, rows)


# --------------------------------------------------------------------------- decision


@dataclass
class Limit:
    name: str
    value: float | None
    limit: float
    unit: str

    @property
    def applies(self) -> bool:
        return self.value is not None

    @property
    def passed(self) -> bool:
        return self.value is None or self.value <= self.limit


def limits_for(run: Run, args: argparse.Namespace) -> list[Limit]:
    latencies = run.column("latency_ms", "process_ms")
    return [
        Limit("RTF on the phone", run.number("rtf"), args.max_rtf, ""),
        Limit("Peak memory (PSS)", run.number("pss_peak_mb"), args.max_pss_mb, " MB"),
        Limit("Model load time", run.number("model_load_ms"), args.max_load_ms, " ms"),
        Limit(
            "Median LLM latency",
            statistics.median(latencies) if latencies else None,
            args.max_latency_ms,
            " ms",
        ),
        Limit("Failed LLM requests", run.failed_requests(), args.max_failed, ""),
    ]


def decide(a: Run, b: Run, args: argparse.Namespace) -> tuple[str, str, list[str]]:
    gain = (a.wer - b.wer) * 100
    failed = [
        limit for limit in limits_for(b, args) if limit.applies and not limit.passed
    ]
    reasons = [
        f"{limit.name} {limit.value:.3g}{limit.unit} is above the limit {limit.limit:g}{limit.unit}"
        for limit in failed
    ]
    if failed:
        return (
            "a",
            f"{a.label} is better",
            reasons + [f"WER change: {(b.wer - a.wer) * 100:+.1f} points"],
        )
    measured = [limit.name for limit in limits_for(b, args) if limit.applies]
    limits_note = (
        "All measured limits passed: " + ", ".join(measured)
        if measured
        else "No speed, memory or latency limits measured for B"
    )
    if gain >= args.min_gain:
        return (
            "b",
            f"{b.label} is better",
            [
                f"WER {gain:.1f} points lower ({a.wer:.1%} to {b.wer:.1%})",
                limits_note,
            ],
        )
    if -gain > args.min_gain:
        return (
            "a",
            f"{a.label} is better",
            [f"WER {-gain:.1f} points higher with {b.label}"],
        )
    return (
        "none",
        "No clear difference",
        [
            f"WER change {(b.wer - a.wer) * 100:+.1f} points is smaller than the minimum gain {args.min_gain:g}"
        ],
    )


# --------------------------------------------------------------------------- terminal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8", errors="replace"
    )  # box characters on Windows
USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ
if os.name == "nt" and USE_COLOR:
    os.system("")  # enables ANSI colors in the Windows console


def paint(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def table(headers: list[str], rows: list[list[str]], right: set[int]) -> str:
    def visible(text: str) -> int:
        length, skip = 0, False
        for char in text:
            if char == "\033":
                skip = True
            elif skip and char == "m":
                skip = False
            elif not skip:
                length += 1
        return length

    widths = [max(visible(r[i]) for r in [headers, *rows]) for i in range(len(headers))]

    def line(cells: list[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            pad = " " * (widths[i] - visible(cell))
            parts.append(pad + cell if i in right else cell + pad)
        return "│ " + " │ ".join(parts) + " │"

    border = lambda l, m, r: l + m.join("─" * (w + 2) for w in widths) + r
    return "\n".join(
        [border("┌", "┬", "┐"), line(headers), border("├", "┼", "┤")]
        + [line(r) for r in rows]
        + [border("└", "┴", "┘")]
    )


def delta(
    a: float | None, b: float | None, fmt: str, lower_is_better: bool = True
) -> str:
    if a is None or b is None:
        return "–"
    diff = b - a
    text = format(diff, "+" + fmt)
    if abs(diff) < 1e-12:
        return text
    good = (diff < 0) == lower_is_better
    return paint(text, "32" if good else "31")


def fmt(value: float | None, spec: str, suffix: str = "") -> str:
    return "–" if value is None else format(value, spec) + suffix


def metric_rows(
    a: Run, b: Run, args: argparse.Namespace
) -> list[tuple[str, float | None, float | None, str, str]]:
    """(name, value A, value B, format, unit)."""
    rows = [
        ("WER (%)", a.wer * 100, b.wer * 100, ".1f", ""),
        ("Word errors", a.errors, b.errors, ".0f", ""),
        (
            "  substitutions",
            a.total("substitutions"),
            b.total("substitutions"),
            ".0f",
            "",
        ),
        ("  deletions", a.total("deletions"), b.total("deletions"), ".0f", ""),
        ("  insertions", a.total("insertions"), b.total("insertions"), ".0f", ""),
        (
            "Files without errors",
            sum(f.errors == 0 for f in a.files.values()),
            sum(f.errors == 0 for f in b.files.values()),
            ".0f",
            "",
        ),
    ]
    for key, name, spec, unit in (
        ("rtf", "RTF on the phone", ".3f", ""),
        ("model_load_ms", "Model load time (ms)", ".0f", ""),
        ("pss_after_load_mb", "Memory after load (MB)", ".0f", ""),
        ("pss_peak_mb", "Peak memory (MB)", ".0f", ""),
    ):
        if a.number(key) is not None or b.number(key) is not None:
            rows.append((name, a.number(key), b.number(key), spec, unit))
    for run_key, name in (("latency_ms", "LLM latency, median (ms)"),):
        la, lb = a.column(run_key, "process_ms"), b.column(run_key, "process_ms")
        if la or lb:
            rows.append(
                (
                    name,
                    statistics.median(la) if la else None,
                    statistics.median(lb) if lb else None,
                    ".0f",
                    "",
                )
            )
            rows.append(
                (
                    "LLM latency, max (ms)",
                    max(la) if la else None,
                    max(lb) if lb else None,
                    ".0f",
                    "",
                )
            )
    for run, other in ((b, a),):
        tokens_in, tokens_out = run.column("input_tokens"), run.column("output_tokens")
        if tokens_in and tokens_out:
            cost = (
                sum(tokens_in) * args.price_in + sum(tokens_out) * args.price_out
            ) / 1e6
            per_request = cost / len(tokens_in)
            rows.append(("LLM cost per request (USD)", None, per_request, ".5f", ""))
            rows.append(
                (
                    "LLM cost per 1,000 requests (USD)",
                    None,
                    per_request * 1000,
                    ".2f",
                    "",
                )
            )
        failed = run.failed_requests()
        if failed is not None:
            rows.append(("Failed requests", None, failed, ".0f", ""))
    return rows


def change_counts(a: Run, b: Run) -> dict[str, int]:
    common = [n for n in a.files if n in b.files]
    counts = {
        "improved": 0,
        "unchanged": 0,
        "worse": 0,
        "fixed": 0,
        "introduced": 0,
        "over": 0,
    }
    for name in common:
        ea, eb = a.files[name].errors, b.files[name].errors
        counts["improved" if eb < ea else "worse" if eb > ea else "unchanged"] += 1
        counts["fixed"] += max(0, ea - eb)
        counts["introduced"] += max(0, eb - ea)
        if ea == 0 and eb > 0:
            counts["over"] += 1  # B broke a file that A had right
    return counts


def print_report(a: Run, b: Run, args: argparse.Namespace, verdict, counts) -> None:
    print()
    print(paint(f"A = {a.label}", "1") + f"   ({a.path})")
    print(paint(f"B = {b.label}", "1") + f"   ({b.path})")
    if a.summary.get("device") or b.summary.get("device"):
        print(
            f"Phone: {b.summary.get('device') or a.summary.get('device')}, "
            f"Android {b.summary.get('android') or a.summary.get('android')}"
        )
    print()
    rows = []
    for name, va, vb, spec, unit in metric_rows(a, b, args):
        lower = not name.startswith("Files without")
        rows.append(
            [name, fmt(va, spec, unit), fmt(vb, spec, unit), delta(va, vb, spec, lower)]
        )
    print(table(["Metric", "A", "B", "B - A"], rows, {1, 2, 3}))
    print()
    file_rows = []
    for name in a.files:
        fa, fb = a.files[name], b.files.get(name)
        file_rows.append(
            [
                name,
                f"{fa.wer:.1%}",
                f"{fb.wer:.1%}" if fb else "missing",
                delta(fa.wer * 100, fb.wer * 100, ".1f") if fb else "–",
            ]
        )
    print(table(["File", "WER A", "WER B", "B - A (points)"], file_rows, {1, 2, 3}))
    print()
    print(
        f"Files improved / unchanged / worse: {counts['improved']} / {counts['unchanged']} / {counts['worse']}"
        f"   Errors fixed: {counts['fixed']}   Errors introduced: {counts['introduced']}"
        f"   Files broken by B: {counts['over']}"
    )
    winner, title, reasons = verdict
    color = {"b": "32", "a": "31", "none": "33"}[winner]
    print()
    print(paint(f"Verdict: {title}", "1;" + color))
    for reason in reasons:
        print(f"  - {reason}")


# --------------------------------------------------------------------------- HTML

CSS = """
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--plane:#f9f9f7;--text-primary:#0b0b0b;
--text-secondary:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
--series-1:#2a78d6;--series-2:#eb6834;--good:#0ca30c;--critical:#d03b3b;--warning:#fab219;
--good-text:#006300;--bad-text:#b42318;--mark-err:#fde4e1;--mark-err-ink:#8a1c12;}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
--surface-1:#1a1a19;--plane:#0d0d0d;--text-primary:#f0efec;--text-secondary:#c3c2b7;--grid:#2c2c2a;
--axis:#383835;--border:rgba(255,255,255,.10);--series-1:#3987e5;--series-2:#d95926;--good-text:#0ca30c;
--bad-text:#f08a80;--mark-err:#4a1f1c;--mark-err-ink:#ffd2cc;}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--plane:#0d0d0d;
--text-primary:#f0efec;--text-secondary:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--series-1:#3987e5;--series-2:#d95926;--good-text:#0ca30c;--bad-text:#f08a80;--mark-err:#4a1f1c;--mark-err-ink:#ffd2cc;}
*{box-sizing:border-box}
body{margin:0}
.viz-root{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--plane);
color:var(--text-primary);min-height:100vh;padding:32px;line-height:1.45}
main{max-width:1100px;margin:0 auto}
h1{font-size:24px;margin:0 0 4px}
h2{font-size:17px;margin:32px 0 12px}
.sub{color:var(--text-secondary);margin:0 0 20px;font-size:14px}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:20px}
.verdict{display:flex;gap:14px;align-items:flex-start;border-left:6px solid var(--muted)}
.verdict.b{border-left-color:var(--good)}.verdict.a{border-left-color:var(--critical)}
.verdict.none{border-left-color:var(--warning)}
.verdict .icon{font-size:26px;line-height:1}
.verdict h2{margin:0 0 6px;font-size:20px}
.verdict ul{margin:0;padding-left:18px;color:var(--text-secondary)}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}
.tile .name{font-size:13px;color:var(--text-secondary)}
.tile .pair{display:flex;gap:18px;margin-top:6px;align-items:baseline}
.tile .val{font-size:22px;font-weight:600}
.tile .tag{font-size:12px;color:var(--text-secondary);display:flex;align-items:center;gap:6px}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.dot.a{background:var(--series-1)}.dot.b{background:var(--series-2)}
.tile .delta{font-size:13px;margin-top:6px}
.up{color:var(--good-text)}.down{color:var(--bad-text)}
.legend{display:flex;gap:20px;font-size:13px;color:var(--text-secondary);margin-bottom:8px}
.legend span{display:flex;align-items:center;gap:6px}
svg text{fill:var(--muted);font-size:12px}
svg .grid{stroke:var(--grid);stroke-width:1}
svg .axis{stroke:var(--axis);stroke-width:1}
svg .bar-a{fill:var(--series-1)}svg .bar-b{fill:var(--series-2)}
svg .hit{fill:transparent}
svg g.bar:hover path,svg g.bar:focus path{opacity:.75}
svg g.bar:focus{outline:none}
svg g.bar:focus path{stroke:var(--text-primary);stroke-width:2}
svg .label{fill:var(--text-secondary);font-size:11px}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--text-secondary);font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.mono{font-family:ui-monospace,Consolas,monospace;font-size:13px}
mark{background:var(--mark-err);color:var(--mark-err-ink);border-radius:3px;padding:0 2px}
del{color:var(--bad-text)}
.small{font-size:12px;color:var(--text-secondary)}
"""


def esc(text: object) -> str:
    return html.escape(str(text))


def highlighted(result: FileResult | None) -> str:
    """Hypothesis text with wrong words marked; deleted reference words struck through."""
    if result is None:
        return '<span class="small">missing</span>'
    if not result.hypothesis:
        return '<span class="small">(empty)</span>'
    ref, hyp = result.reference.split(), result.hypothesis.split()
    parts = []
    for chunk in result.alignment:
        if chunk.type == "equal":
            parts.append(esc(" ".join(hyp[chunk.hyp_start_idx : chunk.hyp_end_idx])))
        elif chunk.type in ("substitute", "insert"):
            words = esc(" ".join(hyp[chunk.hyp_start_idx : chunk.hyp_end_idx]))
            parts.append(f'<mark title="{chunk.type}">{words}</mark>')
        elif chunk.type == "delete":
            words = esc(" ".join(ref[chunk.ref_start_idx : chunk.ref_end_idx]))
            parts.append(f'<del title="missing word">{words}</del>')
    return " ".join(parts)


def bar_path(x: float, y: float, width: float, height: float, radius: float = 4) -> str:
    """Bar with a 4px rounded top and a square end at the baseline."""
    if height <= 0:
        return ""
    r = min(radius, width / 2, height)
    return (
        f"M{x:.1f},{y + height:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
        f"H{x + width - r:.1f} Q{x + width:.1f},{y:.1f} {x + width:.1f},{y + r:.1f} "
        f"V{y + height:.1f} Z"
    )


def chart(a: Run, b: Run) -> str:
    names = list(a.files)
    values = [(a.files[n].wer, b.files[n].wer if n in b.files else 0.0) for n in names]
    top = max([v for pair in values for v in pair] + [0.1])
    step = 0.1 if top <= 0.6 else 0.2
    axis_max = step * (int(top / step) + 1)
    width, height, left, right, plot_top, bottom = 1040, 300, 48, 12, 12, 46
    plot_w, plot_h = width - left - right, height - plot_top - bottom
    band = plot_w / len(names)
    bar_w = min(24, (band - 16) / 2)
    out = [
        (
            f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
            f'aria-label="WER per file, {esc(a.label)} and {esc(b.label)}">'
        )
    ]
    tick = 0.0
    while tick <= axis_max + 1e-9:
        y = plot_top + plot_h - tick / axis_max * plot_h
        out.append(
            f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}"/>'
        )
        out.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{tick:.0%}</text>'
        )
        tick += step
    base = plot_top + plot_h
    out.append(
        f'<line class="axis" x1="{left}" x2="{width - right}" y1="{base}" y2="{base}"/>'
    )
    # Direct labels only where they matter: the largest gain and every file that got worse.
    best = min(range(len(values)), key=lambda i: values[i][1] - values[i][0])
    for index, (name, (wa, wb)) in enumerate(zip(names, values)):
        center = left + band * index + band / 2
        for value, css, label, x in (
            (wa, "bar-a", a.label, center - bar_w - 1),
            (wb, "bar-b", b.label, center + 1),
        ):
            h = value / axis_max * plot_h
            out.append(
                f'<g class="bar" tabindex="0"><title>{esc(name)} · {esc(label)}: WER {value:.1%}</title>'
                f'<rect class="hit" x="{x - 1:.1f}" y="{plot_top}" width="{bar_w + 2:.1f}" height="{plot_h}"/>'
                f'<path class="{css}" d="{bar_path(x, base - h, bar_w, h)}"/></g>'
            )
        short = name.replace(".wav", "")
        out.append(
            f'<text x="{center:.1f}" y="{base + 18}" text-anchor="middle">{esc(short)}</text>'
        )
        if (index == best and wb < wa) or wb > wa:
            out.append(
                f'<text class="label" x="{center:.1f}" y="{base + 34}" text-anchor="middle">'
                f"{(wb - wa) * 100:+.0f} pts</text>"
            )
    out.append("</svg>")
    return "".join(out)


def tile(
    name: str,
    va: float | None,
    vb: float | None,
    spec: str,
    lower_is_better: bool = True,
) -> str:
    diff = ""
    if va is not None and vb is not None and abs(vb - va) > 1e-12:
        good = (vb < va) == lower_is_better
        arrow = "▼" if vb < va else "▲"
        diff = (
            f'<div class="delta {"up" if good else "down"}">{arrow} {format(vb - va, "+" + spec)} '
            f"({'better' if good else 'worse'})</div>"
        )
    return (
        f'<div class="card tile"><div class="name">{esc(name)}</div><div class="pair">'
        f'<div><div class="tag"><span class="dot a"></span>A</div><div class="val">{fmt(va, spec)}</div></div>'
        f'<div><div class="tag"><span class="dot b"></span>B</div><div class="val">{fmt(vb, spec)}</div></div>'
        f"</div>{diff}</div>"
    )


def write_html(path: Path, a: Run, b: Run, args, verdict, counts) -> None:
    winner, title, reasons = verdict
    icon = {"b": "✅", "a": "⛔", "none": "⚖️"}[winner]
    metrics = metric_rows(a, b, args)
    tiles = "".join(
        tile(name, va, vb, spec, not name.startswith("Files without"))
        for name, va, vb, spec, _ in metrics
        if not name.startswith("  ")
    )
    metric_table = "".join(
        f"<tr><td>{esc(name)}</td><td class='num'>{fmt(va, spec)}</td><td class='num'>{fmt(vb, spec)}</td>"
        f"<td class='num'>{'–' if va is None or vb is None else format(vb - va, '+' + spec)}</td></tr>"
        for name, va, vb, spec, _ in metrics
    )
    file_rows = "".join(
        f"<tr><td class='mono'>{esc(n)}</td><td class='num'>{a.files[n].wer:.1%}</td>"
        f"<td class='num'>{b.files[n].wer:.1%}</td>"
        f"<td class='num'>{(b.files[n].wer - a.files[n].wer) * 100:+.1f}</td>"
        f"<td class='num'>{a.files[n].errors} / {b.files[n].errors}</td></tr>"
        for n in a.files
        if n in b.files
    )
    texts = "".join(
        f"<tr><td class='mono'>{esc(n)}</td><td>{esc(a.files[n].reference)}</td>"
        f"<td>{highlighted(a.files[n])}</td><td>{highlighted(b.files.get(n))}</td></tr>"
        for n in a.files
    )
    limit_rows = "".join(
        f"<tr><td>{esc(limit.name)}</td><td class='num'>{limit.limit:g}{esc(limit.unit)}</td>"
        f"<td class='num'>{fmt(limit.value, '.3g')}{esc(limit.unit) if limit.applies else ''}</td>"
        f"<td>{'not measured' if not limit.applies else '✅ pass' if limit.passed else '⛔ fail'}</td></tr>"
        for limit in limits_for(b, args)
    )
    device = b.summary.get("device") or a.summary.get("device")
    device_line = (
        f" · Phone: {esc(device)}, Android {esc(b.summary.get('android') or a.summary.get('android'))}"
        if device
        else ""
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>STT benchmark: {esc(a.label)} vs {esc(b.label)}</title><style>{CSS}</style></head>
<body><div class="viz-root"><main>
<h1>Benchmark comparison</h1>
<p class="sub"><span class="dot a"></span> A = <b>{esc(a.label)}</b> ({esc(a.path.name)}) &nbsp;·&nbsp;
<span class="dot b"></span> B = <b>{esc(b.label)}</b> ({esc(b.path.name)}){device_line}
 · {len(a.files)} files, {a.words} reference words · {datetime.now():%Y-%m-%d %H:%M}</p>

<section class="card verdict {winner}" aria-live="polite"><div class="icon" aria-hidden="true">{icon}</div>
<div><h2>Verdict: {esc(title)}</h2><ul>{"".join(f"<li>{esc(r)}</li>" for r in reasons)}</ul>
<p class="small">Files improved / unchanged / worse: {counts["improved"]} / {counts["unchanged"]} / {counts["worse"]}
 · errors fixed {counts["fixed"]} · errors introduced {counts["introduced"]} · files broken by B {counts["over"]}</p></div></section>

<h2>Key metrics</h2><div class="tiles">{tiles}</div>

<h2>WER per file</h2><div class="card">
<div class="legend"><span><span class="dot a"></span>A: {esc(a.label)}</span>
<span><span class="dot b"></span>B: {esc(b.label)}</span><span>Lower is better</span></div>
{chart(a, b)}</div>

<h2>All metrics</h2><div class="card"><table><thead><tr><th>Metric</th><th class="num">A</th>
<th class="num">B</th><th class="num">B − A</th></tr></thead><tbody>{metric_table}</tbody></table></div>

<h2>Per file</h2><div class="card"><table><thead><tr><th>File</th><th class="num">WER A</th>
<th class="num">WER B</th><th class="num">B − A (points)</th><th class="num">Errors A / B</th></tr></thead>
<tbody>{file_rows}</tbody></table></div>

<h2>Transcripts</h2><div class="card"><p class="small">Normalized text (lowercase, no punctuation).
<mark>marked</mark> = wrong or extra word, <del>struck</del> = missing word.</p>
<table><thead><tr><th>File</th><th>Reference</th><th>A: {esc(a.label)}</th><th>B: {esc(b.label)}</th></tr></thead>
<tbody>{texts}</tbody></table></div>

<h2>Decision rule</h2><div class="card"><p class="small">B is better if its WER is at least
{args.min_gain:g} points lower than A and B passes every limit below that was measured.</p>
<table><thead><tr><th>Limit for B</th><th class="num">Maximum</th><th class="num">B</th><th>Result</th></tr></thead>
<tbody>{limit_rows}</tbody></table></div>
</main></div></body></html>
"""
    path.write_text(page, encoding="utf-8")


# --------------------------------------------------------------------------- main


def main() -> None:
    default_reference = (
        Path(__file__).resolve().parent.parent / "testset" / "transcripts.tsv"
    )
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "run_a", type=Path, help="Result TSV of run A (the reference run)."
    )
    parser.add_argument("run_b", type=Path, help="Result TSV of run B (the new run).")
    parser.add_argument("--label-a", help="Name of run A. Default: file name.")
    parser.add_argument("--label-b", help="Name of run B. Default: file name.")
    parser.add_argument("--reference", default=default_reference, type=Path)
    parser.add_argument(
        "--html", type=Path, help="Output HTML file. Default: next to run B."
    )
    parser.add_argument(
        "--min-gain", type=float, default=1.0, help="Minimum WER gain in points."
    )
    parser.add_argument(
        "--max-rtf", type=float, default=0.8, help="Limit for the phone RTF."
    )
    parser.add_argument(
        "--max-pss-mb", type=float, default=1024, help="Limit for peak memory."
    )
    parser.add_argument(
        "--max-load-ms", type=float, default=15000, help="Limit for model load time."
    )
    parser.add_argument(
        "--max-latency-ms",
        type=float,
        default=3000,
        help="Limit for median LLM latency.",
    )
    parser.add_argument(
        "--max-failed", type=float, default=0, help="Limit for failed LLM requests."
    )
    parser.add_argument(
        "--price-in", type=float, default=4.0, help="USD per 1M input tokens."
    )
    parser.add_argument(
        "--price-out", type=float, default=20.0, help="USD per 1M output tokens."
    )
    args = parser.parse_args()

    references = load_tsv(args.reference)
    a = load_run(args.run_a, args.label_a or args.run_a.stem, references)
    b = load_run(args.run_b, args.label_b or args.run_b.stem, references)
    verdict = decide(a, b, args)
    counts = change_counts(a, b)
    print_report(a, b, args, verdict, counts)

    html_path = args.html or args.run_b.with_name(
        f"compare_{args.run_a.stem}_vs_{args.run_b.stem}.html"
    )
    write_html(html_path, a, b, args, verdict, counts)
    print(f"\nHTML report: {html_path.resolve()}")


if __name__ == "__main__":
    main()
