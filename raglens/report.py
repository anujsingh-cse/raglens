"""Aggregated evaluation report + HTML rendering."""

from __future__ import annotations

import html
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from raglens.attribution import AttributionReport


@dataclass(slots=True)
class SampleReport:
    query: str
    expected_answer: str | None
    tags: dict[str, str]
    metrics: dict[str, float | None]
    attribution: AttributionReport | None = None


@dataclass(slots=True)
class Report:
    samples: list[SampleReport]
    dataset_name: str
    dataset_version: str
    model: str
    lens_name: str = "RagLens"
    strategy: str = "counterfactual"

    # ------------------------------------------------------------------ #
    # Aggregations
    # ------------------------------------------------------------------ #

    def _metric_values(self, name: str) -> list[float]:
        return [s.metrics[name] for s in self.samples
                if s.metrics.get(name) is not None]

    def mean(self, name: str) -> float | None:
        vals = self._metric_values(name)
        return statistics.fmean(vals) if vals else None

    def std(self, name: str) -> float | None:
        vals = self._metric_values(name)
        return statistics.pstdev(vals) if len(vals) > 1 else 0.0

    def failure_mode_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self.samples:
            fm = s.attribution.failure_mode if s.attribution else "n/a"
            counts[fm] = counts.get(fm, 0) + 1
        return counts

    def total_judge_calls(self) -> int:
        return sum((s.attribution.n_judge_calls if s.attribution else 0)
                   for s in self.samples)

    # ------------------------------------------------------------------ #
    # Renderers
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        lines = [
            f"RagLens report — dataset={self.dataset_name} (v{self.dataset_version})",
            f"  model: {self.model} | strategy: {self.strategy}",
            f"  samples: {len(self.samples)} | llm judge calls: {self.total_judge_calls()}",
            "",
        ]
        for m in self.samples[0].metrics if self.samples else []:
            mean, std = self.mean(m), self.std(m)
            lines.append(f"  {m:<22s}  mean={mean:.3f}  std={std:.3f}"
                         if mean is not None else f"  {m:<22s}  n/a")
        counts = self.failure_mode_counts()
        if any(k != "n/a" for k in counts):
            lines.append("")
            lines.append("  failure modes:")
            for fm, n in sorted(counts.items()):
                lines.append(f"    {fm:<22s}  {n}")
        return "\n".join(lines)

    def attribution(self, top_k: int | None = None) -> str:
        lines = []
        for s in self.samples:
            if not s.attribution:
                continue
            lines.append(f"\nQ: {s.query}")
            lines.append(f"  faithfulness = {s.attribution.base_faithfulness:.3f}"
                         f"  failure_mode = {s.attribution.failure_mode}")
            helpful = s.attribution.helpful_chunks(top_k)
            harmful = s.attribution.harmful_chunks(top_k)
            if helpful:
                lines.append("  helpful:")
                for c in helpful:
                    lines.append(f"    +{c.attribution_score:+.3f}  {c.chunk_id}")
            if harmful:
                lines.append("  harmful:")
                for c in harmful:
                    lines.append(f"    {c.attribution_score:+.3f}  {c.chunk_id}")
            if s.attribution.conflicts:
                lines.append(f"  conflicts: {len(s.attribution.conflicts)}")
                for c in s.attribution.conflicts:
                    lines.append(f"    {c.chunk_a_id} <-> {c.chunk_b_id}  "
                                 f"(severity={c.severity:.2f})")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": {"name": self.dataset_name, "version": self.dataset_version},
            "model": self.model,
            "strategy": self.strategy,
            "n_samples": len(self.samples),
            "summary": {
                m: {"mean": self.mean(m), "std": self.std(m)}
                for m in (self.samples[0].metrics.keys() if self.samples else {})
            },
            "failure_modes": self.failure_mode_counts(),
            "samples": [self._sample_to_dict(s) for s in self.samples],
        }

    def _sample_to_dict(self, s: SampleReport) -> dict[str, Any]:
        out: dict[str, Any] = {
            "query": s.query,
            "expected_answer": s.expected_answer,
            "tags": s.tags,
            "metrics": s.metrics,
        }
        if s.attribution:
            out["attribution"] = {
                "base_faithfulness": s.attribution.base_faithfulness,
                "failure_mode": s.attribution.failure_mode,
                "strategy": s.attribution.strategy,
                "n_judge_calls": s.attribution.n_judge_calls,
                "chunks": [
                    {"id": c.chunk_id, "score": c.attribution_score}
                    for c in s.attribution.chunk_attributions
                ],
                "conflicts": [
                    {"a": c.chunk_a_id, "b": c.chunk_b_id,
                     "severity": c.severity, "explanation": c.explanation}
                    for c in s.attribution.conflicts
                ],
            }
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    def save_html(self, path: str | Path) -> None:
        """Render a self-contained diagnostic HTML report (no external deps)."""
        Path(path).write_text(_render_html(self), encoding="utf-8")


# ------------------------------------------------------------------ #
# Tiny no-template HTML renderer — keeps report.py dependency-free.
# ------------------------------------------------------------------ #

def _render_html(report: Report) -> str:
    metric_names = list(report.samples[0].metrics.keys()) if report.samples else []
    rows = []
    for i, s in enumerate(report.samples):
        metric_cells = "".join(
            f'<td class="num">{_fmt(s.metrics.get(m))}</td>'
            for m in metric_names)
        fm = s.attribution.failure_mode if s.attribution else "n/a"
        fm_color = {"ok": "#16a34a", "chunk_dominance": "#dc2626",
                    "retrieval_miss": "#d97706",
                    "generation_ignore": "#9333ea"}.get(fm, "#64748b")
        helpful = s.attribution.helpful_chunks(3) if s.attribution else []
        harmful = s.attribution.harmful_chunks(3) if s.attribution else []
        attr_html = ""
        if s.attribution:
            attr_html = (
                '<div class="attr"><b>Helpful:</b> '
                + ", ".join(f"{c.chunk_id} (+{c.attribution_score:.2f})"
                            for c in helpful) +
                '<br><b>Harmful:</b> '
                + ", ".join(f"{c.chunk_id} ({c.attribution_score:+.2f})"
                            for c in harmful)
                + f'<br><b>Conflicts:</b> {len(s.attribution.conflicts)}</div>'
            )
        rows.append(f"""
        <tr>
          <td>{i+1}</td>
          <td class="query">{_esc(s.query)}</td>
          {metric_cells}
          <td><span class="badge" style="background:{fm_color}">{_esc(fm)}</span></td>
          <td>{attr_html}</td>
        </tr>""")

    metric_headers = "".join(f"<th>{_esc(m)}</th>" for m in metric_names)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>RagLens report — {_esc(report.dataset_name)}</title>
<style>
  :root {{ --fg:#0f172a; --bg:#f8fafc; --border:#e2e8f0; }}
  body {{ font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
         background:var(--bg); color:var(--fg); margin:0; padding:24px; }}
  h1 {{ font-size: 20px; margin: 0 0 4px 0; }}
  .meta {{ color:#475569; font-size: 13px; margin-bottom: 18px; }}
  table {{ width:100%; border-collapse: collapse; background:#fff;
           border:1px solid var(--border); border-radius: 8px; overflow:hidden; }}
  th, td {{ text-align:left; padding:10px 12px; border-bottom:1px solid var(--border);
           font-size:13px; vertical-align: top; }}
  th {{ background:#f1f5f9; font-weight:600; font-size:12px;
        text-transform:uppercase; letter-spacing:0.04em; color:#475569; }}
  td.num {{ font-variant-numeric: tabular-nums; }}
  td.query {{ max-width: 360px; }}
  .badge {{ color:#fff; border-radius: 12px; padding: 2px 8px;
            font-size:11px; font-weight:600; }}
  .attr {{ font-size:12px; color:#334155; line-height:1.55; }}
  .bar-track {{ width: 120px; height: 8px; background:#e2e8f0;
                border-radius: 4px; position:relative; }}
  .bar-fill {{ height:100%; border-radius: 4px; }}
  .muted {{ color:#94a3b8; font-size:11px; }}
  .agg {{ display:flex; gap:24px; margin-bottom: 16px; flex-wrap:wrap; }}
  .agg div {{ background:#fff; padding: 10px 14px; border:1px solid var(--border);
              border-radius: 8px; min-width: 120px; }}
  .agg b {{ display:block; font-size:11px; color:#64748b;
           text-transform:uppercase; letter-spacing:0.04em; }}
  .agg span {{ font-size: 20px; font-weight:600; }}
</style></head><body>
  <h1>{_esc(report.lens_name)} report</h1>
  <div class="meta">dataset: {_esc(report.dataset_name)} (v{report.dataset_version})
    &middot; model: {_esc(report.model)} &middot; strategy: {_esc(report.strategy)}
    &middot; samples: {len(report.samples)} &middot; llm judge calls: {report.total_judge_calls()}</div>
  <div class="agg">
    {"".join(f"<div><b>{_esc(m)}</b><span>{_fmt(report.mean(m))}</span></div>" for m in metric_names)}
    {"".join(f"<div><b>{_esc(k)}</b><span>{v}</span></div>" for k, v in report.failure_mode_counts().items())}
  </div>
  <table>
    <thead><tr><th>#</th><th>query</th>{metric_headers}
      <th>failure mode</th><th>attribution</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</body></html>"""


def _fmt(v: float | None) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


def _esc(s: str) -> str:
    return html.escape(s, quote=True)
