"""Explainable, shareable reports for longitudinal ONA changes.

The report layer deliberately separates three things that are often conflated:
structural drivers (what changed in the org), metric movement (what changed in
network position), and demographic summaries (where movement is concentrated).
Driver effects are descriptive associations, not causal estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from pyduck_ona.temporal import DuckONATemporal, _parse_lookback, _quote

if TYPE_CHECKING:
    from collections.abc import Sequence

_PERIOD_FREQ = {"D": "D", "W": "W", "M": "M", "Q": "Q", "Y": "Y"}


def _period_series(values: pd.Series, freq: str) -> pd.Series:
    """Match pandas period labels to DuckDB date_trunc labels."""
    period_freq = _PERIOD_FREQ.get(freq.upper())
    if period_freq is None:
        raise ValueError(f"unsupported temporal frequency: {freq!r}")
    return pd.to_datetime(values, errors="coerce").dt.to_period(period_freq).dt.start_time.dt.strftime(
        "%Y-%m-%d"
    )


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


@dataclass
class ONAInsightReport:
    """A reproducible explanation of organizational network change.

    Attributes are public DataFrames so analysts can continue in pandas. The
    ``to_markdown`` and ``to_html`` methods provide safe, aggregate-first
    sharing formats; employee-level movers require ``include_individual=True``.
    """

    periods: tuple[str, str]
    headline: str
    driver_summary: pd.DataFrame
    metric_changes: pd.DataFrame
    driver_effects: pd.DataFrame
    demographic_summary: pd.DataFrame
    min_group_size: int = 5
    caveats: list[str] = field(default_factory=list)

    def to_markdown(self, *, include_individual: bool = False, max_rows: int = 25) -> str:
        """Render an aggregate-first Markdown brief."""
        start, end = self.periods
        lines = [
            "# Organizational Network Analysis Brief",
            "",
            f"**Period:** {start} to {end}",
            "",
            f"**Headline:** {self.headline}",
            "",
            "## Structural Drivers",
            "",
            self._table(self.driver_summary, max_rows),
            "",
            "## Metric Driver Associations",
            "",
            self._table(self.driver_effects, max_rows),
            "",
            "## Demographic Group Summary",
            "",
            self._table(self.demographic_summary, max_rows),
        ]
        if include_individual:
            lines.extend([
                "",
                "## Individual Metric Movers",
                "",
                self._table(self.metric_changes.sort_values("abs_delta", ascending=False), max_rows),
            ])
        lines.extend(["", "## Interpretation Notes", ""])
        lines.extend(f"- {note}" for note in self.caveats)
        return "\n".join(lines) + "\n"

    def to_html(self, *, include_individual: bool = False, title: str = "ONA Insight Brief") -> str:
        """Render a self-contained HTML brief suitable for sharing."""
        sections = [
            ("Structural Drivers", self.driver_summary),
            ("Metric Driver Associations", self.driver_effects),
            ("Demographic Group Summary", self.demographic_summary),
        ]
        if include_individual:
            sections.append(("Individual Metric Movers", self.metric_changes))
        tables = "".join(
            f"<section><h2>{escape(name)}</h2>{frame.to_html(index=False, classes='data')}</section>"
            for name, frame in sections
        )
        notes = "".join(f"<li>{escape(note)}</li>" for note in self.caveats)
        start, end = self.periods
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{escape(title)}</title>
<style>
body {{ font: 16px/1.5 system-ui, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #17202a; }}
h1 {{ color: #0b3d4f; }} h2 {{ border-bottom: 2px solid #d8e5e8; padding-bottom: .3rem; }}
.data {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
.data th, .data td {{ border: 1px solid #d8e5e8; padding: .4rem .55rem; text-align: left; }}
.data th {{ background: #eaf3f5; }} .headline {{ background: #f5f1df; padding: 1rem; border-left: 5px solid #d6a928; }}
</style></head><body>
<h1>{escape(title)}</h1><p><strong>Period:</strong> {escape(start)} to {escape(end)}</p>
<p class="headline"><strong>Headline:</strong> {escape(self.headline)}</p>
{tables}<section><h2>Interpretation Notes</h2><ul>{notes}</ul></section>
</body></html>"""

    def save(self, path: str | Path, *, include_individual: bool = False) -> Path:
        """Write Markdown or HTML based on the file suffix and return its path."""
        target = Path(path)
        if target.suffix.lower() in {".html", ".htm"}:
            target.write_text(self.to_html(include_individual=include_individual), encoding="utf-8")
        else:
            target.write_text(self.to_markdown(include_individual=include_individual), encoding="utf-8")
        return target

    @staticmethod
    def _table(frame: pd.DataFrame, max_rows: int) -> str:
        if frame.empty:
            return "_No data for this section._"
        view = frame.head(max_rows).copy()
        columns = [str(column) for column in view.columns]

        def cell(value: Any) -> str:
            if pd.isna(value):
                return ""
            return str(value).replace("|", "\\|").replace("\n", " ")

        rows = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        rows.extend(
            "| " + " | ".join(cell(value) for value in row) + " |"
            for row in view.itertuples(index=False, name=None)
        )
        return "\n".join(rows)


def build_insight_report(
    temporal: DuckONATemporal,
    *,
    lookback: str = "8Q",
    metrics: list[str] | None = None,
    demographic_columns: list[str] | None = None,
    min_group_size: int = 5,
) -> ONAInsightReport:
    """Build an explainable report from a loaded ``DuckONATemporal``.

    ``driver_effects`` quantify differences in metric movement between people
    affected and unaffected by each structural change. They are useful signals
    for investigation, not causal claims. Demographic rows below
    ``min_group_size`` are suppressed in aggregate outputs.
    """
    if not temporal._loaded:
        raise RuntimeError("call load_snapshots() first")
    if min_group_size < 1:
        raise ValueError("min_group_size must be >= 1")

    n_periods, _ = _parse_lookback(lookback)
    periods = temporal._periods[-n_periods:]
    if len(periods) < 2:
        raise ValueError("at least two temporal periods are required for an insight report")
    start, end = periods[0], periods[-1]
    emp = temporal._emp_col
    sup = temporal._sup_col
    date_col = temporal._date_col
    table = temporal._table_name

    snapshot = temporal.con.sql(f"SELECT * FROM {_quote(table)}").df()
    if snapshot.empty:
        raise ValueError("snapshot table is empty")
    snapshot["__period"] = _period_series(snapshot[date_col], temporal._freq)
    snapshot = snapshot[snapshot["__period"].isin(periods)].copy()
    snapshot = snapshot.sort_values(["__period", emp, date_col]).drop_duplicates(
        ["__period", emp], keep="last"
    )
    first = snapshot[snapshot["__period"] == start].copy()
    last = snapshot[snapshot["__period"] == end].copy()
    first_ids = set(first[emp].dropna())
    last_ids = set(last[emp].dropna())
    retained = first_ids & last_ids

    drivers = pd.DataFrame({emp: sorted(first_ids | last_ids, key=str)})
    drivers["joined"] = drivers[emp].isin(last_ids - first_ids)
    drivers["exited"] = drivers[emp].isin(first_ids - last_ids)
    first_cmp = first.set_index(emp)
    last_cmp = last.set_index(emp)
    common = sorted(retained, key=str)
    for name, column in (("manager_changed", sup), ("department_changed", "department"), ("level_changed", "job_level")):
        if column in first.columns and column in last.columns:
            before = first_cmp.reindex(common)[column]
            after = last_cmp.reindex(common)[column]
            drivers[name] = False
            drivers.loc[drivers[emp].isin(common), name] = (
                before.fillna("<NA>").astype(str).to_numpy()
                != after.fillna("<NA>").astype(str).to_numpy()
            )
        else:
            drivers[name] = False

    driver_names = ["joined", "exited", "manager_changed", "department_changed", "level_changed"]
    driver_summary = pd.DataFrame(
        {
            "driver": driver_names,
            "affected": [int(drivers[d].sum()) for d in driver_names],
            "share_of_population": [float(drivers[d].mean()) for d in driver_names],
        }
    )

    temporal_metrics = temporal.compute_temporal_metrics(metrics=metrics)
    if temporal_metrics.empty:
        metric_changes = _empty([emp, "metric", "start_value", "end_value", "delta", "abs_delta"])
    else:
        endpoint = temporal_metrics[temporal_metrics["period"].isin([start, end])]
        pivot = endpoint.pivot_table(index=["employee_id", "metric"], columns="period", values="value", aggfunc="last").reset_index()
        pivot = pivot.rename(columns={start: "start_value", end: "end_value"})
        for col in ("start_value", "end_value"):
            if col not in pivot:
                pivot[col] = pd.NA
        pivot["delta"] = pivot["end_value"] - pivot["start_value"]
        pivot["abs_delta"] = pivot["delta"].abs()
        metric_changes = pivot[["employee_id", "metric", "start_value", "end_value", "delta", "abs_delta"]].rename(
            columns={"employee_id": emp}
        )

    driver_effect_rows: list[dict[str, Any]] = []
    for metric, metric_frame in metric_changes.groupby("metric", sort=True):
        joined_metrics = metric_frame.merge(drivers, on=emp, how="left")
        for driver in driver_names:
            affected = joined_metrics.loc[joined_metrics[driver], "delta"].dropna()
            unaffected = joined_metrics.loc[~joined_metrics[driver], "delta"].dropna()
            driver_effect_rows.append(
                {
                    "metric": metric,
                    "driver": driver,
                    "affected_n": int(affected.size),
                    "affected_mean_delta": float(affected.mean()) if not affected.empty else float("nan"),
                    "unaffected_n": int(unaffected.size),
                    "unaffected_mean_delta": float(unaffected.mean()) if not unaffected.empty else float("nan"),
                    "mean_delta_difference": float(affected.mean() - unaffected.mean())
                    if not affected.empty and not unaffected.empty else float("nan"),
                }
            )
    driver_effects = pd.DataFrame(driver_effect_rows)

    demographic_summary = _demographic_summary(
        metric_changes, last, emp, demographic_columns, min_group_size
    )
    biggest = metric_changes.loc[metric_changes["abs_delta"].idxmax()] if not metric_changes.empty else None
    if biggest is None or pd.isna(biggest["delta"]):
        headline = f"Network structure changed between {start} and {end}; no comparable metric movement was available."
    else:
        direction = "increased" if biggest["delta"] > 0 else "decreased"
        headline = (
            f"The largest observed movement was {biggest['metric']} ({direction} by "
            f"{abs(float(biggest['delta'])):.3g}) between {start} and {end}."
        )
    caveats = [
        "Driver effects are descriptive associations and do not establish causality.",
        f"Demographic groups with fewer than {min_group_size} employees are suppressed.",
        "Metric changes compare the first and last selected periods; intermediate movements remain available in the temporal workspace.",
    ]
    return ONAInsightReport(
        periods=(start, end),
        headline=headline,
        driver_summary=driver_summary,
        metric_changes=metric_changes,
        driver_effects=driver_effects,
        demographic_summary=demographic_summary,
        min_group_size=min_group_size,
        caveats=caveats,
    )


def _demographic_summary(
    metric_changes: pd.DataFrame,
    last: pd.DataFrame,
    emp: str,
    columns: list[str] | None,
    min_group_size: int,
) -> pd.DataFrame:
    """Summarize metric deltas by latest-period demographic attributes."""
    if not columns or metric_changes.empty:
        return _empty(["demographic", "group", "metric", "n", "mean_delta", "median_delta", "pct_positive", "suppressed"])
    available = [column for column in columns if column in last.columns]
    if not available:
        return _empty(["demographic", "group", "metric", "n", "mean_delta", "median_delta", "pct_positive", "suppressed"])
    attrs = last[[emp, *available]].drop_duplicates(emp)
    merged = metric_changes.merge(attrs, on=emp, how="left")
    rows: list[dict[str, Any]] = []
    for demographic in available:
        for group, frame in merged.groupby(demographic, dropna=False, sort=True):
            for metric, metric_frame in frame.groupby("metric", sort=True):
                n = int(metric_frame[emp].nunique())
                suppressed = n < min_group_size
                rows.append(
                    {
                        "demographic": demographic,
                        "group": "[suppressed]" if suppressed else ("[missing]" if pd.isna(group) else group),
                        "metric": metric,
                        "n": n,
                        "mean_delta": float("nan") if suppressed else float(metric_frame["delta"].mean()),
                        "median_delta": float("nan") if suppressed else float(metric_frame["delta"].median()),
                        "pct_positive": float("nan") if suppressed else float((metric_frame["delta"] > 0).mean()),
                        "suppressed": suppressed,
                    }
                )
    return pd.DataFrame(rows)
