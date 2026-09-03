"""
Network visualizations for ONA outputs.

- :func:`centrality_dashboard` - 2×2 panel comparing the four classical
  centrality measures (betweenness, pagerank, eigenvector, degree).
- :func:`silo_map`             - Community-coloured network map (pyvis
  interactive HTML + matplotlib static fallback).
"""
from __future__ import annotations

from typing import Any

import networkx as nx  # noqa: TC002 (used at runtime in silo_map)
import pandas as pd

from .theme import (
    CATEGORICAL,
    PALETTE,
    apply_default_style,
    category_colors,
    configure_axes,
    new_figure,
)

# ---------------------------------------------------------------------------
# Centrality dashboard
# ---------------------------------------------------------------------------

_CENTRALITY_PANELS: list[tuple[str, str, str, str]] = [
    # (panel title, score column, x-axis label, default dataframe column)
    ("Betweenness Centrality",  "betweenness",  "betweenness",  "node_id"),
    ("PageRank",                "pagerank",      "pagerank",      "node_id"),
    ("Eigenvector Centrality",  "eigenvector",   "eigenvector",   "node_id"),
    ("Degree Centrality",       "degree",        "degree",        "node_id"),
]


def centrality_dashboard(
    betweenness: pd.DataFrame,
    pagerank: pd.DataFrame,
    eigenvector: pd.DataFrame,
    degree: pd.DataFrame,
    *,
    id_col: str = "node_id",
    betweenness_col: str = "betweenness",
    pagerank_col: str = "pagerank",
    eigenvector_col: str = "eigenvector",
    degree_col: str = "degree",
    metadata: pd.DataFrame | None = None,
    name_col: str = "name",
    department_col: str | None = "department",
    top_n: int = 12,
    title: str | None = None,
    figsize: tuple[float, float] = (13.0, 9.0),
) -> Any:
    """Plot a 2×2 grid comparing four centrality measures.

    Each panel is a horizontal bar chart of the top ``top_n`` nodes by
    the panel's score. If ``metadata`` and ``department_col`` are supplied,
    bars are coloured by department.

    Returns
    -------
    matplotlib.figure.Figure
    """
    frames: list[tuple[pd.DataFrame, str, str, str]] = [
        (betweenness, betweenness_col, "Betweenness Centrality",  "Broker nodes — control information flow."),
        (pagerank,      pagerank_col,     "PageRank",               "Most-influential nodes by random-walk mass."),
        (eigenvector,   eigenvector_col,  "Eigenvector Centrality", "Connected to other well-connected nodes."),
        (degree,        degree_col,       "Degree Centrality",      "Raw direct connections (in / out degree)."),
    ]

    # Resolve names + departments
    name_lookup: dict[str, str] = {}
    dept_lookup: dict[str, str] = {}
    if metadata is not None and id_col in metadata.columns:
        for _, row in metadata.iterrows():
            key = str(row[id_col])
            if name_col in metadata.columns:
                name_lookup[key] = str(row[name_col])
            if department_col and department_col in metadata.columns:
                v = row[department_col]
                if not pd.isna(v):
                    dept_lookup[key] = str(v)

    apply_default_style()
    fig, axes = new_figure(figsize=figsize, nrows=2, ncols=2)

    title = title or "Network Centrality Dashboard"
    fig.suptitle(title, fontsize=18, fontweight="semibold",
                 color=PALETTE["primary"], x=0.06, ha="left", y=0.99)

    for ax, (df, col, panel_title, subtitle) in zip(axes.flat, frames, strict=False):
        if id_col not in df.columns:
            raise KeyError(f"Column '{id_col}' not in one of the centrality frames")
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not in one of the centrality frames")
        work = df[[id_col, col]].copy()
        work[col] = work[col].fillna(0).astype(float)
        work = work.sort_values(col, ascending=False).head(int(top_n))
        work = work.sort_values(col, ascending=True)

        labels = [name_lookup.get(str(k), str(k)) for k in work[id_col]]
        values = work[col].astype(float).tolist()

        if dept_lookup:
            depts = [dept_lookup.get(str(k), "Unknown") for k in work[id_col]]
            distinct = sorted(set(depts))
            palette = {d: category_colors(len(distinct))[i]
                       for i, d in enumerate(distinct)}
            bar_colors: str | list[str] = [palette[d] for d in depts]
        else:
            bar_colors = PALETTE["primary"]

        ax.barh(range(len(labels)), values,
                color=bar_colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(panel_title, loc="left", pad=18, fontsize=14,
                     fontweight="semibold", color=PALETTE["primary"])
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes,
                fontsize=9.5, color=PALETTE["neutral_lt"], ha="left", va="bottom")
        configure_axes(ax, grid=True, grid_axis="x")
        ax.set_xlabel(col.replace("_", " "), fontsize=10)
        for i, v in enumerate(values):
            ax.text(v, i, f" {v:.3g}", va="center", ha="left",
                    fontsize=8, color=PALETTE["neutral"])

    return fig


# ---------------------------------------------------------------------------
# Silo map (network of communities)
# ---------------------------------------------------------------------------

def silo_map(
    edges: pd.DataFrame,
    components: pd.DataFrame | None = None,
    communities: pd.DataFrame | None = None,
    *,
    source_col: str = "employee_id",
    target_col: str = "supervisor_id",
    node_col: str = "node_id",
    component_col: str = "component",
    community_col: str = "community",
    metadata: pd.DataFrame | None = None,
    name_col: str = "name",
    department_col: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (12.0, 9.0),
    return_html: bool = False,
    physics: bool = True,
) -> Any:
    """Render an organisational silo map.

    If both ``components`` and ``communities`` are given, ``communities``
    takes precedence (it's the more analytically interesting partition).

    Parameters
    ----------
    return_html
        If True, return pyvis interactive HTML. If False, return a matplotlib
        Figure with the same communities laid out via a force-directed
        spring layout.

    Returns
    -------
    str (HTML) or matplotlib.figure.Figure
    """
    import networkx as nx

    if source_col not in edges.columns or target_col not in edges.columns:
        raise KeyError(
            f"Edge DataFrame must contain '{source_col}' and '{target_col}'"
        )

    graph: nx.DiGraph[Any] = nx.DiGraph()
    for _, row in edges.iterrows():
        if pd.isna(row[target_col]):
            continue
        graph.add_edge(str(row[source_col]), str(row[target_col]))

    partition_source = communities if communities is not None else components
    partition_col = community_col if communities is not None else component_col

    if partition_source is None or partition_col not in partition_source.columns:
        # Default: every node is its own partition
        node_partition: dict[str, int] = {n: i for i, n in enumerate(graph.nodes())}
    else:
        node_partition = {
            str(row[node_col]): int(row[partition_col])
            for _, row in partition_source.iterrows()
            if not pd.isna(row[partition_col])
        }

    # Metadata: name + department
    name_lookup: dict[str, str] = {}
    dept_lookup: dict[str, str] = {}
    if metadata is not None and node_col in metadata.columns:
        for _, row in metadata.iterrows():
            key = str(row[node_col])
            if name_col in metadata.columns:
                name_lookup[key] = str(row[name_col])
            if department_col and department_col in metadata.columns:
                v = row[department_col]
                if not pd.isna(v):
                    dept_lookup[key] = str(v)

    # Stable colour map per partition id
    pids = sorted(set(node_partition.values()))
    palette = {pid: CATEGORICAL[i % len(CATEGORICAL)] for i, pid in enumerate(pids)}

    title = title or "Organizational Silo Map"
    subtitle = (
        f"Coloured by Louvain communities · {len(pids)} silos"
        if communities is not None else
        f"Coloured by connected components · {len(pids)} components"
    )

    if return_html:
        return _silo_map_pyvis(
            graph, node_partition, palette, name_lookup, dept_lookup,
            title=title, subtitle=subtitle, physics=physics,
        )
    return _silo_map_matplotlib(
        graph, node_partition, palette, name_lookup, dept_lookup,
        title=title, subtitle=subtitle, figsize=figsize,
    )


def _silo_map_matplotlib(
    graph: nx.DiGraph[Any],
    node_partition: dict[str, int],
    palette: dict[int, str],
    name_lookup: dict[str, str],
    dept_lookup: dict[str, str],
    *,
    title: str,
    subtitle: str,
    figsize: tuple[float, float],
) -> Any:
    import networkx as nx

    apply_default_style()
    fig, ax = new_figure(figsize=figsize)

    # Use an undirected view for the layout to avoid arrowhead clutter.
    undirected = graph.to_undirected()
    pos = nx.spring_layout(undirected, seed=42, k=1.0 / max(len(undirected), 1) ** 0.5)

    # Node sizes proportional to degree
    deg = dict(graph.degree())
    sizes = [30 + 18 * deg.get(n, 0) for n in undirected.nodes()]
    colors = [palette[node_partition.get(n, 0)] for n in undirected.nodes()]

    nx.draw_networkx_edges(
        undirected, pos, ax=ax, alpha=0.25, edge_color=PALETTE["neutral_lt"], width=0.7,
    )
    nx.draw_networkx_nodes(
        undirected, pos, ax=ax, node_size=sizes, node_color=colors,
        edgecolors="white", linewidths=1.0, alpha=0.9,
    )

    # Label only the largest nodes by degree (top 10)
    top = sorted(deg.items(), key=lambda kv: -kv[1])[:10]
    label_pos = {n: pos[n] for n, _ in top}
    labels = {n: name_lookup.get(n, n) for n, _ in top}
    nx.draw_networkx_labels(
        undirected, label_pos, labels=labels, ax=ax,
        font_size=8.5, font_color=PALETTE["primary"], font_weight="bold",
    )

    ax.set_title(title, loc="left", pad=28, fontsize=16,
                 fontweight="semibold", color=PALETTE["primary"])
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes,
            fontsize=10, color=PALETTE["neutral_lt"], ha="left", va="bottom")
    ax.set_axis_off()

    # Legend
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=palette[pid], label=str(pid))
               for pid in sorted(palette.keys())[:12]]
    ax.legend(handles=handles, loc="lower left", title=subtitle.split("·")[0].strip(),
              fontsize=8.5, title_fontsize=9, frameon=True,
              facecolor="white", edgecolor="#E5E5E5")

    return fig


def _silo_map_pyvis(
    graph: nx.DiGraph[Any],
    node_partition: dict[str, int],
    palette: dict[int, str],
    name_lookup: dict[str, str],
    dept_lookup: dict[str, str],
    *,
    title: str,
    subtitle: str,
    physics: bool,
) -> str:
    from pyvis.network import Network

    net = Network(
        height="780px", width="100%",
        directed=True, notebook=False, cdn_resources="remote",
        heading=title, bgcolor="white", font_color=PALETTE["neutral"],
    )
    if not physics:
        net.toggle_physics(False)

    deg = dict(graph.degree())
    for n in graph.nodes():
        label = name_lookup.get(n, n)
        title_attr = "\n".join(filter(None, [
            label,
            dept_lookup.get(n, ""),
            f"degree = {deg.get(n, 0)}",
            f"silo = {node_partition.get(n, '?')}",
        ]))
        net.add_node(
            n,
            label=label,
            title=title_attr,
            color=palette[node_partition.get(n, 0)],
            size=18 + 4 * deg.get(n, 0),
        )
    for u, v in graph.edges():
        net.add_edge(u, v, color="#BFBFBF", arrows="to")

    html = net.generate_html(notebook=False)
    # Wrap in a proper HTML doc with the subtitle
    wrapped = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ margin: 0; font-family: 'DejaVu Sans', sans-serif; background: {PALETTE["background"]}; color: {PALETTE["neutral"]}; }}
  #header {{ padding: 16px 24px; background: white; border-bottom: 1px solid #E5E5E5; }}
  #header h1 {{ margin: 0; font-size: 18px; font-weight: 600; color: {PALETTE["primary"]}; }}
  #header .sub {{ font-size: 12px; color: {PALETTE["neutral_lt"]}; margin-top: 2px; }}
</style>
</head><body>
<div id="header">
  <h1>{title}</h1>
  <div class="sub">{subtitle}</div>
</div>
{html}
</body></html>"""
    return wrapped


__all__ = ["centrality_dashboard", "silo_map"]
