"""
Organizational chart visualizations.

Two functions:

- :func:`org_chart_tree`        - Interactive D3-style HTML org chart with
                                   collapsible nodes, zoom/pan, and optional
                                   colour coding by department or level.
- :func:`reporting_chain_walk`  - Clean horizontal flow diagram showing the
                                   chain of command from an employee up to
                                   the top of the organization.
"""
from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd

from .theme import (
    CATEGORICAL,
    PALETTE,
    apply_default_style,
    new_figure,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_hierarchy(
    df: pd.DataFrame,
    *,
    id_col: str = "employee_id",
    supervisor_col: str = "supervisor_id",
) -> pd.DataFrame:
    """Coerce the input to a DataFrame with at minimum id/supervisor columns."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame (use .df() on the relation)")
    if id_col not in df.columns:
        raise KeyError(f"Column '{id_col}' not found in DataFrame")
    if supervisor_col not in df.columns:
        raise KeyError(f"Column '{supervisor_col}' not found in DataFrame")
    return df


def _build_tree(
    df: pd.DataFrame, id_col: str, supervisor_col: str,
) -> dict[str, Any]:
    """Build a parent → children mapping from a long-form hierarchy table."""
    children: dict[str, list[str]] = {}
    roots: list[str] = []
    for _, row in df.iterrows():
        child = row[id_col]
        parent = row[supervisor_col]
        if pd.isna(parent) or parent is None or parent == "":
            roots.append(child)
            children.setdefault(child, [])
        else:
            children.setdefault(parent, []).append(child)
            children.setdefault(child, [])
    if not roots:
        # No roots declared: synthesize one synthetic root to keep the chart
        # connected. Pick the first row's parent or any node.
        all_ids = set(df[id_col].dropna().tolist())
        roots = [next(iter(all_ids))]
    return {"children_map": children, "roots": roots}


# ---------------------------------------------------------------------------
# Interactive org chart (HTML / D3-style)
# ---------------------------------------------------------------------------

def org_chart_tree(
    df: pd.DataFrame,
    *,
    id_col: str = "employee_id",
    supervisor_col: str = "supervisor_id",
    metadata: pd.DataFrame | None = None,
    name_col: str = "name",
    title_col: str = "title",
    department_col: str = "department",
    level_col: str = "level",
    color_by: str | None = "department",
    root_id: str | None = None,
    title: str | None = "Organizational Chart",
    height: str = "820px",
    width: str = "100%",
) -> str:
    """Return a standalone HTML string containing an interactive org chart.

    Parameters
    ----------
    df
        Long-form hierarchy (one row per manager → report edge). The output
        of ``pyduck_ona.hierarchy_long(...)`` after ``.df()`` works directly.
    metadata
        Optional per-employee metadata DataFrame keyed by ``id_col``.
    color_by
        One of ``"department"``, ``"level"`` or ``None``. Controls node fill.
    root_id
        Optional explicit root node; defaults to the topmost supervisor.

    Returns
    -------
    str
        A full HTML document with an embedded D3 tree layout. Save to disk
        with ``Path("org.html").write_text(html)`` to view in a browser.
    """
    df = _normalize_hierarchy(df, id_col=id_col, supervisor_col=supervisor_col)

    tree = _build_tree(df, id_col=id_col, supervisor_col=supervisor_col)
    children_map: dict[str, list[str]] = tree["children_map"]
    roots: list[str] = tree["roots"]

    # Lookup metadata per id
    meta_lookup: dict[str, dict[Any, Any]] = {}
    if metadata is not None and id_col in metadata.columns:
        for _, row in metadata.iterrows():
            meta_lookup[str(row[id_col])] = row.to_dict()

    # Resolve categorical colors for color_by
    color_lookup: dict[str, str] = {}
    if color_by and metadata is not None and color_by in metadata.columns:
        distinct = [str(v) for v in metadata[color_by].dropna().unique()]
        for i, v in enumerate(sorted(distinct)):
            color_lookup[v] = CATEGORICAL[i % len(CATEGORICAL)]

    # Build the nested tree structure that D3 expects.
    def make_node(node_id: str) -> dict[str, Any]:
        m = meta_lookup.get(str(node_id), {})
        name = str(m.get(name_col, node_id)) if name_col in m else str(node_id)
        title_str = str(m.get(title_col, "")) if title_col in m else ""
        dept_str = str(m.get(department_col, "")) if department_col in m else ""
        level_str = str(m.get(level_col, "")) if level_col in m else ""
        if color_by and color_by in m and not pd.isna(m[color_by]):
            fill = color_lookup.get(str(m[color_by]), PALETTE["primary"])
        else:
            fill = PALETTE["primary"]
        node: dict[str, Any] = {
            "id":       str(node_id),
            "name":     name,
            "title":    title_str,
            "department": dept_str,
            "level":    level_str,
            "fill":     fill,
            "children": [],
        }
        for child_id in children_map.get(node_id, []):
            node["children"].append(make_node(child_id))
        return node

    # Pick the rendering root
    chosen_root = root_id or roots[0]
    tree_obj = make_node(chosen_root)

    # Pre-compute the legend entries
    legend_entries: list[dict[str, str]] = []
    if color_by:
        for k in sorted(color_lookup.keys()):
            legend_entries.append({"label": k, "fill": color_lookup[k]})

    import json
    tree_json = json.dumps(tree_obj, ensure_ascii=False)
    legend_json = json.dumps(legend_entries, ensure_ascii=False)

    # The D3 v7 tree is loaded inline from a CDN; the layout is implemented
    # directly so the output works offline-friendly if the user downloads d3.
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{escape(title or "Organizational Chart")}</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  body {{
    margin: 0;
    font-family: 'DejaVu Sans', 'Helvetica Neue', Arial, sans-serif;
    background: {PALETTE["background"]};
    color: {PALETTE["neutral"]};
  }}
  #header {{
    padding: 16px 24px;
    background: white;
    border-bottom: 1px solid #E5E5E5;
  }}
  #header h1 {{
    margin: 0;
    font-size: 18px;
    font-weight: 600;
    color: {PALETTE["primary"]};
  }}
  #header .subtitle {{
    font-size: 12px;
    color: {PALETTE["neutral_lt"]};
    margin-top: 2px;
  }}
  #chart {{
    width: {width};
    height: {height};
    background: white;
    position: relative;
    overflow: hidden;
  }}
  .node circle {{
    stroke: white;
    stroke-width: 2;
    cursor: pointer;
    transition: r 120ms ease-out;
  }}
  .node:hover circle {{ stroke: {PALETTE["accent"]}; stroke-width: 3; }}
  .node text {{ font-size: 11px; fill: {PALETTE["neutral"]}; pointer-events: none; }}
  .node .name {{ font-weight: 600; }}
  .node .title-text {{ font-size: 9px; fill: {PALETTE["neutral_lt"]}; }}
  .link {{
    fill: none;
    stroke: #C8C8C8;
    stroke-width: 1.2;
    stroke-opacity: 0.85;
  }}
  #legend {{
    position: absolute;
    top: 16px;
    right: 16px;
    background: white;
    border: 1px solid #E5E5E5;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 11px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    z-index: 10;
  }}
  #legend h3 {{
    margin: 0 0 6px 0;
    font-size: 11px;
    color: {PALETTE["neutral_lt"]};
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  #legend .swatch {{
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
  }}
  #controls {{
    position: absolute;
    bottom: 16px;
    right: 16px;
    z-index: 10;
    display: flex;
    gap: 6px;
  }}
  #controls button {{
    background: white;
    border: 1px solid {PALETTE["neutral_lt"]};
    color: {PALETTE["primary"]};
    border-radius: 4px;
    padding: 4px 10px;
    cursor: pointer;
    font-size: 11px;
  }}
  #controls button:hover {{ background: {PALETTE["background"]}; }}
</style>
</head>
<body>
<div id="header">
  <h1>{escape(title or "Organizational Chart")}</h1>
  <div class="subtitle">Click a node to collapse / expand its subtree. Scroll to zoom, drag to pan.</div>
</div>
<div id="chart">
  <div id="legend">
    <h3>{escape(color_by or "")}</h3>
    <div id="legend-rows"></div>
  </div>
  <div id="controls">
    <button onclick="chart.zoomBy(1.2)">+</button>
    <button onclick="chart.zoomBy(0.83)">−</button>
    <button onclick="chart.reset()">Reset</button>
    <button onclick="chart.expandAll()">Expand all</button>
    <button onclick="chart.collapseAll()">Collapse all</button>
  </div>
</div>
<script>
  const treeData = {tree_json};
  const legendEntries = {legend_json};

  // Render legend
  const legendRows = document.getElementById("legend-rows");
  if (legendEntries.length === 0) {{
    document.getElementById("legend").style.display = "none";
  }} else {{
    legendEntries.forEach(entry => {{
      const div = document.createElement("div");
      div.style.margin = "2px 0";
      div.innerHTML = `<span class="swatch" style="background:${{entry.fill}}"></span>${{entry.label}}`;
      legendRows.appendChild(div);
    }});
  }}

  const container = document.getElementById("chart");
  const margin = {{ top: 24, right: 24, bottom: 24, left: 24 }};
  const width = container.clientWidth - margin.left - margin.right;
  const height = container.clientHeight - margin.top - margin.bottom;

  const svg = d3.select("#chart")
    .append("svg")
    .attr("width", container.clientWidth)
    .attr("height", container.clientHeight);

  const g = svg.append("g").attr("transform", `translate(${{margin.left}},${{margin.top}})`);

  const zoom = d3.zoom()
    .scaleExtent([0.2, 4])
    .on("zoom", (event) => g.attr("transform", event.transform));
  svg.call(zoom);

  const root = d3.hierarchy(treeData, d => d.children);
  root.x0 = height / 2;
  root.y0 = 0;

  // Initialize all nodes expanded
  if (root.children) root.children.forEach(collapseInit);

  function collapseInit(d) {{
    if (d.children) {{
      d._children = d.children;
      d.children = null;
      d.children && d.children.forEach(collapseInit);
    }}
  }}

  // Re-expand everything for full view
  function expandAll(d = root) {{
    if (d._children) {{
      d.children = d._children;
      d._children = null;
    }}
    if (d.children) d.children.forEach(expandAll);
  }}
  function collapseAll(d = root) {{
    if (d.children) {{
      d._children = d.children;
      d.children = null;
    }}
    if (d._children) d._children.forEach(collapseAll);
  }}

  const treeLayout = d3.tree().size([height, width]);
  const linkGen = d3.linkHorizontal().x(d => d.y).y(d => d.x);

  function update(source) {{
    treeLayout(root);
    const nodes = root.descendants().reverse();
    const links = root.links();

    // Normalize depth for x spacing
    nodes.forEach(d => {{ d.y = d.depth * 220; }});

    const node = g.selectAll("g.node")
      .data(nodes, d => d.id || (d.id = ++i));
    const nodeEnter = node.enter().append("g")
      .attr("class", "node")
      .attr("transform", () => `translate(${{source.y0 || source.y}},${{source.x0 || source.x}})`)
      .on("click", (event, d) => {{
        if (d.children) {{
          d._children = d.children; d.children = null;
        }} else if (d._children) {{
          d.children = d._children; d._children = null;
        }}
        update(d);
      }});

    nodeEnter.append("circle")
      .attr("r", 1e-6)
      .style("fill", d => d.data.fill);

    nodeEnter.append("text")
      .attr("class", "name")
      .attr("dy", "-0.25em")
      .attr("text-anchor", "middle")
      .text(d => truncate(d.data.name, 18));

    nodeEnter.append("text")
      .attr("class", "title-text")
      .attr("dy", "1.0em")
      .attr("text-anchor", "middle")
      .text(d => truncate(d.data.title || d.data.department || "", 22));

    const nodeUpdate = nodeEnter.merge(node);
    nodeUpdate.transition().duration(250)
      .attr("transform", d => `translate(${{d.y}},${{d.x}})`);
    nodeUpdate.select("circle")
      .transition().duration(250)
      .attr("r", d => d._children ? 8 : 6)
      .style("fill", d => d.data.fill);

    const nodeExit = node.exit().transition().duration(250)
      .attr("transform", d => `translate(${{source.y}},${{source.x}})`).remove();
    nodeExit.select("circle").attr("r", 1e-6);

    const link = g.selectAll("path.link")
      .data(links, d => d.target.id);
    link.enter().insert("path", "g")
      .attr("class", "link")
      .attr("d", d => {{
        const o = {{ x: source.x0 || source.x, y: source.y0 || source.y }};
        return linkGen({{ source: o, target: o }});
      }})
      .merge(link)
      .transition().duration(250)
      .attr("d", d => linkGen(d));
    link.exit().transition().duration(250)
      .attr("d", d => {{
        const o = {{ x: source.x, y: source.y }};
        return linkGen({{ source: o, target: o }});
      }}).remove();

    nodes.forEach(d => {{ d.x0 = d.x; d.y0 = d.y; }});
  }}

  function truncate(s, n) {{ return s.length > n ? s.slice(0, n - 1) + "…" : s; }}

  let i = 0;
  update(root);

  // Fit initial view
  const bbox = g.node().getBBox();
  const fullW = container.clientWidth;
  const fullH = container.clientHeight;
  const scale = Math.min(fullW / (bbox.width + 200), fullH / (bbox.height + 100), 1);
  const tx = (fullW - bbox.width * scale) / 2 - bbox.x * scale;
  const ty = (fullH - bbox.height * scale) / 2 - bbox.y * scale;
  svg.transition().duration(500).call(
    zoom.transform,
    d3.zoomIdentity.translate(tx, ty).scale(scale)
  );

  window.chart = {{
    zoomBy(k) {{ svg.transition().call(zoom.scaleBy, k); }},
    reset() {{ svg.transition().call(zoom.transform, d3.zoomIdentity); }},
    expandAll() {{ expandAll(); update(root); }},
    collapseAll() {{ collapseAll(); update(root); }}
  }};
</script>
</body>
</html>
"""
    return html


# ---------------------------------------------------------------------------
# Reporting chain walk (matplotlib)
# ---------------------------------------------------------------------------

def reporting_chain_walk(
    df: pd.DataFrame,
    employee_id: str,
    *,
    id_col: str = "employee_id",
    supervisor_col: str = "supervisor_id",
    metadata: pd.DataFrame | None = None,
    name_col: str = "name",
    title_col: str = "title",
    level_col: str = "level",
    title: str | None = None,
    figsize: tuple[float, float] = (12.0, 4.5),
) -> Any:
    """Plot the reporting chain from ``employee_id`` up to the top of the org.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = _normalize_hierarchy(df, id_col=id_col, supervisor_col=supervisor_col)

    parent_lookup: dict[str, str | None] = {}
    for _, row in df.iterrows():
        parent_lookup[str(row[id_col])] = (
            None if pd.isna(row[supervisor_col]) else str(row[supervisor_col])
        )

    # Walk up to the root
    chain: list[str] = []
    current: str | None = employee_id
    visited: set[str] = set()
    while current is not None and current not in visited:
        visited.add(current)
        chain.append(current)
        current = parent_lookup.get(current)
    chain.reverse()  # top of org first, employee last

    meta_lookup: dict[str, dict[Any, Any]] = {}
    if metadata is not None and id_col in metadata.columns:
        for _, row in metadata.iterrows():
            meta_lookup[str(row[id_col])] = row.to_dict()

    apply_default_style()
    fig, ax = new_figure(figsize=figsize, nrows=1, ncols=1)
    ax.set_xlim(-0.5, len(chain) - 0.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis("off")

    if title is None:
        title = f"Reporting chain · {employee_id}"

    fig.suptitle(title, fontsize=16, fontweight="semibold",
                 color=PALETTE["primary"], x=0.06, ha="left", y=0.97)

    for i, node_id in enumerate(chain):
        m = meta_lookup.get(node_id, {})
        name = str(m.get(name_col, node_id)) if name_col in m else str(node_id)
        title_str = str(m.get(title_col, "")) if title_col in m else ""
        level_str = str(m.get(level_col, "")) if level_col in m else ""

        # Node box
        is_target = node_id == employee_id
        fill = PALETTE["accent"] if is_target else PALETTE["primary"]
        box = dict(
            boxstyle="round,pad=0.6,rounding_size=0.25",
            facecolor="white",
            edgecolor=fill,
            linewidth=1.6,
        )
        ax.text(i, 0, name, ha="center", va="center",
                fontsize=11, fontweight="bold", color=PALETTE["primary"], bbox=box)

        if title_str or level_str:
            sub = "\n".join(filter(None, [title_str, level_str]))
            ax.text(i, -0.85, sub, ha="center", va="top",
                    fontsize=8.5, color=PALETTE["neutral_lt"], linespacing=1.4)

        # Connector arrow to next
        if i < len(chain) - 1:
            ax.annotate(
                "", xy=(i + 0.42, 0), xytext=(i + 0.30, 0),
                arrowprops=dict(arrowstyle="->", color=PALETTE["neutral_lt"],
                                lw=1.5, shrinkA=2, shrinkB=2),
            )

    # Footer caption
    fig.text(0.06, 0.04,
             f"{len(chain)} levels · from top of org to {employee_id}",
             fontsize=9, color=PALETTE["neutral_lt"], ha="left")

    return fig


__all__ = ["org_chart_tree", "reporting_chain_walk"]
