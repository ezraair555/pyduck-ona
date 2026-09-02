#!/usr/bin/env python3
"""Generate per-function/class API pages in docs/api/ from live docstrings.

Usage:
    python scripts/generate_api_docs.py

Reads the public API exported by pyduck_ona, writes one markdown file per
function/class/method, and prints a README-ready catalog table.
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pyduck_ona as pona

API_DIR = Path(__file__).resolve().parent.parent / "docs" / "api"
API_DIR.mkdir(parents=True, exist_ok=True)


SECTION_ORDER = [
    ("DuckONAFrame", "v0.3 frame API"),
    ("DuckONA", "Analysis class"),
    ("DuckONATemporal", "Temporal class"),
    ("hierarchy", "Hierarchy primitives"),
    ("graph", "Graph metrics"),
    ("stats", "Statistical modeling"),
    ("bridge", "Graph export"),
    ("utility", "Utilities"),
]

CATEGORY_MAP: dict[str, str] = {
    "DuckONAFrame": "DuckONAFrame",
    "DuckONA": "DuckONA",
    "DuckONATemporal": "DuckONATemporal",
    "hierarchy_valid": "hierarchy",
    "hierarchy_long": "hierarchy",
    "hierarchy_wide": "hierarchy",
    "hierarchy_stats": "hierarchy",
    "betweenness": "graph",
    "pagerank": "graph",
    "connected_components": "graph",
    "shortest_path": "graph",
    "degree_centrality": "graph",
    "eigenvector_centrality": "graph",
    "louvain_communities": "graph",
    "correlation": "stats",
    "anova": "stats",
    "ols": "stats",
    "logistic": "stats",
    "chi_square": "stats",
    "plot_ols": "stats",
    "plot_residuals": "stats",
    "plot_coefficients": "stats",
    "vif": "stats",
    "model_compare_stats": "stats",
    "tidy_to_duckdb": "stats",
    "to_duckdb": "stats",
    "save_figure": "stats",
    "to_networkx": "bridge",
    "to_igraph": "bridge",
}


def _slug(name: str) -> str:
    return name.replace(".", "_").lower()


def _first_line(doc: str | None) -> str:
    if not doc:
        return ""
    first = doc.strip().splitlines()[0]
    return first.rstrip(".")


def _sections(doc: str | None) -> dict[str, str]:
    if not doc:
        return {}
    out: dict[str, str] = {}
    current = None
    lines: list[str] = []
    for line in doc.splitlines():
        stripped = line.strip()
        if re.match(r"^(Parameters|Returns|Raises|Notes|Examples|See Also|References)$", stripped):
            if current:
                out[current] = "\n".join(lines).strip()
            current = stripped
            lines = []
        else:
            lines.append(line)
    if current:
        out[current] = "\n".join(lines).strip()
    return out


def _signature_html(sig: inspect.Signature) -> str:
    # A compact Markdown code representation of the signature.
    params = []
    for name, param in sig.parameters.items():
        annotation = ""
        if param.annotation is not inspect.Parameter.empty:
            try:
                annotation = inspect.formatannotation(param.annotation)
            except Exception:
                annotation = str(param.annotation)
        default = ""
        if param.default is not inspect.Parameter.empty:
            default = f"={param.default!r}"
        params.append(f"{name}{annotation}{default}")
    return "(" + ", ".join(params) + ")"


def _render_doc_page(name: str, obj, sig: inspect.Signature | None, module: str) -> str:
    title = name
    doc = inspect.getdoc(obj) or ""
    first = _first_line(doc)
    sections = _sections(doc)

    lines = [
        f"# `{title}`",
        "",
        f"**Module:** `{module}`",
        "",
    ]
    if sig:
        lines.append("## Signature")
        lines.append("")
        lines.append(f"```python\n{title}{_signature_html(sig)}\n```")
        lines.append("")

    lines.append("## Description")
    lines.append("")
    lines.append(first if first else "No description available.")
    lines.append("")

    if "Parameters" in sections:
        lines.append("## Parameters")
        lines.append("")
        lines.append(_dedent_block(sections["Parameters"]))
        lines.append("")

    if "Returns" in sections:
        lines.append("## Returns")
        lines.append("")
        lines.append(_dedent_block(sections["Returns"]))
        lines.append("")

    if "Raises" in sections:
        lines.append("## Raises")
        lines.append("")
        lines.append(_dedent_block(sections["Raises"]))
        lines.append("")

    if "Examples" in sections:
        lines.append("## Example")
        lines.append("")
        lines.append(_dedent_block(sections["Examples"]))
        lines.append("")
    else:
        lines.append("## Example")
        lines.append("")
        lines.append("```python")
        lines.append("import pyduck_ona as pona")
        lines.append("# TODO: add a runnable example")
        lines.append("```")
        lines.append("")

    if "Notes" in sections:
        lines.append("## Notes")
        lines.append("")
        lines.append(_dedent_block(sections["Notes"]))
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("[Back to API catalog](../README.md#api-catalog)")
    lines.append("")
    return "\n".join(lines)


def _dedent_block(text: str) -> str:
    lines = text.splitlines()
    # Find the minimum indentation of non-empty lines.
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    if not indents:
        return text
    min_indent = min(indents)
    return "\n".join(line[min_indent:] if line.strip() else line for line in lines)


def main() -> int:
    entries: list[tuple[str, object, str, str]] = []  # (name, obj, module, category)

    public = [n for n in pona.__all__ if n != "__version__"]
    for public_name in public:
        obj = getattr(pona, public_name)
        category = CATEGORY_MAP.get(public_name, "utility")
        module = getattr(obj, "__module__", "pyduck_ona")
        entries.append((public_name, obj, module, category))

        # Also emit class methods
        if inspect.isclass(obj):
            for method_name, method in inspect.getmembers(obj, predicate=inspect.isfunction):
                if method_name.startswith("_"):
                    continue
                full = f"{public_name}.{method_name}"
                entries.append((full, method, getattr(method, "__module__", module), category))

    # Write pages
    catalog: dict[str, list[tuple[str, str, str]]] = {}
    for name, obj, module, category in entries:
        try:
            sig = inspect.signature(obj)
        except Exception:
            sig = None
        page = _render_doc_page(name, obj, sig, module)
        slug = _slug(name)
        path = API_DIR / f"{slug}.md"
        path.write_text(page)
        catalog.setdefault(category, []).append((name, _first_line(inspect.getdoc(obj)), slug))

    # Print README-ready grouped table
    print("\n<!-- API catalog generated by scripts/generate_api_docs.py -->")
    for group, label in SECTION_ORDER:
        if group not in catalog:
            continue
        print(f"\n### {label}")
        print("")
        print("| Function / Class | Description |")
        print("|---|---|")
        for name, desc, slug in sorted(catalog[group], key=lambda x: x[0]):
            short = desc[:90] + "..." if len(desc) > 90 else desc
            print(f"| [`{name}`](docs/api/{slug}.md) | {short} |")

    return 0


if __name__ == "__main__":
    sys.exit(main())
