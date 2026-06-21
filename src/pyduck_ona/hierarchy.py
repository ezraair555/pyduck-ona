"""
hierarchy: thin alias layer exposing core hierarchy functions under
a pyduck-janitor-style API. Most users will import from pyduck_ona
directly, but this sub-package lets you do:

    from pyduck_ona.hierarchy import valid, long, wide, stats

if you prefer short names. The behavior is identical to pyduck_ona.core.
"""
from __future__ import annotations

from pyduck_ona.core import (
    hierarchy_long,
    hierarchy_stats,
    hierarchy_valid,
    hierarchy_wide,
)

# Short-form aliases (the API an ONA analyst will type 100x a day)
valid = hierarchy_valid
long = hierarchy_long
wide = hierarchy_wide
stats = hierarchy_stats

__all__ = ["valid", "long", "wide", "stats"]