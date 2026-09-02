# `DuckONA.build_temporal_slices`

**Module:** `pyduck_ona.analysis`

## Signature

```python
DuckONA.build_temporal_slices(self, table_name'str', date_col'str', freq'str'='M')
```

## Description

Return time-sliced relations for a registered table

## Parameters

----------
table_name : str
    Registered table to slice.
date_col : str
    Date / datetime column used for slicing.
freq : {"D", "W", "M", "Q", "Y"}, default "M"
    Slice frequency.

## Returns

-------
list of (slice_label, start_date, end_date, relation)
    One tuple per slice covering the observed date range in
    ``table_name``.

## Example

--------
>>> slices = ona.build_temporal_slices("attendance", "date", freq="M")
>>> for label, start, end, rel in slices:
...     print(label, rel.count("*").fetchone()[0])

---

[Back to API catalog](../README.md#api-catalog)
