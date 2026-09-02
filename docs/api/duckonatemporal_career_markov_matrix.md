# `DuckONATemporal.career_markov_matrix`

**Module:** `pyduck_ona.temporal`

## Signature

```python
DuckONATemporal.career_markov_matrix(self, state_col'str'='job_level', lookback'str'='8Q', by'str | None'='department')
```

## Description

Estimate career-transition Markov probabilities from snapshot history

## Parameters

----------
state_col : str, default "job_level"
    Employee state used for transitions (e.g., job_level, role_band).
lookback : str, default "8Q"
    Number of periods to include.
by : str, optional
    Segment column for separate transition matrices (e.g., department).
    Set to ``None`` for one global matrix.

## Returns

-------
pandas.DataFrame
    Columns: ``segment, from_state, to_state, transitions, probability``.
    If ``by is None``, ``segment`` is ``"all"``.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
