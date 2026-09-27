"""Write board JSON a browser can parse.

Python's json writes float NaN and Infinity as bare tokens a browser's JSON.parse
rejects, which blanks the whole board on the site (the MLB tab on 2026-09-27,
from a game with no probable pitcher announced). `clean` turns them into null;
`dumps` then refuses any that slip through, so the failure is loud at publish,
not silent on the page.
"""
import json
import math


def clean(value):
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    try:                                   # numpy scalars
        import numpy as np
        if isinstance(value, np.floating):
            return clean(float(value))
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
    except Exception:
        pass
    return value


def dumps(payload, **kw) -> str:
    return json.dumps(clean(payload), allow_nan=False, **kw)
