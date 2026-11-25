"""Holiday-week shift of forecasts (optional; config 'holiday_shift', off when absent).

Reported values are higher than the weeks around them in the week that contains Christmas and,
for admissions, in the week that contains New Year's Day. The networks have no calendar input,
so a forecast whose target week is one of these weeks is shifted up by a fixed amount on the
model's z scale (z = log1p(value * scale)); the spread around the median is unchanged and the
other horizons are left as they are.

    config['holiday_shift'] = {target: {'christmas': shift, 'newyear': shift}}

A target week (week ending Saturday t) is 'christmas' when it contains December 25 and
'newyear' when it contains January 1.
"""
import numpy as np
import pandas as pd

TAGS = ('christmas', 'newyear')


def week_tag(week_end):
    """'christmas', 'newyear' or None for the week ending week_end (a Saturday)."""
    t = pd.Timestamp(week_end)
    if t.month == 12 and t.day >= 25:
        return 'christmas'
    if t.month == 1 and t.day <= 7:
        return 'newyear'
    return None


def shifts(config, target, last_date, horizons):
    """z shift per horizon (array of length horizons) for forecasts from the week ending last_date
    (horizon h targets last_date + h + 1 weeks); None when nothing is shifted."""
    spec = (config.get('holiday_shift') or {}).get(target)
    if not spec:
        return None
    unknown = set(spec) - set(TAGS)
    if unknown:
        raise ValueError(f'holiday_shift {target}: unknown weeks {sorted(unknown)}')
    out = np.zeros(horizons)
    for h in range(horizons):
        tag = week_tag(pd.Timestamp(last_date) + pd.Timedelta(weeks=h + 1))
        if tag:
            out[h] = float(spec.get(tag, 0.0))
    return out if out.any() else None


def apply(deltas, shift):
    """z-change quantiles or draws (N, H, ...) + shift per horizon; deltas itself when shift is None.
    The shift takes the dtype of deltas (network outputs are float32), so unshifted horizons stay
    bitwise unchanged."""
    if shift is None:
        return deltas
    deltas = np.asarray(deltas)
    return deltas + np.asarray(shift, dtype=deltas.dtype).reshape((1, -1) + (1,) * (deltas.ndim - 2))


def describe(shift):
    """Short note for logs, e.g. 'holiday shift h2 +0.24, h3 +0.24'; '' when None."""
    if shift is None:
        return ''
    return 'holiday shift ' + ', '.join(f'h{h} {s:+.3g}' for h, s in enumerate(shift) if s)
