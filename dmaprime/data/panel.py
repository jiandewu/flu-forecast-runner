"""Weekly date x location panel of target values, and the z = log1p(value * scale) transform."""
import os

import numpy as np
import pandas as pd


def to_z(values, spec):
    """spec: a hub target spec with 'scale'."""
    return np.log1p(np.asarray(values, dtype=float) * spec['scale'])


def from_z(z, spec):
    """Inverse of to_z, clipped to [0, spec['upper']] (no upper bound when it is None)."""
    values = np.expm1(z) / spec['scale']
    return np.clip(values, 0, spec['upper'])


class Panel:
    """Weekly date x location matrices of values and z for every target.

    targets: {target: spec} from a hub; each spec has 'file' (CSV with date, location, value
    columns in data_dir) and 'scale'.
    frames: optional {target: date x location DataFrame of values}; default reads the files.
    """

    def __init__(self, targets, data_dir=None, frames=None):
        if frames is None:
            frames = {}
            for target, spec in targets.items():
                df = pd.read_csv(os.path.join(data_dir, spec['file']),
                                 parse_dates=['date'], dtype={'location': str})
                frames[target] = df.pivot_table(index='date', columns='location', values='value')
        self.targets = targets
        self.dates = pd.DatetimeIndex(sorted(set().union(*(f.index for f in frames.values()))))
        self.locations = sorted(set().union(*(f.columns for f in frames.values())))
        self.values = {t: f.reindex(index=self.dates, columns=self.locations) for t, f in frames.items()}
        self.z = {t: to_z(v.to_numpy(), targets[t]) for t, v in self.values.items()}
        week = self.dates.isocalendar().week.to_numpy(dtype=float)
        self.season = np.stack([np.sin(2 * np.pi * week / 52.18), np.cos(2 * np.pi * week / 52.18)], axis=1)

    def date_index(self, date):
        return int(self.dates.get_loc(pd.Timestamp(date)))

    def latest_origin(self, target):
        """Index of the latest week with any data for target."""
        origin = len(self.dates) - 1
        while not np.isfinite(self.z[target][origin]).any():
            origin -= 1
        return origin
