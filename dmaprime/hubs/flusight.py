"""CDC FluSight Forecast Hub: targets, data sources, plausibility checks and submission format.

Submission: one CSV per reference date with every location and both targets, 23 quantiles
for horizons 0-3 (model-output/<team>-<model>/<reference_date>-<team>-<model>.csv).
"""
import os

import numpy as np
import pandas as pd

from dmaprime.data import sync as data_sync
from dmaprime.data.panel import Panel

NAME = 'flusight'
DATA_DIR = os.path.join('data', NAME)
REMOTE_BASE_URL = 'https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/main'
REMOTE_FILES = (
    'target-data/target-ed-visits-prop.csv',
    'target-data/target-hospital-admissions.csv',
    'auxiliary-data/locations.csv',  # populations for plausibility checks
)

OUTPUT_TYPE = 'quantile'
QUANTILES = np.array([0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.50,
                      0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99])
MEDIAN_IDX = 11
HORIZONS = 4
NATIONAL = 'US'

# file: data file in DATA_DIR; aux: co-signal target; scale: value multiplier before log1p;
# upper: maximum valid value
TARGETS = {
    'wk inc flu hosp': {
        'file': 'target-hospital-admissions.csv',
        'aux': 'wk inc flu prop ed visits',
        'scale': 1.0,     # counts
        'upper': None,
    },
    'wk inc flu prop ed visits': {
        'file': 'target-ed-visits-prop.csv',
        'aux': 'wk inc flu hosp',
        'scale': 100.0,   # proportion -> percent before log1p
        'upper': 1.0,
    },
}

# Plausibility checks run by the hub on the upper bound of the forecast distribution
MAX_HOSP_SHARE_OF_POP = 0.30
MAX_ED_PROP = 0.25
# The hub requires integer forecasts of admission counts
INTEGER_TARGETS = ('wk inc flu hosp',)


def sync(verbose=True):
    return data_sync.ensure_current(REMOTE_BASE_URL, REMOTE_FILES, DATA_DIR, verbose)


def data_files():
    """Local file names this hub's forecasts read (snapshotted with every submission)."""
    return [os.path.basename(f) for f in REMOTE_FILES]


def load_panel(frames=None, data_dir=None):
    return Panel(TARGETS, data_dir or DATA_DIR, frames)


def populations(data_dir=None):
    path = os.path.join(data_dir or DATA_DIR, 'locations.csv')
    return pd.read_csv(path, dtype={'location': str}).set_index('location')['population']


def plausibility_caps(target, locations, values, pops):
    """Cap quantiles at the FluSight plausibility limits; report any location that was capped."""
    locations = pd.Series(list(locations))
    if target == 'wk inc flu hosp':
        # Floor so that a capped count stays an integer within the limit
        cap = np.floor(locations.map(pops).to_numpy(dtype=float).reshape(-1, 1, 1) * MAX_HOSP_SHARE_OF_POP)
    else:
        cap = MAX_ED_PROP
    capped = values > cap
    if capped.any():
        locs = sorted(set(locations[capped.any(axis=(1, 2))]))
        print(f'WARNING {target}: capped quantiles at plausibility limit for {locs}')
    return np.minimum(values, cap)


def submission_frame(fc, pops):
    """One Forecast (quantile output) as submission rows: admission counts rounded to integers,
    plausibility caps applied, one row per location, horizon and quantile."""
    if fc.output_type != OUTPUT_TYPE:
        raise ValueError(f'{NAME} needs {OUTPUT_TYPE} forecasts, got {fc.output_type}')
    values = fc.values
    if fc.target in INTEGER_TARGETS:
        values = np.round(values)  # rounding keeps the quantiles in order
    values = plausibility_caps(fc.target, fc.locations, values, pops)
    ref = fc.last_date + pd.Timedelta(weeks=1)
    n_ids = len(fc.output_type_ids)
    horizon = np.tile(np.repeat(np.arange(HORIZONS), n_ids), len(fc.locations))
    return pd.DataFrame({
        'reference_date': ref.strftime('%Y-%m-%d'),
        'target': fc.target,
        'horizon': horizon,
        'target_end_date': (ref + pd.to_timedelta(horizon * 7, unit='D')).strftime('%Y-%m-%d'),
        'location': np.repeat(np.asarray(fc.locations), HORIZONS * n_ids),
        'output_type': OUTPUT_TYPE,
        'output_type_id': np.tile(fc.output_type_ids, len(fc.locations) * HORIZONS),
        'value': values.reshape(-1),
    })


def write_submission(forecasts, model_id, out_dir, data_dir=None):
    """forecasts: list of dmaprime.models.base.Forecast (quantile output). Returns the file path."""
    pops = populations(data_dir)
    ref_dates = sorted({fc.last_date + pd.Timedelta(weeks=1) for fc in forecasts})
    if len(ref_dates) > 1:
        print(f'WARNING: targets have different latest data weeks; reference dates {ref_dates}')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{ref_dates[-1]:%Y-%m-%d}-{model_id}.csv')
    pd.concat([submission_frame(fc, pops) for fc in forecasts], ignore_index=True).to_csv(path, index=False)
    print(f'Submission saved to {path}')
    return path
