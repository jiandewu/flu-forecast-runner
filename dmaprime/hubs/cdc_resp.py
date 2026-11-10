"""CDC COVID-19 and RSV Forecast Hubs: shared target data handling and submission format.

Both hubs (CDCgov/covid19-forecast-hub, CDCgov/rsv-forecast-hub) publish their target data as one
hubverse time series, target-data/time-series.parquet (target_end_date, location, observation,
as_of, target), holding every weekly release. sync() downloads it with auxiliary-data/locations.csv
and writes the newest release of each target as a CSV (date, location, value) in DATA_DIR; those
CSVs and locations.csv are the files a run reads and snapshots.

A hub module (dmaprime/hubs/covid.py, rsv.py) defines NAME, DATA_DIR, REMOTE_BASE_URL and TARGETS
and calls these helpers with itself as `hub`. Target specs are as in dmaprime/hubs/flusight.py,
plus an optional `exclude`: (first, last) target_end_date ranges, inclusive, left out of the panel
(None for an open end). NHSN hospital reporting was voluntary from 2024-05-01 to 2024-10-31, so
admissions in those weeks cover only part of the hospitals.

Submission: one CSV per reference date with both targets, the 23 quantiles for horizons 0-3
(model-output/<team>-<model>/<reference_date>-<team>-<model>.csv). The hubs accept any value >= 0;
ED proportions are kept <= 1.
"""
import os

import numpy as np
import pandas as pd

from dmaprime.data import sync as data_sync
from dmaprime.data.panel import Panel

TIME_SERIES = 'time-series.parquet'
REMOTE_FILES = ('target-data/' + TIME_SERIES, 'auxiliary-data/locations.csv')
OUTPUT_TYPE = 'quantile'
QUANTILES = np.array([0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.50,
                      0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99])
MEDIAN_IDX = 11
HORIZONS = 4
NATIONAL = 'US'


def read_time_series(hub, data_dir=None):
    """The hub's time series for our targets: target, date, location, value, as_of."""
    df = pd.read_parquet(os.path.join(data_dir or hub.DATA_DIR, TIME_SERIES))
    df = df[df['target'].isin(hub.TARGETS)]
    return pd.DataFrame({'target': df['target'].astype(str),
                         'date': pd.to_datetime(df['target_end_date']),
                         'location': df['location'].astype(str),
                         'value': df['observation'].astype(float),
                         'as_of': pd.to_datetime(df['as_of'])})


def write_target_files(hub, data_dir=None):
    """Write each target's newest release as data_dir/<spec file> (date, location, value)."""
    data_dir = data_dir or hub.DATA_DIR
    df = read_time_series(hub, data_dir)
    for target, spec in hub.TARGETS.items():
        rows = df[df['target'] == target]
        if rows.empty:
            raise RuntimeError(f'{hub.NAME}: no {target} rows in {TIME_SERIES}')
        newest = rows[rows['as_of'] == rows['as_of'].max()].dropna(subset=['value'])
        newest = newest.sort_values(['date', 'location'])[['date', 'location', 'value']]
        newest.to_csv(os.path.join(data_dir, spec['file']), index=False, date_format='%Y-%m-%d')


def sync(hub, verbose=True):
    status = data_sync.ensure_current(hub.REMOTE_BASE_URL, REMOTE_FILES, hub.DATA_DIR, verbose)
    write_target_files(hub)
    return status


def data_files(hub):
    """Local file names a forecast reads (snapshotted with every submission)."""
    return [spec['file'] for spec in hub.TARGETS.values()] + ['locations.csv']


def exclude(frame, ranges):
    """frame (date x location) with the rows of every (first, last) range removed."""
    keep = np.ones(len(frame), dtype=bool)
    for first, last in ranges or ():
        lo = frame.index >= pd.Timestamp(first) if first else True
        hi = frame.index <= pd.Timestamp(last) if last else True
        keep &= ~(lo & hi)
    return frame[keep]


def load_panel(hub, frames=None, data_dir=None):
    """Panel of the hub's targets from the target CSVs, or from {target: date x location frame};
    the `exclude` ranges are removed either way."""
    if frames is None:
        frames = {}
        for target, spec in hub.TARGETS.items():
            df = pd.read_csv(os.path.join(data_dir or hub.DATA_DIR, spec['file']),
                             parse_dates=['date'], dtype={'location': str})
            frames[target] = df.pivot_table(index='date', columns='location', values='value')
    frames = {t: exclude(f, hub.TARGETS[t].get('exclude')) for t, f in frames.items()}
    return Panel(hub.TARGETS, frames=frames)


def releases(hub, data_dir=None):
    """Every weekly release: {target: {as_of: date x location frame}}."""
    df = read_time_series(hub, data_dir)
    return {t: {as_of: g.pivot_table(index='date', columns='location', values='value')
                for as_of, g in gt.groupby('as_of')}
            for t, gt in df.groupby('target')}


def populations(hub, data_dir=None):
    path = os.path.join(data_dir or hub.DATA_DIR, 'locations.csv')
    return pd.read_csv(path, dtype={'location': str}).set_index('location')['population']


def submission_frame(hub, fc):
    """One Forecast (quantile output) as submission rows: one row per location, horizon and quantile."""
    if fc.output_type != OUTPUT_TYPE:
        raise ValueError(f'{hub.NAME} needs {OUTPUT_TYPE} forecasts, got {fc.output_type}')
    values = np.clip(fc.values, 0, hub.TARGETS[fc.target]['upper'])
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


def write_submission(hub, forecasts, model_id, out_dir):
    """forecasts: list of dmaprime.models.base.Forecast (quantile output). Returns the file path."""
    ref_dates = sorted({fc.last_date + pd.Timedelta(weeks=1) for fc in forecasts})
    if len(ref_dates) > 1:
        print(f'WARNING: targets have different latest data weeks; reference dates {ref_dates}')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{ref_dates[-1]:%Y-%m-%d}-{model_id}.csv')
    pd.concat([submission_frame(hub, fc) for fc in forecasts], ignore_index=True).to_csv(path, index=False)
    print(f'Submission saved to {path}')
    return path
