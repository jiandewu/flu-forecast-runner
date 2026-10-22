"""CDC NSSP emergency department visits for influenza, every state and health service area (HSA).

Source: "NSSP Emergency Department Visit Trajectories by State and Sub State Regions"
(data.cdc.gov dataset rdmq-nq56), released on Wednesdays with data through the previous
Saturday. Rows are counties; every county of an HSA carries the HSA's value, so the file is
reduced to one row per (week, state, HSA). State rows have hsa_nci_id 'All' and the national
row has geography 'United States'.

Reliability flags follow the ObsAwareData rules (github.com/DMA-PRIME/ObsAwareData, METHODS.md
section 3.3), so they can be computed on the Wednesday release, before ObsAwareData's own
weekly refresh:
    low_coverage  CDC's trend label is 'Data Unavailable' or 'Limited Data'
    anomalous     the smoothed percent changed by at least the state's threshold relative to
                  the closest preceding non-missing week (comparisons against 0 are skipped)
"""
import io
import os
from urllib.request import urlopen

import numpy as np
import pandas as pd

from dmaprime.data.states import ABBR_TO_NAME

DATASET_URL = 'https://data.cdc.gov/resource/rdmq-nq56.csv'
COLUMNS = {
    'week_end': 'date',
    'geography': 'geography',
    'hsa_nci_id': 'hsa_nci_id',
    'percent_visits_influenza': 'value',
    'percent_visits_smoothed_1': 'smoothed',   # smoothed influenza percent
    'ed_trends_influenza': 'trend',
}
THRESHOLDS_URL = ('https://raw.githubusercontent.com/DMA-PRIME/ObsAwareData/main/'
                  'metadata/observation_state_thresholds.csv')
LOW_COVERAGE_TRENDS = ('Data Unavailable', 'Limited Data')
NATIONAL_GEOGRAPHY = 'United States'


def download(path, timeout=300):
    """Fetch the influenza columns of every week and region; write one row per (date, geography, HSA)."""
    query = ('?$limit=5000000&$select=' + ','.join(COLUMNS)
             + '&$order=week_end,geography,hsa_nci_id')
    with urlopen(DATASET_URL + query, timeout=timeout) as response:
        df = pd.read_csv(io.BytesIO(response.read()), dtype=str, keep_default_na=False)
    df = df.rename(columns=COLUMNS)
    df['date'] = df['date'].str[:10]
    df = df.drop_duplicates()
    dup = df.duplicated(['date', 'geography', 'hsa_nci_id'], keep=False)
    if dup.any():
        raise RuntimeError(f'NSSP: {int(dup.sum())} rows disagree within an HSA')
    df = df.sort_values(['date', 'geography', 'hsa_nci_id'])
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    write(df, path)
    return df


def write(df, path):
    """Parquet (zstd; about 1-2 MB instead of 9 MB of CSV, which every run record snapshots) or CSV,
    by extension; all columns stay text as downloaded."""
    if path.endswith('.parquet'):
        df.to_parquet(path, index=False, compression='zstd')
    else:
        df.to_csv(path, index=False)


def download_thresholds(path, timeout=60):
    with urlopen(THRESHOLDS_URL, timeout=timeout) as response:
        content = response.read()
    with open(path, 'wb') as f:
        f.write(content)


def read(path):
    """A file written by download (.parquet, or .csv from before 2026-10-16)."""
    if path.endswith('.parquet'):
        df = pd.read_parquet(path).astype(str)
    else:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df['date'] = pd.to_datetime(df['date'])
    for col in ('value', 'smoothed'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def anomalous(smoothed, threshold_pct):
    """Flag weeks whose smoothed value moved by >= threshold_pct % from the closest preceding
    non-missing week. smoothed: 1-D array in date order (NaN = missing)."""
    flags = np.zeros(len(smoothed), dtype=bool)
    prev = np.nan
    for i, v in enumerate(smoothed):
        if np.isnan(v):
            continue
        if np.isfinite(prev) and prev != 0:
            flags[i] = abs(v - prev) / prev * 100 >= threshold_pct
        prev = v
    return flags


def with_flags(df, thresholds_path):
    """Add boolean columns low_coverage and anomalous (ObsAwareData rules)."""
    thr = pd.read_csv(thresholds_path)
    state_thr = dict(zip(thr['state'].map(ABBR_TO_NAME), thr['nssp_smoothed_change_pct'].astype(float)))
    df = df.sort_values(['geography', 'hsa_nci_id', 'date']).copy()
    df['low_coverage'] = df['trend'].isin(LOW_COVERAGE_TRENDS)
    df['anomalous'] = False
    for (geo, _), idx in df.groupby(['geography', 'hsa_nci_id']).groups.items():
        if geo not in state_thr:   # national row: no threshold
            continue
        df.loc[idx, 'anomalous'] = anomalous(df.loc[idx, 'smoothed'].to_numpy(dtype=float), state_thr[geo])
    return df.sort_values(['date', 'geography', 'hsa_nci_id'])

