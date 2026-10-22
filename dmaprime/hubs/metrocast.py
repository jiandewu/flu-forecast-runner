"""Flu MetroCast Hub (reichlab/flu-metrocast): % of ED visits due to influenza for metro areas,
health service areas (HSAs) and their states.

Submission: one Parquet file per reference date with 100 sample trajectories per location
over horizons 0-3 (model-output/<team>-<model>/<reference_date>-<team>-<model>.parquet).
Columns: reference_date, target, location, target_end_date, output_type, output_type_id,
value (the 2026/27 round has no horizon task id). Each sample index belongs to one location,
so a sample is a trajectory across target_end_dates (compound_taskid_set = reference_date,
target, location).

Panel: the hub's target data for its locations, plus every other state, HSA and the nation
from CDC NSSP (dmaprime.data.nssp) as training series and model inputs. NSSP series that a
hub location already reports are not duplicated; the hub's NC regions and NYC come from other
surveillance systems, so their NSSP counterparts are kept as separate series.
"""
import json
import os

import numpy as np
import pandas as pd

from dmaprime.data import nssp
from dmaprime.data import sync as data_sync
from dmaprime.data.panel import Panel
from dmaprime.data.states import NAME_TO_ABBR

NAME = 'metrocast'
DATA_DIR = os.path.join('data', NAME)
REMOTE_BASE_URL = 'https://raw.githubusercontent.com/reichlab/flu-metrocast/main'
REMOTE_FILES = (
    'target-data/latest-data.csv',
    'auxiliary-data/locations.csv',
    'hub-config/tasks.json',
)
HISTORY_FILE = 'target-data/time-series.csv'   # data versions (as_of): the data as released on past dates
NSSP_FILE = 'nssp.parquet'
LEGACY_NSSP_FILE = 'nssp.csv'   # data directories and run records from before 2026-10-16
THRESHOLDS_FILE = 'nssp_thresholds.csv'

OUTPUT_TYPE = 'sample'
N_SAMPLES = 100
HORIZONS = 4
TARGET = 'Flu ED visits pct'
# Quantile levels the model is trained on; samples are drawn from these marginals
QUANTILES = np.array([0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.50,
                      0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99])
MEDIAN_IDX = 11
NATIONAL = 'nssp-us'

# file: hub data file in DATA_DIR; scale: value multiplier before log1p (values are percents);
# upper: maximum valid value
TARGETS = {
    TARGET: {'file': 'latest-data.csv', 'scale': 1.0, 'upper': 100.0},
}

# Hub locations listed in tasks.json but no longer in locations.csv: their state
EXTRA_LOCATION_STATES = {'provo': 'Utah', 'salt-lake-city': 'Utah', 'ogden': 'Utah'}


def sync(verbose=True):
    status = data_sync.ensure_current(REMOTE_BASE_URL, REMOTE_FILES, DATA_DIR, verbose)
    nssp.download(os.path.join(DATA_DIR, NSSP_FILE))
    nssp.download_thresholds(os.path.join(DATA_DIR, THRESHOLDS_FILE))
    if verbose:
        print(f'{NSSP_FILE}, {THRESHOLDS_FILE}: downloaded')
    return status


def sync_history(verbose=True):
    """Download the hub's versioned target data (time-series.csv), to rebuild the data released on a past date."""
    return data_sync.ensure_current(REMOTE_BASE_URL, [HISTORY_FILE], DATA_DIR, verbose)


def data_files():
    """Local file names this hub's forecasts read (snapshotted with every submission)."""
    return [os.path.basename(f) for f in REMOTE_FILES] + [NSSP_FILE, THRESHOLDS_FILE]


def nssp_path(data_dir=None):
    """The NSSP file in data_dir: Parquet, or the CSV of older data directories."""
    data_dir = data_dir or DATA_DIR
    path = os.path.join(data_dir, NSSP_FILE)
    return path if os.path.exists(path) else os.path.join(data_dir, LEGACY_NSSP_FILE)


def slug(name):
    return name.lower().replace(' ', '-')


def hub_locations(data_dir=None):
    """locations.csv as a frame indexed by location."""
    return pd.read_csv(os.path.join(data_dir or DATA_DIR, 'locations.csv'), dtype=str).set_index('location')


def round_locations(data_dir=None, round_idx=-1):
    """Locations of the target in a round of tasks.json (default: the latest round)."""
    with open(os.path.join(data_dir or DATA_DIR, 'tasks.json')) as f:
        tasks = json.load(f)
    for task in tasks['rounds'][round_idx]['model_tasks']:
        ids = task['task_ids']
        if TARGET in (ids['target']['required'] or []) + (ids['target']['optional'] or []):
            return (ids['location']['required'] or []) + (ids['location']['optional'] or [])
    raise ValueError(f'{TARGET} not in round {round_idx} of tasks.json')


def _hub_frame(data_dir, as_of=None):
    """date x location values of the hub target: latest data, or the version released on as_of.

    A version in time-series.csv holds only its own season, so earlier weeks come from the
    latest data (revisions of past seasons are small)."""
    latest = pd.read_csv(os.path.join(data_dir, 'latest-data.csv'), parse_dates=['target_end_date'])
    latest = latest[latest['target'] == TARGET]
    if as_of is not None:
        hist = pd.read_csv(os.path.join(data_dir, 'time-series.csv'), parse_dates=['as_of', 'target_end_date'])
        hist = hist[(hist['target'] == TARGET) & (hist['as_of'] == pd.Timestamp(as_of))]
        if hist.empty:
            raise ValueError(f'no {TARGET} version as of {as_of} in time-series.csv')
        # per location: the version's weeks, and the latest data before its first week there
        # (locations missing from the version had not been released and are left out)
        start = latest['location'].map(hist.groupby('location')['target_end_date'].min())
        latest = pd.concat([latest[latest['target_end_date'] < start], hist], ignore_index=True)
    return latest.pivot_table(index='target_end_date', columns='location', values='observation')


def _hub_nssp_keys(locs):
    """Hub location -> (NSSP geography, hsa_nci_id) for the hub locations whose data are NSSP."""
    keys = {}
    for loc, row in locs.iterrows():
        if row['location_type'] != 'hsa_nci_id' or loc == 'nyc':   # NYC: city health department data
            continue
        keys[loc] = (row['state'], row['original_location_code'])
    return keys


def load_panel(frames=None, data_dir=None, as_of=None, extra_locations=True):
    """Panel of the hub target for hub locations and (extra_locations) every NSSP state and HSA.

    as_of: a release date (Wednesday) to reconstruct the data available then: hub locations use
    that version; NSSP series are cut at the Saturday before (their revisions are not kept).
    Extra attributes: hub_locations, parent (location -> parent location), flags
    ({'low_coverage', 'anomalous'}: date x location bool arrays), nssp_through (last week of the
    national NSSP series: how far the NSSP release goes, which can lag the hub's data).
    """
    data_dir = data_dir or DATA_DIR
    locs = hub_locations(data_dir)
    hub = frames[TARGET] if frames is not None else _hub_frame(data_dir, as_of)
    hub_keys = _hub_nssp_keys(locs)

    df = nssp.with_flags(nssp.read(nssp_path(data_dir)), os.path.join(data_dir, THRESHOLDS_FILE))
    if as_of is not None:
        df = df[df['date'] <= pd.Timestamp(as_of) - pd.Timedelta(days=4)]
    key_to_hub = {v: k for k, v in hub_keys.items()}
    hsa_seen = set()
    names = []
    for geo, hsa in zip(df['geography'], df['hsa_nci_id']):
        if (geo, hsa) in key_to_hub:
            names.append(key_to_hub[(geo, hsa)])
        elif geo == nssp.NATIONAL_GEOGRAPHY:
            names.append(NATIONAL)
        elif hsa == 'All':
            names.append(f'nssp-{slug(geo)}')
        else:
            names.append(f'nssp-{hsa}')   # an HSA crossing state lines appears once per state
    df = df.assign(location=names).drop_duplicates(['date', 'location'])

    values = {}
    for loc in hub.columns:
        values[loc] = hub[loc]
    if extra_locations:
        extra = df[~df['location'].isin(hub.columns)]
    else:
        extra = df[df['location'] == NATIONAL]
    for loc, g in extra.groupby('location'):
        values[loc] = g.set_index('date')['value']
    frame = pd.DataFrame(values)
    panel = Panel(TARGETS, frames={TARGET: frame})

    flags = {}
    for name in ('low_coverage', 'anomalous'):
        f = df.assign(flag=df[name].astype(int)).pivot_table(index='date', columns='location', values='flag', aggfunc='max')
        flags[name] = f.reindex(index=panel.dates, columns=panel.locations).fillna(0).to_numpy() > 0
    panel.flags = flags
    panel.hub_locations = [l for l in panel.locations if l in set(hub.columns)]
    panel.parent = _parents(panel.locations, locs, df)
    national = df[(df['location'] == NATIONAL) & df['value'].notna()]
    panel.nssp_through = national['date'].max() if len(national) else None
    return panel


def _parents(locations, locs, df):
    """Parent series of every location: HSA -> its state, state -> nation, nation -> itself."""
    hsa_state = dict(zip('nssp-' + df['hsa_nci_id'], df['geography']))
    parent = {}
    for loc in locations:
        if loc == NATIONAL:
            parent[loc] = NATIONAL
        elif loc in locs.index or loc in EXTRA_LOCATION_STATES:
            if loc in locs.index:
                row = locs.loc[loc]
                is_state = row['original_location_code'] == 'All'
                state = row['state']
            else:
                is_state, state = False, EXTRA_LOCATION_STATES[loc]
            parent[loc] = NATIONAL if is_state else slug(state)
        elif loc.startswith('nssp-') and loc in hsa_state:
            state = hsa_state[loc]
            parent[loc] = slug(state) if slug(state) in locations and _nssp_state(slug(state), locs) \
                else f'nssp-{slug(state)}'
        else:   # an NSSP state
            parent[loc] = NATIONAL
        if parent[loc] not in locations:
            parent[loc] = NATIONAL
    return parent


def _nssp_state(loc, locs):
    """True when a hub state location reports NSSP data (not NC, which uses NC DETECT)."""
    return loc in locs.index and locs.loc[loc, 'location_type'] == 'hsa_nci_id'


def submission_frame(fc, locations):
    """One Forecast (sample output) as submission rows for the given locations."""
    if fc.output_type != OUTPUT_TYPE:
        raise ValueError(f'{NAME} needs {OUTPUT_TYPE} forecasts, got {fc.output_type}')
    idx = [fc.locations.index(l) for l in locations]
    values = np.clip(fc.values[idx], 0, TARGETS[fc.target]['upper'])   # (L, H, S)
    n_loc, n_h, n_s = values.shape
    ref = fc.last_date + pd.Timedelta(weeks=1)
    horizon = np.tile(np.repeat(np.arange(n_h), n_s), n_loc)
    # Sample ids are unique per location: a sample is one location's trajectory
    sample_id = (np.repeat(np.arange(n_loc), n_h * n_s) * n_s + np.tile(np.arange(n_s), n_loc * n_h))
    return pd.DataFrame({
        'reference_date': ref.date(),
        'target': fc.target,
        'location': np.repeat(np.asarray(locations), n_h * n_s),
        'target_end_date': (ref + pd.to_timedelta(horizon * 7, unit='D')).date,
        'output_type': OUTPUT_TYPE,
        'output_type_id': sample_id.astype(str),
        'value': np.round(values.reshape(-1), 4),
    })


def write_submission(forecasts, model_id, out_dir, data_dir=None):
    """forecasts: list of dmaprime.models.base.Forecast (sample output). Returns the file path.

    Only the locations of the latest round in tasks.json are written; locations without a
    forecast (no recent data) are reported."""
    wanted = round_locations(data_dir)
    frames, ref_dates = [], set()
    for fc in forecasts:
        have = [l for l in wanted if l in fc.locations]
        missing = sorted(set(wanted) - set(have))
        if missing:
            print(f'WARNING {fc.target}: no forecast for {missing}')
        frames.append(submission_frame(fc, have))
        ref_dates.add(fc.last_date + pd.Timedelta(weeks=1))
    ref = max(ref_dates)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{ref:%Y-%m-%d}-{model_id}.parquet')
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
    print(f'Submission saved to {path}')
    return path
