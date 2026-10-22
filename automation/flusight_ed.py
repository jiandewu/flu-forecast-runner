"""FluSight ED forecast from DMAPRIME-ObsHYB (used by automation/auto_submit.py).

The weekly MetroCast run (automation/metrocast_weekly.py combine) writes flusight_ed.csv to its run
record (branch runs-metrocast, or runs-metrocast-shadow for shadow runs): quantiles of the % ED visits for influenza (as proportions)
for the nation and every state's NSSP series, from the networks trained for MetroCast.

The FluSight weekly run replaces the ED rows of the HYB file with them, location by location
(hub.submission_frame, as for HYB's own rows). A location keeps HYB's forecast when the file has
no forecast for it from the same week, or its last observed value differs from FluSight's data.
The file used is copied into the FluSight run record (obshyb_ed.csv) and listed in run.json.

    python -m automation.flusight_ed <submission file> <obshyb_ed.csv> [<data dir>]   # apply a saved file
"""
import hashlib
import io
import json
import os
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from dmaprime.hubs import flusight as hub
from dmaprime.models.base import Forecast

ED_TARGET = 'wk inc flu prop ed visits'
ED_FILE = 'flusight_ed.csv'
RECORD_FILE = 'obshyb_ed.csv'
RUNS_BRANCH = 'runs-metrocast'          # shadow runs (--no-upload) read SHADOW_BRANCH
SHADOW_BRANCH = 'runs-metrocast-shadow'
WAIT_UNTIL_ET = 21      # a submission waits for the file until 21:00 ET, then goes ahead with HYB's ED
REL_TOL, ABS_TOL = 0.02, 1e-4   # last observed value: ObsHYB's input vs FluSight's data


def _git(*args):
    return subprocess.run(['git', *args], capture_output=True, text=True)


def fetch(reference_date, branch=RUNS_BRANCH):
    """(flusight_ed.csv of the MetroCast run for reference_date on origin/<branch> or None,
    the record branch's commit or None)."""
    shallow = _git('rev-parse', '--is-shallow-repository').stdout.strip() == 'true'
    r = _git('fetch', '-q', *(['--depth', '1'] if shallow else []), 'origin', branch)
    if r.returncode:
        print(f'WARNING: cannot fetch {branch}: {r.stderr.strip()}')
        return None, None
    commit = _git('rev-parse', 'FETCH_HEAD').stdout.strip()
    r = _git('show', f'FETCH_HEAD:{reference_date}/{ED_FILE}')
    if r.returncode:
        return None, commit
    return pd.read_csv(io.StringIO(r.stdout)), commit


def waiting(now=None):
    """True while a submission should still wait for the MetroCast run's file."""
    now = now or datetime.now(ZoneInfo('America/New_York'))
    return now.hour < WAIT_UNTIL_ET


def slug(name):
    return name.lower().replace(' ', '-')


def apply(path, ed, data_dir=None):
    """Replace the ED rows of the submission at path with ObsHYB's where it can (in place).
    Returns (locations replaced, {location: reason} of the ED locations kept from HYB)."""
    df = pd.read_csv(path, dtype={'location': str})
    columns = list(df.columns)
    last = pd.Timestamp(df['reference_date'].iloc[0]) - pd.Timedelta(weeks=1)
    ed = ed[pd.to_datetime(ed['last_date']) == last]
    if len(ed) and not np.allclose(np.sort(ed['quantile'].unique()), hub.QUANTILES):
        raise ValueError(f'{ED_FILE} has quantile levels {sorted(ed["quantile"].unique())}')
    locs = pd.read_csv(os.path.join(data_dir or hub.DATA_DIR, 'locations.csv'), dtype=str)
    state_of = {code: 'us' if code == 'US' else slug(name) for code, name in zip(locs['location'], locs['location_name'])}
    data = pd.read_csv(os.path.join(data_dir or hub.DATA_DIR, hub.TARGETS[ED_TARGET]['file']),
                       dtype={'location': str}, parse_dates=['date'])
    observed = data[data['date'] == last].set_index('location')['value']

    replaced, kept, values = [], {}, []
    for loc in sorted(df.loc[df['target'] == ED_TARGET, 'location'].unique()):
        g = ed[ed['state'] == state_of.get(loc)]
        if g.empty:
            kept[loc] = f'no ObsHYB forecast from {last:%Y-%m-%d}'
            continue
        ours, theirs = g['last_value'].iloc[0], observed.get(loc, np.nan)
        if not np.isfinite(theirs) or abs(ours - theirs) > max(ABS_TOL, REL_TOL * theirs):
            kept[loc] = f'last value {ours:.5f} (NSSP) vs {theirs:.5f} (FluSight data)'
            continue
        v = g.pivot_table(index='horizon', columns='quantile', values='value').sort_index()
        values.append(np.sort(v.to_numpy(), axis=1))
        replaced.append(loc)
    if replaced:
        fc = Forecast(target=ED_TARGET, last_date=last, locations=replaced, output_type=hub.OUTPUT_TYPE,
                      output_type_ids=hub.QUANTILES, values=np.stack(values))
        new = hub.submission_frame(fc, hub.populations(data_dir))
        old = (df['target'] == ED_TARGET) & df['location'].isin(replaced)
        df = pd.concat([df[~old], new], ignore_index=True)
        order = {t: i for i, t in enumerate(hub.TARGETS)}
        df = df.sort_values(['target', 'location', 'horizon', 'output_type_id'],
                            key=lambda s: s.map(order) if s.name == 'target' else s, kind='stable')
        df[columns].to_csv(path, index=False)
    return replaced, kept


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def use_obshyb_ed(path, ed, commit, summary, branch=RUNS_BRANCH):
    """Apply ed to the recorded submission at path; copy it into the run record and update run.json."""
    replaced, kept = apply(path, ed)
    run_dir = os.path.dirname(path)
    ed.to_csv(os.path.join(run_dir, RECORD_FILE), index=False)
    run_json = os.path.join(run_dir, 'run.json')
    if os.path.exists(run_json):
        with open(run_json) as f:
            record = json.load(f)
        record['submission']['sha256'] = sha256(path)
        record['ed_obshyb'] = {'file': RECORD_FILE, 'sha256': sha256(os.path.join(run_dir, RECORD_FILE)),
                               'branch': branch, 'commit': commit, 'replaced': replaced, 'kept': kept}
        with open(run_json, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2)
    summary.append(f'ED: DMAPRIME-ObsHYB for {len(replaced)} locations ({branch}@{(commit or "")[:7]})')
    if kept:
        summary.append(f'WARNING: ED kept from HYB for {len(kept)} locations: '
                       + '; '.join(f'{k} ({v})' for k, v in kept.items()))


def main():
    path, ed_path = sys.argv[1:3]
    replaced, kept = apply(path, pd.read_csv(ed_path), sys.argv[3] if len(sys.argv) > 3 else None)
    print(f'ED replaced for {len(replaced)} locations; kept from HYB: {kept}')


if __name__ == '__main__':
    main()
