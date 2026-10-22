"""Check a FluSight submission file against the hub's rules before uploading.

Rules come from the hub's hub-config/tasks.json (downloaded each time, cached in
data/flusight/tasks.json) plus README requirements the hub's own checks do not enforce:
    - columns, file name and a single reference date in the round's list
    - submission window (submissions_due in tasks.json), unless window=False
    - target, location, horizon and output_type_id allowed for each target
    - every (target, location, horizon) has each required quantile exactly once
    - target_end_date = reference_date + horizon weeks
    - values finite and within [minimum, maximum]; admission counts are integers
    - quantiles non-decreasing
    - plausibility: admissions <= 30% of population, ED proportion <= 0.25

    python -m automation.check_submission <file> [--no-window]
"""
import argparse
import functools
import json
import os
import sys
from datetime import date, timedelta
from urllib.request import urlopen

import numpy as np
import pandas as pd

from dmaprime.hubs import flusight as hub

COLUMNS = ['reference_date', 'target', 'horizon', 'target_end_date', 'location',
           'output_type', 'output_type_id', 'value']
TASKS_CACHE = os.path.join(hub.DATA_DIR, 'tasks.json')


@functools.lru_cache(maxsize=1)
def load_tasks():
    """Latest tasks.json from the hub (once per process); fall back to the cached copy if the download fails."""
    try:
        with urlopen(f'{hub.REMOTE_BASE_URL}/hub-config/tasks.json', timeout=30) as response:
            text = response.read().decode('utf-8')
        tasks = json.loads(text)
        os.makedirs(os.path.dirname(TASKS_CACHE), exist_ok=True)
        with open(TASKS_CACHE, 'w', encoding='utf-8') as f:
            f.write(text)
        return tasks, []
    except Exception as exc:  # network or JSON error
        if os.path.exists(TASKS_CACHE):
            with open(TASKS_CACHE, encoding='utf-8') as f:
                return json.load(f), [f'tasks.json download failed ({exc}); used the cached copy']
        raise RuntimeError(f'tasks.json download failed and no cached copy: {exc}') from exc


def _allowed(task_id):
    return (task_id.get('required') or []) + (task_id.get('optional') or [])


def check(path, window=True, today=None):
    """Return (errors, notes). The file is fine to upload when errors is empty."""
    errors, notes = [], []
    tasks, warnings = load_tasks()
    notes += warnings
    round_ = tasks['rounds'][0]

    df = pd.read_csv(path, dtype={'location': str, 'reference_date': str, 'target_end_date': str})
    if list(df.columns) != COLUMNS:
        return [f'columns are {list(df.columns)}, expected {COLUMNS}'], notes
    if df.empty:
        return ['file has no rows'], notes
    if df['value'].isna().any() or not np.isfinite(df['value']).all():
        errors.append(f'{(~np.isfinite(df["value"])).sum()} values are missing or not finite')

    refs = df['reference_date'].unique()
    if len(refs) != 1:
        return errors + [f'expected one reference_date, found {sorted(refs)}'], notes
    ref = refs[0]
    if not os.path.basename(path).startswith(ref + '-'):
        errors.append(f'file name {os.path.basename(path)} does not start with reference_date {ref}')

    # Every model task shares the reference_date list
    if ref not in _allowed(round_['model_tasks'][0]['task_ids']['reference_date']):
        errors.append(f'reference_date {ref} is not a round in tasks.json')
    due = round_.get('submissions_due', {})
    if window and due.get('relative_to') == 'reference_date':
        today = today or date.today()
        start = date.fromisoformat(ref) + timedelta(days=due['start'])
        end = date.fromisoformat(ref) + timedelta(days=due['end'])
        if not start <= today <= end:
            errors.append(f'today {today} is outside the submission window {start} to {end} for {ref}')

    expected_end = (pd.to_datetime(ref) + pd.to_timedelta(df['horizon'] * 7, unit='D')).dt.strftime('%Y-%m-%d')
    if (df['target_end_date'] != expected_end).any():
        errors.append(f'{(df["target_end_date"] != expected_end).sum()} rows have target_end_date '
                      f'!= reference_date + horizon weeks')

    pops = hub.populations()
    for (target, output_type), rows in df.groupby(['target', 'output_type']):
        task = next((t for t in round_['model_tasks']
                     if target in _allowed(t['task_ids']['target']) and output_type in t['output_type']), None)
        if task is None:
            errors.append(f'{target} / {output_type} is not accepted by the hub')
            continue
        label = f'{target}:'
        bad_loc = sorted(set(rows['location']) - set(_allowed(task['task_ids']['location'])))
        if bad_loc:
            errors.append(f'{label} locations not accepted: {bad_loc}')
        bad_h = sorted(set(rows['horizon']) - set(_allowed(task['task_ids']['horizon'])))
        if bad_h:
            errors.append(f'{label} horizons not accepted: {bad_h}')

        spec = task['output_type'][output_type]
        ids = [float(q) for q in _allowed(spec['output_type_id'])]
        required = [float(q) for q in (spec['output_type_id'].get('required') or [])]
        bad_ids = sorted(set(rows['output_type_id'].astype(float)) - set(ids))
        if bad_ids:
            errors.append(f'{label} output_type_id not accepted: {bad_ids}')
        counts = rows.groupby(['location', 'horizon', 'output_type_id']).size()
        if (counts > 1).any():
            errors.append(f'{label} {int((counts > 1).sum())} duplicated rows')
        per_task = rows.groupby(['location', 'horizon'])['output_type_id'].apply(lambda s: set(s.astype(float)))
        missing = per_task[per_task.apply(lambda s: not set(required) <= s)]
        if len(missing):
            errors.append(f'{label} {len(missing)} location-horizons miss required quantiles, '
                          f'e.g. {missing.index[0]}')

        value = spec.get('value', {})
        if 'minimum' in value and (rows['value'] < value['minimum']).any():
            errors.append(f'{label} {(rows["value"] < value["minimum"]).sum()} values below {value["minimum"]}')
        if 'maximum' in value and (rows['value'] > value['maximum']).any():
            errors.append(f'{label} {(rows["value"] > value["maximum"]).sum()} values above {value["maximum"]}')
        fractional = np.isfinite(rows['value']) & (rows['value'] != rows['value'].round())
        if target in hub.INTEGER_TARGETS and fractional.any():
            errors.append(f'{label} {fractional.sum()} values are not integers '
                          f'(the hub requires integer counts)')

        if output_type == 'quantile':
            ordered = rows.sort_values(['location', 'horizon', 'output_type_id'])
            decreasing = ordered.groupby(['location', 'horizon'])['value'].apply(
                lambda v: bool((np.diff(v.to_numpy()) < 0).any()))
            if decreasing.any():
                errors.append(f'{label} quantiles decrease for {int(decreasing.sum())} location-horizons, '
                              f'e.g. {decreasing[decreasing].index[0]}')

        if target == 'wk inc flu hosp':
            limit = rows['location'].map(pops) * hub.MAX_HOSP_SHARE_OF_POP
            over = rows['value'] > limit
            if over.any():
                errors.append(f'{label} {over.sum()} values above 30% of population '
                              f'({sorted(rows.loc[over, "location"].unique())})')
        elif target == 'wk inc flu prop ed visits' and (rows['value'] > hub.MAX_ED_PROP).any():
            errors.append(f'{label} {(rows["value"] > hub.MAX_ED_PROP).sum()} values above {hub.MAX_ED_PROP}')

        notes.append(f'{target}: {rows["location"].nunique()} locations, horizons '
                     f'{sorted(int(h) for h in rows["horizon"].unique())}, {len(rows)} rows')
    return errors, notes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('file')
    parser.add_argument('--no-window', action='store_true', help='do not check the submission window')
    args = parser.parse_args()
    errors, notes = check(args.file, window=not args.no_window)
    for line in notes:
        print(f'  {line}')
    for line in errors:
        print(f'ERROR {line}')
    print('Submission check: ' + ('passed' if not errors else f'FAILED ({len(errors)} problems)'))
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
