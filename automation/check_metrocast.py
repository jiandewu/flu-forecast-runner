"""Check a Flu MetroCast submission file against the hub's rules before uploading.

Rules from hub-config/tasks.json (latest round) and the hub's custom validation
src/validations/R/check_target_end_date.R:
    file name <reference_date>-<model_id>.parquet; columns reference_date, target, location,
    target_end_date, output_type, output_type_id, value (no others; no horizon column)
    reference_date is a round of tasks.json; the submission window is reference_date -6 to -2 days
    target and location are valid for the round; for every location target_end_date covers
    exactly reference_date + 0..3 weeks (long_term_forecasts: false)
    output_type 'sample'; output_type_id a string of at most 20 characters; exactly 100 samples
    per (target, location, target_end_date); every sample id belongs to one location (a
    trajectory) and covers every target_end_date of that location
    value finite and within [0, 100]

Usage:
    python -m automation.check_metrocast <file> [--no-window] [--tasks data/metrocast/tasks.json]
"""
import argparse
import json
import os
import re
import sys
from datetime import date, datetime, timedelta

import pandas as pd
import pyarrow.parquet as pq

COLUMNS = ['reference_date', 'target', 'location', 'target_end_date', 'output_type', 'output_type_id', 'value']
SHORT_TERM_LAST_HORIZON = 3
FIRST_HORIZON = {'Flu hosp admits': -1}


def round_spec(tasks_path):
    with open(tasks_path) as f:
        rnd = json.load(f)['rounds'][-1]
    spec = {}
    for task in rnd['model_tasks']:
        ids = task['task_ids']
        opts = lambda k: set((ids[k]['required'] or []) + (ids[k]['optional'] or []))
        for target in opts('target'):
            spec[target] = {'reference_dates': opts('reference_date'), 'locations': opts('location'),
                            'target_end_dates': opts('target_end_date'),
                            'sample': task['output_type']['sample']}
    return spec, rnd['submissions_due']


def check(path, tasks_path, window=True, today=None, rounds=True):
    """Problems found in the file (empty: OK). rounds=False skips the check that the reference date
    is a round of tasks.json (shadow runs before the season)."""
    errors = []
    name = os.path.basename(path)
    m = re.fullmatch(r'(\d{4}-\d{2}-\d{2})-([A-Za-z0-9_]+-[A-Za-z0-9_]+)\.parquet', name)
    if not m:
        return [f'file name {name} is not <YYYY-MM-DD>-<team>-<model>.parquet']
    ref = date.fromisoformat(m.group(1))
    schema = pq.read_schema(path)
    df = pd.read_parquet(path)
    if list(df.columns) != COLUMNS and sorted(df.columns) != sorted(COLUMNS):
        errors.append(f'columns {list(df.columns)} != {COLUMNS}')
        return errors
    for col, kind in (('reference_date', 'date32'), ('target_end_date', 'date32'), ('output_type_id', 'string'),
                      ('value', 'double'), ('target', 'string'), ('location', 'string'), ('output_type', 'string')):
        t = str(schema.field(col).type)
        if not t.startswith(kind) and not (kind == 'string' and t == 'large_string'):
            errors.append(f'column {col} has type {t}, expected {kind}')
    spec, due = round_spec(tasks_path)

    refs = set(pd.to_datetime(df['reference_date']).dt.date)
    if refs != {ref}:
        errors.append(f'reference_date values {sorted(refs)} differ from the file name {ref}')
    if window:
        today = today or datetime.now().date()
        start, end = ref + timedelta(days=due['start']), ref + timedelta(days=due['end'])
        if not start <= today <= end:
            errors.append(f'today {today} is outside the submission window {start} to {end}')
    if (df['output_type'] != 'sample').any():
        errors.append(f"output_type values {sorted(df['output_type'].unique())}, expected only 'sample'")
    if df['output_type_id'].str.len().max() > 20:
        errors.append('output_type_id longer than 20 characters')
    if not df['value'].notna().all():
        errors.append(f"{int(df['value'].isna().sum())} missing values")

    df = df.assign(ted=pd.to_datetime(df['target_end_date']).dt.date)
    for target, g in df.groupby('target'):
        if target not in spec:
            errors.append(f'target {target} is not in the round')
            continue
        s = spec[target]
        if rounds and ref.isoformat() not in s['reference_dates']:
            errors.append(f'{target}: reference_date {ref} is not a round of tasks.json')
        bad = sorted(set(g['location']) - s['locations'])
        if bad:
            errors.append(f'{target}: invalid locations {bad}')
        h0 = FIRST_HORIZON.get(target, 0)
        expected = {ref + timedelta(weeks=h) for h in range(h0, SHORT_TERM_LAST_HORIZON + 1)}
        for loc, gl in g.groupby('location'):
            have = set(gl['ted'])
            if have != expected:
                errors.append(f'{target} {loc}: target_end_dates {sorted(have)} != {sorted(expected)}')
        n = g.groupby(['location', 'ted']).size()
        p = s['sample']['output_type_id_params']
        off = n[(n < p['min_samples_per_task']) | (n > p['max_samples_per_task'])]
        if len(off):
            errors.append(f'{target}: {len(off)} location-weeks without exactly {p["min_samples_per_task"]} samples')
        if g.duplicated(['location', 'ted', 'output_type_id']).any():
            errors.append(f'{target}: duplicate sample ids within a location-week')
        locs_per_id = g.groupby('output_type_id')['location'].nunique()
        if (locs_per_id > 1).any():
            errors.append(f'{target}: {int((locs_per_id > 1).sum())} sample ids span several locations')
        weeks_per_id = g.groupby('output_type_id')['ted'].nunique()
        if (weeks_per_id != len(expected)).any():
            errors.append(f'{target}: {int((weeks_per_id != len(expected)).sum())} sample ids miss some target_end_dates')
        v = s['sample']['value']
        lo, hi = v.get('minimum', float('-inf')), v.get('maximum', float('inf'))
        out = g[(g['value'] < lo) | (g['value'] > hi)]
        if len(out):
            errors.append(f'{target}: {len(out)} values outside [{lo}, {hi}]')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('file')
    parser.add_argument('--tasks', default=os.path.join('data', 'metrocast', 'tasks.json'))
    parser.add_argument('--no-window', action='store_true', help='skip the submission window check')
    args = parser.parse_args()
    errors = check(args.file, args.tasks, window=not args.no_window)
    df = pd.read_parquet(args.file)
    print(f"{args.file}: {len(df)} rows, {df['location'].nunique()} locations")
    for e in errors:
        print(f'ERROR: {e}')
    print('OK' if not errors else f'{len(errors)} problem(s)')
    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()
