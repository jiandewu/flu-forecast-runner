"""Cheap check before a weekly MetroCast run (first step of weekly_metrocast.yml).

Standard library only, so it runs before Python packages are installed and before the full data
download (about 50 MB of NSSP data). Exit 0: go on; 2: nothing to do this time; 1: error.
    1. this week already run: <runs dir>/<reference_date>/run.json exists
    2. this week's data not released yet (decide(); metrocast_submit.preflight applies the same
       rule again after the full download):
       - hub: some hub location with NSSP data has a value for the last Saturday in the hub's
         latest-data.csv (the target data, the reference date depends on it)
       - NSSP: the national series of data.cdc.gov rdmq-nq56 (our model inputs; the hub's data and
         this release are updated separately, either can come first) reaches the same Saturday
       Fallback: from FALLBACK_HOUR_ET on, if NSSP is exactly one week behind, go anyway; the model
       then forecasts every location from NSSP's last week (pooled_obshyb.align_to_nssp), so the
       inputs stay consistent; the run is reported with a warning.

    python automation/metrocast_ready.py [--runs-dir DIR] [--today YYYY-MM-DD]
"""
import argparse
import csv
import io
import json
import os
import sys
from datetime import date, datetime, timedelta
from urllib.request import urlopen

# Same as dmaprime.hubs.metrocast.REMOTE_BASE_URL (not imported: that module needs pandas)
HUB_RAW = 'https://raw.githubusercontent.com/reichlab/flu-metrocast/main'
TARGET = 'Flu ED visits pct'
NSSP_LATEST_URL = ("https://data.cdc.gov/resource/rdmq-nq56.json?$select=max(week_end)%20as%20w"
                   "&$where=geography=%27United%20States%27")
FALLBACK_HOUR_ET = 18   # submissions are due at 20:00 ET and a run takes about 50 minutes
RUNS_DIR = os.path.join('outputs', 'metrocast', 'pooled_obshyb', 'submissions')


def this_week(today):
    """(Saturday the new release should end on, the reference date it gives)."""
    want = today - timedelta(days=(today.weekday() - 5) % 7)
    return want, want + timedelta(weeks=1)


def read_csv(path, timeout=60):
    with urlopen(f'{HUB_RAW}/{path}', timeout=timeout) as r:
        return list(csv.DictReader(io.StringIO(r.read().decode('utf-8'))))


def nssp_hub_locations(locations):
    """Hub locations whose data are NSSP (as dmaprime.hubs.metrocast._hub_nssp_keys: not NYC, not NC regions)."""
    return {r['location'] for r in locations if r['location_type'] == 'hsa_nci_id' and r['location'] != 'nyc'}


def nssp_latest(timeout=60):
    """Last week of the national NSSP series on data.cdc.gov (one-row query)."""
    with urlopen(NSSP_LATEST_URL, timeout=timeout) as r:
        rows = json.loads(r.read().decode('utf-8'))
    return date.fromisoformat(rows[0]['w'][:10]) if rows and rows[0].get('w') else None


def now_et():
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo('America/New_York'))


def decide(hub_last, nssp_last, want, now):
    """('go' | 'fallback' | 'wait', message) for this week's run; now: datetime in ET."""
    if hub_last is None or hub_last < want:
        return 'wait', f'hub data only through {hub_last}, not {want} (not released yet)'
    if nssp_last is not None and nssp_last >= want:
        return 'go', f'hub and NSSP data through {want}'
    if nssp_last is not None and nssp_last == want - timedelta(weeks=1):
        if now.hour >= FALLBACK_HOUR_ET:
            return 'fallback', (f'NSSP data only through {nssp_last} at {now:%H:%M} ET; every location is '
                                f'forecast from {nssp_last} (fallback)')
        return 'wait', (f'hub data through {hub_last} but NSSP only through {nssp_last}; waiting '
                        f'(fallback from {FALLBACK_HOUR_ET}:00 ET)')
    return 'wait', f'NSSP data only through {nssp_last}, more than a week behind {want}; no automatic run'


def latest_week(rows, locations):
    weeks = []
    for r in rows:
        if r['target'] != TARGET or r['location'] not in locations:
            continue
        try:
            float(r['observation'])
        except ValueError:
            continue
        weeks.append(date.fromisoformat(r['target_end_date'][:10]))
    return max(weeks) if weeks else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs-dir', default=RUNS_DIR)
    parser.add_argument('--today', default=None, help='YYYY-MM-DD (default: today)')
    args = parser.parse_args()
    today = date.fromisoformat(args.today) if args.today else date.today()
    want, reference_date = this_week(today)

    run_json = os.path.join(args.runs_dir, reference_date.isoformat(), 'run.json')
    if os.path.exists(run_json):
        print(f'nothing to do: {reference_date} was already run ({run_json})')
        return 2
    hub_last = latest_week(read_csv('target-data/latest-data.csv'), nssp_hub_locations(read_csv('auxiliary-data/locations.csv')))
    nssp_last = nssp_latest() if hub_last is not None and hub_last >= want else None
    decision, message = decide(hub_last, nssp_last, want, now_et())
    if decision == 'wait':
        print(f'nothing to do: {message}')
        return 2
    print(f'{decision}: {message}; reference date {reference_date} not run yet')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:   # network or format problem: fail the run so GitHub reports it
        print(f'error: {type(exc).__name__}: {exc}')
        sys.exit(1)
