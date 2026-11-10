"""Cheap check before a weekly COVID-19 / RSV run (first step of weekly_resp.yml).

Standard library only, so it runs before Python packages are installed. Exit 0: go on; 2: nothing to
do this time; 1: error.
    1. this week already run: <runs dir>/<reference_date>/run.json exists
    2. this week's data not out yet: the hub's target-data/time-series.parquet has no commit since the
       Monday after the last Saturday (the hub updates it on Wednesdays, a day later in holiday weeks).
       A commit is only a hint (the hub also commits fixes); automation/resp_submit.py checks that the
       data reach the last Saturday after the download.
Uses GITHUB_TOKEN for the GitHub API when it is set (unauthenticated requests are rate limited).

    python automation/resp_ready.py covid [--runs-dir DIR] [--today YYYY-MM-DD]
"""
import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from urllib.request import Request, urlopen

# Same as dmaprime.hubs.<hub>.REMOTE_BASE_URL (not imported: those modules need pandas)
HUB_REPOS = {'covid': 'CDCgov/covid19-forecast-hub', 'rsv': 'CDCgov/rsv-forecast-hub'}
DATA_PATH = 'target-data/time-series.parquet'


def this_week(today):
    """(Saturday the new release should end on, the reference date it gives)."""
    want = today - timedelta(days=(today.weekday() - 5) % 7)
    return want, want + timedelta(weeks=1)


def last_data_commit(repo):
    """UTC time of the newest commit touching the hub's time series."""
    req = Request(f'https://api.github.com/repos/{repo}/commits?path={DATA_PATH}&per_page=1',
                  headers={'Accept': 'application/vnd.github+json', 'User-Agent': 'resp-ready'})
    if os.environ.get('GITHUB_TOKEN'):
        req.add_header('Authorization', f'Bearer {os.environ["GITHUB_TOKEN"]}')
    with urlopen(req, timeout=30) as resp:
        commits = json.loads(resp.read().decode('utf-8'))
    if not commits:
        raise RuntimeError(f'no commits for {DATA_PATH} in {repo}')
    return datetime.fromisoformat(commits[0]['commit']['committer']['date'].replace('Z', '+00:00'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('hub', choices=list(HUB_REPOS))
    parser.add_argument('--runs-dir', default=None, help='default: outputs/<hub>/pooled_hyb/submissions')
    parser.add_argument('--today', default=None)
    args = parser.parse_args()
    today = date.fromisoformat(args.today) if args.today else date.today()
    want, reference_date = this_week(today)
    runs_dir = args.runs_dir or os.path.join('outputs', args.hub, 'pooled_hyb', 'submissions')
    run_json = os.path.join(runs_dir, reference_date.isoformat(), 'run.json')
    if os.path.exists(run_json):
        print(f'nothing to do: {reference_date} was already run ({run_json})')
        return 2
    monday = datetime.combine(want + timedelta(days=2), datetime.min.time(), timezone.utc)
    try:
        last = last_data_commit(HUB_REPOS[args.hub])
    except Exception as exc:   # the full check after the download decides then
        print(f'go: could not read the hub commits ({exc}); the data check after the download decides')
        return 0
    if last < monday:
        print(f'not yet: {DATA_PATH} last changed {last:%Y-%m-%d %H:%M} UTC, before {monday:%Y-%m-%d}; '
              f'reference date {reference_date} not run yet')
        return 2
    print(f'go: {DATA_PATH} changed {last:%Y-%m-%d %H:%M} UTC; reference date {reference_date} not run yet')
    return 0


if __name__ == '__main__':
    sys.exit(main())
