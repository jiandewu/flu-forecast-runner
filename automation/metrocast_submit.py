"""Automation for Flu MetroCast: forecast DMAPRIME-ObsHYB, check the Parquet file,
upload it to the hub fork and open a pull request to reichlab/flu-metrocast.

Separate from automation/auto_submit.py (FluSight), whose behavior it does not change; it reuses
the upload (automation/github_upload.py) and pull-request (automation/github_pr.py) helpers.

Settings (environment or .env):
    METROCAST_REPO    the fork, default DMA-PRIME/flu-metrocast (used as GITHUB_REPO for the upload)
    METROCAST_TOKEN   fine-grained token for that fork (used as GITHUB_TOKEN for the upload);
                      falls back to GITHUB_TOKEN
    PR_TOKEN          as for FluSight: classic token for opening the PR on GitHub Actions

Before anything else: the repository is on its run branch (dev; RUN_BRANCH) with no uncommitted changes; after syncing,
the hub's NSSP locations have data through the most recent Saturday (the week the new release
ends on) and so does the national NSSP series (or, from 18:00 ET, it is one week behind: then every
location is forecast from NSSP's last week, see automation/metrocast_ready.py); this week has not been run yet (no submissions/<reference_date>/run.json); and the
fork's main does not already have this week's file. --force skips these; --no-upload only
forecasts and checks (shadow run). Each run is appended to
outputs/metrocast/pooled_obshyb/logs/metrocast_submit.log.
Exit codes: 0 success, 1 stopped by a problem, 2 nothing to do, 3 error.

    python -m automation.metrocast_submit --no-upload          # forecast and check only
    python -m automation.metrocast_submit                      # forecast, sync fork, upload, open PR
    python -m automation.metrocast_submit --upload-only <file> --pr-dry-run
"""
import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import date, datetime, timedelta

import numpy as np

from dmaprime import forecast
from dmaprime.hubs import metrocast as hub
from dmaprime.utils.env import load_env_file
from automation.check_metrocast import check
from automation.metrocast_ready import decide, now_et
from automation.week import RUN_BRANCH

MODEL = 'pooled_obshyb'
BASE = os.path.join('outputs', hub.NAME, MODEL)
LOG = os.path.join(BASE, 'logs', 'metrocast_submit.log')
DEFAULT_REPO = 'DMA-PRIME/flu-metrocast'


class Stop(Exception):
    def __init__(self, message, code=1):
        super().__init__(message)
        self.code = code


def log(message):
    line = f'{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}  {message}'
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def this_week(today):
    """(Saturday the new release should end on, the reference date it gives)."""
    want = today - timedelta(days=(today.weekday() - 5) % 7)
    return want, (want + timedelta(weeks=1)).isoformat()


def model_id():
    with open(forecast.config_path(hub.NAME, MODEL)) as f:
        return json.load(f)['model_id']


def use_fork_settings():
    """Point the upload helper (GITHUB_REPO / GITHUB_TOKEN) at the MetroCast fork."""
    os.environ['GITHUB_REPO'] = os.environ.get('METROCAST_REPO', DEFAULT_REPO)
    if os.environ.get('METROCAST_TOKEN'):
        os.environ['GITHUB_TOKEN'] = os.environ['METROCAST_TOKEN']


def preflight(no_upload):
    """Stop (Stop exception) unless this week's submission can be made; syncs the data."""
    want, reference_date = this_week(date.today())
    branch = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'],
                                capture_output=True, text=True).stdout.strip())
    if branch != RUN_BRANCH or dirty:
        raise Stop(f'the repository must be on {RUN_BRANCH} with no uncommitted changes (branch={branch})')
    hub.sync(verbose=False)
    panel = hub.load_panel()
    nssp_hub = [l for l in panel.hub_locations if l in hub._hub_nssp_keys(hub.hub_locations())]
    cols = [panel.locations.index(l) for l in nssp_hub]
    last = panel.dates[np.isfinite(panel.z[hub.TARGET][:, cols]).any(axis=1)].max().date()
    nssp_last = panel.nssp_through.date() if panel.nssp_through is not None else None
    decision, message = decide(last, nssp_last, want, now_et())   # the rule of the quick check
    if decision == 'wait':
        raise Stop(message, code=2)
    if decision == 'fallback':
        log(f'WARNING: {message}')
    if os.path.exists(os.path.join(BASE, 'submissions', reference_date, 'run.json')):
        raise Stop(f'{reference_date} was already run', code=2)
    if not no_upload:
        from automation.github_upload import file_exists
        mid = model_id()
        if file_exists(f'model-output/{mid}/{reference_date}-{mid}.parquet', branch='main'):
            raise Stop(f"the fork's main already has this week's file", code=2)
    log(f'checks passed: data through {last}, reference date {reference_date}')
    return reference_date


def check_file(path, window, rounds=True):
    errors = check(path, os.path.join(hub.DATA_DIR, 'tasks.json'), window=window, rounds=rounds)
    if errors:
        raise Stop('submission check failed; not uploaded:\n  ' + '\n  '.join(errors))
    log('submission check passed')


def upload(path, pr_dry_run, summary):
    """Sync the fork, upload the file to model-output/<model_id>/ and open (or reuse) the PR."""
    from automation.github_pr import open_pull_request
    from automation.github_upload import push_file_to_github, sync_fork
    if sync_fork(branch='main') is None:
        summary.append('WARNING: fork sync failed (the upload went ahead)')
    name = os.path.basename(path)
    title = f'Forecast for week {name[:10]}'
    result = push_file_to_github(path, f'model-output/{model_id()}/{name}', title, branch='main')
    summary.append(f"Upload: {result['html_url']}")
    url = open_pull_request(result['repo'], 'main', title, dry_run=pr_dry_run)
    summary.append(f'Pull request: {url}' if url else 'WARNING: pull request NOT opened; open it manually')


def run(args, summary):
    if args.upload_only:
        path = args.upload_only
    else:
        if not args.force:
            preflight(args.no_upload)
        sys.argv = [sys.argv[0], '--hub', hub.NAME, '--model', MODEL] + (['--no-sync'] if not args.force else [])
        path = forecast.main()
    summary.append(f'File: {path}')
    check_file(path, window=not (args.force or args.no_upload), rounds=not args.no_upload)
    if args.no_upload:
        summary.append('Upload: none (--no-upload)')
        return
    upload(path, args.pr_dry_run, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--upload-only', metavar='FILE', help='upload an existing submission file')
    parser.add_argument('--no-upload', action='store_true', help='forecast and check only')
    parser.add_argument('--pr-dry-run', action='store_true', help='print the pull request instead of opening it')
    parser.add_argument('--force', action='store_true', help='skip the checks on the repository, data and this week')
    args = parser.parse_args()
    load_env_file('.env')
    use_fork_settings()
    summary = []
    log(f'===== metrocast_submit {" ".join(sys.argv[1:])}')
    try:
        run(args, summary)
        status, code = 'OK', 0
    except Stop as stop:
        log(f'STOP: {stop}')
        status, code = 'STOPPED', stop.code
        summary.insert(0, f'Stopped: {stop}')
    except Exception as exc:
        traceback.print_exc()
        status, code = 'FAILED', 3
        summary.insert(0, f'Error: {type(exc).__name__}: {exc}')
    for line in summary:
        log(line)
    log(f'===== {status}')
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as f:
            f.write(f'## metrocast_submit: {status}\n\n' + ''.join(f'- {line}\n' for line in summary))
    return code


if __name__ == '__main__':
    sys.exit(main())
