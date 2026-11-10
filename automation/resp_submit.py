"""Automation for the COVID-19 and RSV hubs: run dmaprime.forecast, upload to the fork, open a PR.

The same steps as automation/auto_submit.py (FluSight) for one of the hubs in dmaprime/hubs/cdc_resp.py,
with config/<hub>/pooled_hyb.json:
    checks   the repository is on its run branch (RUN_BRANCH, default dev) with no uncommitted changes;
             the hub data (synced first) reach the most recent Saturday for every target, so the
             reference date is this week's; this week has not been run yet (no
             outputs/<hub>/pooled_hyb/submissions/<reference_date>/run.json); the fork's main does not
             already have this week's file. A failed check stops the run before any work.
    forecast one file with every location and both targets; checked against the hub's rules
             (automation/check_resp_submission.py) and not uploaded if it fails
    upload   the fork (FORKS, never the GITHUB_REPO of .env) is synced with upstream, the file goes to
             model-output/<model_id>/ on its main, and a PR to the upstream hub is opened (or an open
             one reused) with the gh login, or PR_TOKEN on GitHub Actions. The PR author must be in the
             model metadata's designated_github_users for the hub to merge it automatically.
--no-upload does the checks, the forecast and the file check only (shadow); --force skips the checks.
GITHUB_TOKEN: a token that can write to the fork. Each run is appended to
outputs/<hub>/pooled_hyb/logs/resp_submit.log; on GitHub Actions the summary also goes to the run page.
Exit codes: 0 success, 1 stopped by a problem, 2 nothing to do (no new data yet, or this week already
run or submitted), 3 error.

    python -m automation.resp_submit covid                  # forecast, sync the fork, upload, open the PR
    python -m automation.resp_submit rsv --no-upload        # forecast and check only
    python -m automation.resp_submit covid --upload-only <file> --pr-dry-run
"""
import argparse
import io
import json
import os
import sys
import traceback
from datetime import date, datetime, timedelta

import numpy as np

from dmaprime import forecast
from dmaprime.hubs import covid, rsv
from automation.check_resp_submission import check
from automation.github_pr import open_pull_request
from automation.github_upload import file_exists, push_file_to_github, sync_fork
from automation.week import RUN_BRANCH, expected_last_week, git

HUBS = {covid.NAME: covid, rsv.NAME: rsv}
FORKS = {covid.NAME: 'DMA-PRIME/covid19-forecast-hub', rsv.NAME: 'DMA-PRIME/rsv-forecast-hub'}
MODEL = 'pooled_hyb'


class Stop(Exception):
    """A check failed: nothing more is done. notify=False for the routine 'no new data yet'."""
    def __init__(self, message, notify=True):
        super().__init__(message)
        self.notify = notify


class _Tee(io.TextIOBase):
    """Write to the console, the log file and an in-memory copy of this run."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def base_dir(hub):
    return os.path.join('outputs', hub.NAME, MODEL)


def this_week(hub, today):
    """(Saturday the new release should end on, the reference date, that week's run.json)."""
    want = expected_last_week(today)
    reference_date = (want + timedelta(weeks=1)).isoformat()
    return want, reference_date, os.path.join(base_dir(hub), 'submissions', reference_date, 'run.json')


def model_id(hub):
    with open(forecast.config_path(hub.NAME, MODEL)) as f:
        return json.load(f)['model_id']


def data_status(hub, want):
    """(every target has data through want, 'target: last week, ...') from the local data."""
    panel = hub.load_panel()
    weeks = {t: panel.dates[np.isfinite(panel.z[t]).any(axis=1)].max().date() for t in hub.TARGETS}
    return all(d >= want for d in weeks.values()), ', '.join(f'{t}: {d}' for t, d in weeks.items())


def preflight(hub, args):
    """Stop before any work unless this week's submission can be made."""
    want, reference_date, run_json = this_week(hub, date.today())
    branch = git('rev-parse', '--abbrev-ref', 'HEAD')
    dirty = bool(git('status', '--porcelain', '--untracked-files=no'))
    if branch != RUN_BRANCH or dirty:
        raise Stop(f'the repository must be on {RUN_BRANCH} with no uncommitted changes (branch={branch}), '
                   f'so run.json records committed code. Use --force to submit anyway.')
    if args.upload_only:
        file_date = os.path.basename(args.upload_only)[:10]
        if file_date != reference_date:
            raise Stop(f"{args.upload_only} is for {file_date}, but this week's reference date is "
                       f'{reference_date}. Use --force to upload it anyway.')
        return
    try:
        hub.sync(verbose=False)
    except RuntimeError as exc:
        raise Stop(f'data sync failed: {exc}')
    ready, status = data_status(hub, want)
    if not ready:
        raise Stop(f'data not yet through {want} ({status}); the hub has not published this '
                   f"week's data. Nothing was run or uploaded.", notify=False)
    if os.path.exists(run_json):
        raise Stop(f'{reference_date} was already run ({run_json}). To upload that file, use '
                   f'--upload-only; to run the forecast again, use --force.', notify=False)
    if not args.no_upload:
        remote = f'model-output/{model_id(hub)}/{reference_date}-{model_id(hub)}.csv'
        if file_exists(remote, branch='main'):
            raise Stop(f"the fork's main already has {remote}: this week was submitted from "
                       f'elsewhere. Use --force to submit again.', notify=False)
    print(f'Checks passed: data through {want} ({status}); reference date {reference_date}')


def submit(hub, args, summary):
    if args.force:
        print('Checks skipped (--force)')
    else:
        preflight(hub, args)
    if args.upload_only:
        path = args.upload_only
    else:
        # sync() above already downloaded the data, except with --force
        argv = ['--hub', hub.NAME, '--model', MODEL] + ([] if args.force else ['--no-sync'])
        sys.argv = [sys.argv[0]] + argv
        path = forecast.main()
    summary.append(f'File: {path}')

    errors, notes = check(hub.NAME, path, window=not args.force)
    for line in notes:
        print(f'  {line}')
        summary.append(line)
    if errors:
        for line in errors:
            print(f'ERROR {line}')
        raise Stop(f'submission check failed ({len(errors)} problems); the file was not uploaded:\n  '
                   + '\n  '.join(errors))
    print('Submission check passed')
    if args.no_upload:
        summary.append('Upload: none (--no-upload)')
        return

    if sync_fork(branch='main') is None:
        summary.append('WARNING: fork sync failed (the upload went ahead; see the log)')
    filename = os.path.basename(path)
    title = f'Forecast for week {filename[:10]}'
    result = push_file_to_github(path, f'model-output/{model_id(hub)}/{filename}', title, branch='main')
    print(f"GitHub upload {result['action']} on {result['repo']}:{result['branch']}: {result['html_url']}")
    summary.append(f"Upload: {result['html_url']}")
    url = open_pull_request(result['repo'], 'main', title, dry_run=args.pr_dry_run)
    if url is None:
        summary.append('WARNING: pull request NOT opened; open it manually (see the log)')
    else:
        summary.append(f'Pull request{" (dry run)" if args.pr_dry_run else ""}: {url}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('hub', choices=list(HUBS))
    parser.add_argument('--upload-only', metavar='FILE', help='upload an existing submission file; skip the forecast')
    parser.add_argument('--pr-dry-run', action='store_true', help='print the pull request instead of opening it')
    parser.add_argument('--force', action='store_true', help="skip the checks on the repository, data and this week's run")
    parser.add_argument('--no-upload', action='store_true', help='forecast and check the file only (shadow run)')
    args = parser.parse_args()
    hub = HUBS[args.hub]
    os.environ['GITHUB_REPO'] = FORKS[hub.NAME]   # set before github_upload reads .env, which never overrides it

    log = os.path.join(base_dir(hub), 'logs', 'resp_submit.log')
    os.makedirs(os.path.dirname(log), exist_ok=True)
    run_text = io.StringIO()
    summary = []
    with open(log, 'a', encoding='utf-8') as log_file:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(stdout, log_file, run_text)
        sys.stderr = _Tee(stderr, log_file, run_text)
        try:
            print(f'===== {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}  resp_submit {" ".join(sys.argv[1:])}')
            try:
                submit(hub, args, summary)
                status, code = 'OK', 0
                if any(line.startswith('WARNING') for line in summary):
                    status = 'OK with warnings'
            except Stop as stop:
                print(f'STOP: {stop}')
                status, code = 'STOPPED', 1 if stop.notify else 2
                summary.insert(0, f'Stopped: {stop}')
            except Exception as exc:
                traceback.print_exc()
                status, code = 'FAILED', 3
                summary.insert(0, f'Error: {type(exc).__name__}: {exc} (traceback in the log)')
            print(f'===== {status}')
        finally:
            sys.stdout, sys.stderr = stdout, stderr
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as f:
            f.write(f'## resp_submit {hub.NAME}: {status}\n\n' + ''.join(f'- {line}\n' for line in summary))
    print('\n'.join(summary))
    return code


if __name__ == '__main__':
    sys.exit(main())
