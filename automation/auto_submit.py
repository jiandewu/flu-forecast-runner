"""Automation: run dmaprime.forecast, upload the submission file to the fork, open a PR.

Requires GITHUB_TOKEN / GITHUB_REPO / GITHUB_BRANCH (environment or .env) for the upload, and a gh login for the pull request. Before uploading
to the fork's main branch, the fork is synced with upstream; after the upload, a PR from the
fork's main to the upstream hub is opened (or an open one is reused). A failed sync or PR
only prints a warning. Options not listed below are passed through to dmaprime.forecast,
e.g. `python -m automation.auto_submit --seeds 0 1 2 3 4`.

Before anything else (forecast, upload, PR), a submission to the fork's main is checked:
the repository is on its run branch (dev; RUN_BRANCH) with no uncommitted changes; the hub data (synced first) runs
through the most recent Saturday, so the reference date is this week's; and this week has
not been run yet (no submissions/<reference_date>/run.json) and the fork's main does not
already have this week's file (so a run here and one on GitHub Actions never both submit).
With --upload-only, the file's reference date must be this week's instead. If a check
fails, nothing is done. --force skips the checks; test branches (--branch) are not checked.
--no-upload (the shadow runs on GitHub Actions) does the checks, the forecast and the
submission check, but does not touch the fork; it skips the fork check.

When the config sets "ed_model": "obshyb", the ED rows come from DMAPRIME-ObsHYB
(automation/flusight_ed.py): the file written by this week's MetroCast run (branch runs-metrocast;
runs-metrocast-shadow for --no-upload). Until 21:00 ET a submission waits for it (exit 2, like
'no new data yet'); later, or without it, HYB's ED forecast is submitted with a warning.

The submission file is then checked against the hub's rules (automation/check_submission.py)
before it is uploaded; a file that fails is not uploaded. Each run is appended to
outputs/flusight/pooled_hyb/logs/auto_submit.log, and an email with the summary and the
run's log is sent (local runs) when the run succeeds or fails (not when it only stops
because the hub has not published this week's data yet or this week is already done;
--no-email turns emails off). On GitHub Actions the summary also goes to the run's page.
Exit codes: 0 success, 1 stopped by a problem, 2 nothing to do (no new data yet, or this
week already run or submitted), 3 error.

    python -m automation.auto_submit                          # forecast, sync fork, upload, open PR
    python -m automation.auto_submit --upload-only <file> --pr-dry-run   # upload, print the PR only
    python -m automation.auto_submit --branch test-auto-submit --upload-only <file>   # test upload
    python -m automation.auto_submit --no-upload              # forecast and check only (shadow run)
"""
import argparse
import io
import json
import os
import sys
import traceback
from datetime import date, datetime

from dmaprime import forecast
from dmaprime.hubs import flusight as hub
from dmaprime.utils.env import load_env_file
from automation import flusight_ed
from automation.check_submission import check
from automation.github_pr import open_pull_request
from automation.github_upload import file_exists, push_file_to_github, sync_fork
from automation.week import BASE, RUN_BRANCH, data_status, repo_state, this_week

LOG = os.path.join(BASE, 'logs', 'auto_submit.log')


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


def upload_forecast(local_path, branch):
    """Upload to model-output/<model_id>/ on the fork; returns (upload result, PR/commit title)."""
    # Submission files are named <reference_date>-<team>-<model>.csv
    filename = os.path.basename(local_path)
    forecast_week = filename[:10]
    model_id = os.path.splitext(filename)[0][len(forecast_week) + 1:]
    title = f"Forecast for week {forecast_week}"
    result = push_file_to_github(local_path, f"model-output/{model_id}/{filename}", title, branch=branch)
    print(f"GitHub upload {result['action']} on {result['repo']}:{result['branch']}: {result['html_url']}")
    return result, title


def ed_model():
    with open(forecast.config_path(hub.NAME, 'pooled_hyb')) as f:
        return json.load(f).get('ed_model', 'hyb')


def preflight(args, forecast_args):
    """Stop before any work unless this week's submission can be made. Returns forecast args."""
    want, reference_date, run_json = this_week(date.today())

    branch, dirty = repo_state()
    if branch != RUN_BRANCH or dirty:
        raise Stop(f'the repository must be on {RUN_BRANCH} with no uncommitted changes (branch={branch}), '
                   f'so run.json records committed code. Use --force to submit anyway.')

    if args.upload_only:
        file_date = os.path.basename(args.upload_only)[:10]
        if file_date != reference_date:
            raise Stop(f"{args.upload_only} is for {file_date}, but this week's reference date is "
                       f'{reference_date}. Use --force to upload it anyway.')
        return forecast_args

    if '--no-sync' not in forecast_args:
        try:
            hub.sync(verbose=False)
        except RuntimeError as exc:
            raise Stop(f'data sync failed: {exc}')
        forecast_args = forecast_args + ['--no-sync']  # already synced
    ready, status = data_status(want)
    if not ready:
        raise Stop(f'data not yet through {want} ({status}); the hub has not published this '
                   f"week's data. Nothing was run or uploaded.", notify=False)
    if os.path.exists(run_json):
        raise Stop(f'{reference_date} was already run ({run_json}). To upload that file, use '
                   f'--upload-only; to run the forecast again, use --force.', notify=False)
    if not args.no_upload:
        with open(forecast.config_path(hub.NAME, 'pooled_hyb')) as f:
            model_id = json.load(f)['model_id']
        remote = f'model-output/{model_id}/{reference_date}-{model_id}.csv'
        if file_exists(remote, branch='main'):
            raise Stop(f"the fork's main already has {remote}: this week was submitted from "
                       f'elsewhere. Use --force to submit again.', notify=False)
        if ed_model() == 'obshyb':
            args.ed = flusight_ed.fetch(reference_date)
            if args.ed[0] is None and flusight_ed.waiting():
                raise Stop(f'waiting for the ED forecast of the MetroCast run ({flusight_ed.RUNS_BRANCH} has no '
                           f'{reference_date}/{flusight_ed.ED_FILE} yet; after {flusight_ed.WAIT_UNTIL_ET}:00 ET '
                           f"HYB's ED forecast is submitted instead)", notify=False)
    print(f'Checks passed: data through {want} ({status}); reference date {reference_date}')
    return forecast_args


def submit(args, forecast_args, summary):
    """The run itself; appends the lines for the email to summary."""
    branch = args.branch or os.environ.get('GITHUB_BRANCH') or 'main'
    if args.force:
        print('Checks skipped (--force)')
    elif branch != 'main':
        print(f'Checks skipped for test branch {branch}')
    else:
        forecast_args = preflight(args, forecast_args)

    if args.upload_only:
        path = args.upload_only
    else:
        # One file with every location and both targets
        sys.argv = [sys.argv[0]] + forecast_args
        path = forecast.main()
        if ed_model() == 'obshyb':
            ed_branch = flusight_ed.SHADOW_BRANCH if args.no_upload else flusight_ed.RUNS_BRANCH
            ed, commit = getattr(args, 'ed', None) or flusight_ed.fetch(os.path.basename(path)[:10], ed_branch)
            if ed is None:
                summary.append(f'WARNING: ED from HYB: no {flusight_ed.ED_FILE} from the MetroCast run ({ed_branch})')
            else:
                flusight_ed.use_obshyb_ed(path, ed, commit, summary, ed_branch)
    summary.append(f'File: {path}')

    # The submission window only matters for a real submission
    errors, notes = check(path, window=branch == 'main' and not args.force)
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

    if args.no_fork_sync:
        pass
    elif branch == 'main':
        if sync_fork(branch='main') is None:
            summary.append('WARNING: fork sync failed (the upload went ahead; see the log)')
    else:
        # Test branches have no upstream counterpart; keep the fork's main untouched
        print(f'Fork sync skipped for branch {branch}')
    result, title = upload_forecast(path, branch)
    summary.append(f"Upload: {result['html_url']}")

    if args.no_pr:
        summary.append('Pull request: not opened (--no-pr)')
        return
    if branch != 'main' and not args.pr_dry_run:
        # Test branches never become real submissions
        print(f'Pull request skipped for branch {branch}')
        summary.append(f'Pull request: skipped for test branch {branch}')
        return
    url = open_pull_request(result['repo'], branch, title, dry_run=args.pr_dry_run)
    if url is None:
        summary.append('WARNING: pull request NOT opened; open it manually (see the log)')
    else:
        summary.append(f'Pull request{" (dry run)" if args.pr_dry_run else ""}: {url}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--branch', default=None, help='target branch on the fork (default: GITHUB_BRANCH, else main)')
    parser.add_argument('--upload-only', metavar='FILE', help='upload an existing submission file; skip the forecast')
    parser.add_argument('--no-fork-sync', action='store_true', help='do not sync the fork with upstream first')
    parser.add_argument('--no-pr', action='store_true', help='upload only; do not open a pull request')
    parser.add_argument('--pr-dry-run', action='store_true', help='print the pull request instead of opening it')
    parser.add_argument('--force', action='store_true', help="skip the checks on the repository, data and this week's run")
    parser.add_argument('--no-email', action='store_true', help='do not send the result email')
    parser.add_argument('--no-upload', action='store_true', help='forecast and check the file only (shadow run)')
    args, forecast_args = parser.parse_known_args()
    if '--reproduce' in forecast_args:
        parser.error('--reproduce does not produce a new submission; run it with dmaprime.forecast')
    load_env_file('.env')

    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    run_text = io.StringIO()
    summary = []
    with open(LOG, 'a', encoding='utf-8') as log_file:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(stdout, log_file, run_text)
        sys.stderr = _Tee(stderr, log_file, run_text)
        try:
            print(f'===== {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}  auto_submit {" ".join(sys.argv[1:])}')
            try:
                submit(args, forecast_args, summary)
                status, code, notify = 'OK', 0, True
                if any(line.startswith('WARNING') for line in summary):
                    status = 'OK with warnings'
            except Stop as stop:
                print(f'STOP: {stop}')
                status, code, notify = 'STOPPED', 1 if stop.notify else 2, stop.notify
                summary.insert(0, f'Stopped: {stop}')
            except Exception as exc:
                traceback.print_exc()
                status, code, notify = 'FAILED', 3, True
                summary.insert(0, f'Error: {type(exc).__name__}: {exc} (traceback in the log below)')
            print(f'===== {status}')
        finally:
            sys.stdout, sys.stderr = stdout, stderr

    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as f:
            f.write(f'## auto_submit: {status}\n\n' + ''.join(f'- {line}\n' for line in summary))
    if notify and not args.no_email:
        path = next((line[len('File: '):] for line in summary if line.startswith('File: ')), '')
        subject = f'[DMAPRIME-HYB] {status} {os.path.basename(path)}'.strip()
        body = '\n'.join(summary) + f'\n\nLog file: {os.path.abspath(LOG)}\n\n' + run_text.getvalue()
        from automation.notify import send_email   # local runs only; GitHub Actions runs use --no-email
        send_email(subject, body)
    return code


if __name__ == "__main__":
    sys.exit(main())
