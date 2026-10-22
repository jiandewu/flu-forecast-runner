"""Weekly FluSight helpers shared by the weekly FluSight scripts."""
import os
import subprocess
from datetime import timedelta

import numpy as np

from dmaprime.hubs import flusight as hub

MODEL = 'pooled_hyb'
BASE = os.path.join('outputs', hub.NAME, MODEL)
# The branch submissions run from: dev here; a repository that only runs the weekly submissions sets
# RUN_BRANCH to its default branch (the workflows pass github.event.repository.default_branch)
RUN_BRANCH = os.environ.get('RUN_BRANCH') or 'dev'


def git(*args):
    return subprocess.run(['git', *args], capture_output=True, text=True).stdout.strip()


def expected_last_week(today):
    """Most recent Saturday on or before today: the week the new release should end on."""
    return today - timedelta(days=(today.weekday() - 5) % 7)


def this_week(today):
    """(Saturday the new release should end on, the reference date it gives, that week's run.json)."""
    want = expected_last_week(today)
    reference_date = (want + timedelta(weeks=1)).isoformat()
    return want, reference_date, os.path.join(BASE, 'submissions', reference_date, 'run.json')


def data_status(want):
    """(every target has data through want, 'target: last week, ...') from the local data."""
    panel = hub.load_panel()
    weeks = {t: panel.dates[np.isfinite(panel.z[t]).any(axis=1)].max().date() for t in hub.TARGETS}
    return all(d >= want for d in weeks.values()), ', '.join(f'{t}: {d}' for t, d in weeks.items())


def repo_state():
    """(current branch, whether tracked files have uncommitted changes)."""
    return git('rev-parse', '--abbrev-ref', 'HEAD'), bool(git('status', '--porcelain', '--untracked-files=no'))
