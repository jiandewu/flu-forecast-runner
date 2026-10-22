"""Weekly hub submission: sync data, run a model for every target, write the submission file.

The model's settings (submission model_id, seeds, calibration factors) are read from
config/<hub>/<model>.json.

Every run is recorded in outputs/<hub>/<model>/submissions/<reference_date>/:
    <reference_date>-<model_id>.csv   the submission file
    data/                             snapshot of the hub data files the run read
    run.json                          git commit, effective config, data sha256, versions
--reproduce <reference_date> reruns a recorded week from its snapshot and config and checks
that the result matches the recorded submission (training is deterministic on CPU).

Usage:
    python -m dmaprime.forecast                                  # --hub flusight --model pooled_hyb
    python -m dmaprime.forecast --seeds 0 1 2 3 4                # override the config's seeds
    python -m dmaprime.forecast --hub flusight --model pooled_hyb --no-sync
    python -m dmaprime.forecast --reproduce 2026-10-10
    python -m dmaprime.forecast --hub metrocast --model pooled_obshyb   # Flu MetroCast (Parquet samples)
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from dmaprime.hubs import HUBS
from dmaprime.models import MODELS


def config_path(hub_name, model_name):
    return os.path.join('config', hub_name, f'{model_name}.json')


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def git_state():
    """(commit, dirty) of the working tree, or (None, None) when git is unavailable."""
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def versions():
    import torch
    return {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
            'torch': torch.__version__, 'platform': platform.platform()}


def run_forecast(hub, model, config, out_dir, data_dir=None):
    panel = hub.load_panel(data_dir=data_dir)
    forecasts = [model.forecast(hub, panel, target, config) for target in hub.TARGETS]
    return hub.write_submission(forecasts, config['model_id'], out_dir, data_dir=data_dir)


def record_run(hub, model_name, config, path, args):
    """Move the submission into <out>/<reference_date>/ with a data snapshot and run.json."""
    out_dir, filename = os.path.split(path)
    reference_date = filename[:10]
    run_dir = os.path.join(out_dir, reference_date)
    if os.path.exists(run_dir):
        print(f'WARNING: replacing the earlier run recorded in {run_dir}')
        shutil.rmtree(run_dir)
    os.makedirs(os.path.join(run_dir, 'data'))
    final_path = os.path.join(run_dir, filename)
    shutil.move(path, final_path)
    for name in hub.data_files():
        shutil.copy2(os.path.join(hub.DATA_DIR, name), os.path.join(run_dir, 'data', name))
    commit, dirty = git_state()
    if dirty:
        print('WARNING: the working tree has uncommitted changes; run.json records the commit only')
    record = {
        'created_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'hub': hub.NAME,
        'model': model_name,
        'reference_date': reference_date,
        'submission': {'file': filename, 'sha256': sha256(final_path)},
        'git_commit': commit,
        'git_dirty': dirty,
        'config': config,
        'data_files': {name: sha256(os.path.join(run_dir, 'data', name)) for name in hub.data_files()},
        'argv': sys.argv[1:],
        'synced': not args.no_sync,
        'versions': versions(),
    }
    with open(os.path.join(run_dir, 'run.json'), 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2)
    print(f'Run recorded in {run_dir}')
    return final_path


def reproduce(hub, model, out_dir, reference_date):
    """Rerun a recorded week from its data snapshot and config; compare with its submission."""
    run_dir = os.path.join(out_dir, reference_date)
    with open(os.path.join(run_dir, 'run.json'), encoding='utf-8') as f:
        record = json.load(f)
    data_dir = os.path.join(run_dir, 'data')
    for name, digest in record['data_files'].items():
        if sha256(os.path.join(data_dir, name)) != digest:
            sys.exit(f'{name} in {data_dir} does not match the sha256 recorded in run.json')
    commit, dirty = git_state()
    if commit != record['git_commit'] or dirty:
        print(f"WARNING: code differs from the recorded run (recorded {record['git_commit']}, "
              f"now {commit}{' with uncommitted changes' if dirty else ''}); results may differ. "
              f"For an exact rerun: git switch --detach {record['git_commit']}")
    repro_dir = os.path.join(run_dir, 'reproduce')
    path = run_forecast(hub, model, record['config'], repro_dir, data_dir=data_dir)
    original_path = os.path.join(run_dir, record['submission']['file'])
    if original_path.endswith('.parquet'):   # sample hubs: no horizon column
        keys = ['target', 'target_end_date', 'location', 'output_type', 'output_type_id']
        original, rerun = pd.read_parquet(original_path), pd.read_parquet(path)
    else:
        keys = ['target', 'horizon', 'location', 'output_type', 'output_type_id']
        original = pd.read_csv(original_path, dtype={'location': str})
        rerun = pd.read_csv(path, dtype={'location': str})
    merged = original.merge(rerun, on=keys, how='outer', suffixes=('', '_rerun'), indicator=True)
    same_rows = (merged['_merge'] == 'both').all()
    max_diff = np.abs(merged['value'] - merged['value_rerun']).max() if same_rows else float('nan')
    print(f'Reproduced {reference_date}: same rows {same_rows}, max abs difference {max_diff}')
    return same_rows and max_diff == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hub', default='flusight', choices=list(HUBS))
    parser.add_argument('--model', default='pooled_hyb', choices=list(MODELS))
    parser.add_argument('--config', default=None, help='default: config/<hub>/<model>.json')
    parser.add_argument('--seeds', nargs='+', type=int, default=None, help="override the config's seeds")
    parser.add_argument('--out', default=None, help='default: outputs/<hub>/<model>/submissions')
    parser.add_argument('--no-sync', action='store_true', help='skip downloading the latest target data')
    parser.add_argument('--reproduce', metavar='REFERENCE_DATE', help='rerun a recorded week and compare')
    args = parser.parse_args()

    hub, model = HUBS[args.hub], MODELS[args.model]
    out_dir = args.out or os.path.join('outputs', args.hub, args.model, 'submissions')
    if args.reproduce:
        return reproduce(hub, model, out_dir, args.reproduce)

    with open(args.config or config_path(args.hub, args.model)) as f:
        config = json.load(f)
    if args.seeds:
        config['seeds'] = args.seeds
    if not args.no_sync:
        hub.sync()
    path = run_forecast(hub, model, config, out_dir)
    return record_run(hub, args.model, config, path, args)


if __name__ == '__main__':
    main()
