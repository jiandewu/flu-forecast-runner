"""Weekly Flu MetroCast run split for a GitHub Actions matrix (.github/workflows/weekly_metrocast.yml).

The six networks of DMAPRIME-ObsHYB (config seeds x early-stopping modes) are independent, so
each matrix job trains one and the results are combined afterwards; on one machine the result
is identical to `python -m dmaprime.forecast --hub metrocast --model pooled_obshyb` (the same
averaging code, model order and copula draws).

    prepare [--no-upload] [--force]   checks (automation.metrocast_submit.preflight) and data sync;
                                      writes the networks to train to PARTS_DIR/plan.json.
                                      Exit 0: go; 2: nothing to do (no new data, already run or
                                      submitted); 1: stopped by a problem.
    train --val MODE --seed N         trains one network on the synced data;
                                      writes PARTS_DIR/part_<MODE>_s<N>.npz (+ .json)
    combine [--no-upload] [--pr-dry-run] [--force]
                                      averages the parts, draws the samples, writes the submission
                                      and its run record (outputs/metrocast/pooled_obshyb/submissions/
                                      <reference_date>/, run.json lists the parts and their CPUs),
                                      checks the file, then syncs the fork, uploads and opens the PR

The same networks also forecast the nation's and every state's NSSP series, the FluSight ED target
(% ED visits for influenza): train saves them in the part as ed_* arrays (the MetroCast forecast
does not use them), and combine writes their seed average as quantiles to flusight_ed.csv in the
run record, which the FluSight weekly run (automation/flusight_ed.py) uses for its ED forecast.
Not written in the fallback (NSSP a week behind): FluSight then keeps its own ED forecast.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import traceback
import types

import numpy as np
import pandas as pd

from dmaprime import forecast
from dmaprime.data.panel import from_z
from dmaprime.hubs import metrocast as hub
from dmaprime.models import pooled_obshyb as po
from dmaprime.utils.env import load_env_file
from automation.metrocast_submit import (BASE, MODEL, Stop, check_file, log, model_id, preflight, upload,
                                      use_fork_settings)

PARTS_DIR = os.path.join(BASE, 'parts')
ED_FILE = 'flusight_ed.csv'


def load_config():
    with open(forecast.config_path(hub.NAME, MODEL)) as f:
        return json.load(f)


def plan(config):
    """(mode, seed) of every network, in the order fit_seeds trains and averages them."""
    s = po._settings(config)
    modes = s['val'] if isinstance(s['val'], list) else [s['val']]
    return [(mode, seed) for mode in modes for seed in config['seeds']]


def part_path(mode, seed):
    return os.path.join(PARTS_DIR, f'part_{mode}_s{seed}')


def ed_locations(panel):
    """{state slug ('us' for the nation): panel location} of the NSSP state and national series (the
    FluSight ED target). The NSSP series is preferred to a hub location of the same state (North
    Carolina's hub data are NC DPH's, not NSSP's)."""
    out = {l[len('nssp-'):]: l for l in panel.locations if l.startswith('nssp-') and not l[len('nssp-'):].isdigit()}
    locs = hub.hub_locations()
    for loc in panel.hub_locations:
        if loc in locs.index and locs.loc[loc, 'original_location_code'] == 'All':
            out.setdefault(hub.slug(locs.loc[loc, 'state']), loc)
    return out


def cpu():
    out = subprocess.run(['lscpu'], capture_output=True, text=True).stdout
    return next((l.split(':', 1)[1].strip() for l in out.splitlines() if l.startswith('Model name')), platform.processor())


def cmd_prepare(args):
    if args.force:
        hub.sync(verbose=False)
    else:
        preflight(args.no_upload)
    os.makedirs(PARTS_DIR, exist_ok=True)
    with open(os.path.join(PARTS_DIR, 'plan.json'), 'w') as f:
        json.dump([{'val': m, 'seed': s} for m, s in plan(load_config())], f)
    log(f'networks to train: {plan(load_config())}')


def cmd_train(args):
    import torch
    config = load_config()
    if (args.val, args.seed) not in plan(config):
        raise Stop(f'({args.val}, {args.seed}) is not in the config plan {plan(config)}')
    panel = hub.load_panel()
    origin = po.latest_origin(panel, hub.TARGET)
    po.align_to_nssp(panel, hub.TARGET, origin, po._settings(config)['lag_horizons'])
    settings = {**po._settings(config), 'val': args.val}
    model = po.PooledObshybForecaster(hub, hub.TARGET, seed=args.seed, settings=settings).fit(panel, origin, verbose=True)
    preds = po.model_predictions(model, panel, origin, locations=panel.hub_locations)
    arrays = {'origin': np.array(origin), 'corr': model.corr}
    for i, (lag, keys, deltas) in enumerate(preds):
        arrays[f'lag{i}'] = np.array(lag)
        arrays[f'locations{i}'] = keys['location'].to_numpy(dtype=str)
        arrays[f'deltas{i}'] = deltas
    ed = po.model_predictions(model, panel, origin, locations=list(ed_locations(panel).values()))
    for i, (lag, keys, deltas) in enumerate(ed):
        arrays[f'ed_lag{i}'] = np.array(lag)
        arrays[f'ed_locations{i}'] = keys['location'].to_numpy(dtype=str)
        arrays[f'ed_deltas{i}'] = deltas
    os.makedirs(PARTS_DIR, exist_ok=True)
    np.savez(part_path(args.val, args.seed) + '.npz', **arrays)
    with open(part_path(args.val, args.seed) + '.json', 'w') as f:
        json.dump({'val': args.val, 'seed': args.seed, 'data_through': f'{panel.dates[origin]:%Y-%m-%d}',
                   'cpu': cpu(), 'torch_threads': torch.get_num_threads(),
                   'torch_cpu_capability': torch.backends.cpu.get_cpu_capability()}, f)
    log(f'part ({args.val}, seed {args.seed}) saved')


def load_part(mode, seed):
    with np.load(part_path(mode, seed) + '.npz') as z:
        n = sum(1 for k in z.files if k.startswith('lag'))
        preds = [(int(z[f'lag{i}']), pd.DataFrame({'location': z[f'locations{i}'].astype(object)}), z[f'deltas{i}'])
                 for i in range(n)]
        return int(z['origin']), z['corr'], preds


def load_ed_part(mode, seed):
    """The part's predictions for the FluSight ED series (empty for parts trained before they were added)."""
    with np.load(part_path(mode, seed) + '.npz') as z:
        n = sum(1 for k in z.files if k.startswith('ed_lag'))
        return [(int(z[f'ed_lag{i}']), pd.DataFrame({'location': z[f'ed_locations{i}'].astype(object)}),
                 z[f'ed_deltas{i}']) for i in range(n)]


def write_ed(panel, origin, ed_preds, corrs, run_dir):
    """Seed-averaged quantiles of the FluSight ED series that are current (lag 0) to run_dir/ED_FILE,
    as proportions (the FluSight unit). Returns the number of locations written."""
    if not ed_preds or not all(ed_preds):
        return 0
    slug_of = {loc: s for s, loc in ed_locations(panel).items()}
    spec = hub.TARGETS[hub.TARGET]
    rows = []
    for o, keys, deltas, _ in po.combine_predictions(ed_preds, corrs, origin, hub.HORIZONS):
        if o != origin:
            continue   # a week behind: FluSight keeps its own forecast there
        cols = [panel.locations.index(l) for l in keys['location']]
        z_last = panel.z[hub.TARGET][o, cols]
        values = from_z(z_last[:, None, None] + np.sort(deltas, axis=2), spec) / 100.0
        for j, loc in enumerate(keys['location']):
            for h in range(hub.HORIZONS):
                rows.append(pd.DataFrame({
                    'state': slug_of[loc], 'panel_location': loc, 'last_date': f'{panel.dates[o]:%Y-%m-%d}',
                    'last_value': panel.values[hub.TARGET].iloc[o, cols[j]] / 100.0,
                    'horizon': h, 'quantile': hub.QUANTILES, 'value': values[j, h]}))
    if not rows:
        return 0
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(os.path.join(run_dir, ED_FILE), index=False, float_format='%.6g')
    return df['state'].nunique()


def cmd_combine(args, summary):
    config = load_config()
    panel = hub.load_panel()
    origin = po.latest_origin(panel, hub.TARGET)
    fallback = po.align_to_nssp(panel, hub.TARGET, origin, po._settings(config)['lag_horizons'])
    if fallback:
        summary.append(f'WARNING: NSSP data only through {panel.nssp_through:%Y-%m-%d} (hub through '
                       f'{panel.dates[origin]:%Y-%m-%d}); every location forecast from that week (fallback)')
    preds, corrs, parts, ed_preds = [], [], [], []
    for mode, seed in plan(config):
        o, corr, p = load_part(mode, seed)
        if o != origin:
            raise Stop(f'part ({mode}, {seed}) starts at {panel.dates[o]:%Y-%m-%d}, the data at {panel.dates[origin]:%Y-%m-%d}')
        preds.append(p)
        corrs.append(corr)
        ed_preds.append(load_ed_part(mode, seed))
        with open(part_path(mode, seed) + '.json') as f:
            parts.append(json.load(f))
    groups = po.combine_predictions(preds, corrs, origin, hub.HORIZONS)
    fc = po.groups_forecast(hub, panel, hub.TARGET, config, origin, groups)
    path = hub.write_submission([fc], config['model_id'], args.out)
    path = forecast.record_run(hub, MODEL, config, path, types.SimpleNamespace(no_sync=True))
    # FluSight ED: written before anything can stop the run, so FluSight gets it even if the upload fails
    n_ed = 0 if fallback else write_ed(panel, origin, ed_preds, corrs, os.path.dirname(path))
    summary.append(f'FluSight ED: {n_ed} locations in {ED_FILE}' if n_ed else
                   f'WARNING: no {ED_FILE} for FluSight' + (' (fallback)' if fallback else ''))
    run_json = os.path.join(os.path.dirname(path), 'run.json')
    with open(run_json) as f:
        record = json.load(f)
    record['parts'] = parts
    record['argv'] = ['automation.metrocast_weekly'] + sys.argv[1:]
    with open(run_json, 'w') as f:
        json.dump(record, f, indent=2)
    summary.append(f'File: {path}')
    summary.append('Parts: ' + ', '.join(f"{p['val']} s{p['seed']} ({p['cpu']})" for p in parts))
    check_file(path, window=not (args.force or args.no_upload), rounds=not args.no_upload)
    if args.no_upload:
        summary.append('Upload: none (--no-upload)')
        return
    upload(path, args.pr_dry_run, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--no-upload', action='store_true')
    p.add_argument('--force', action='store_true')
    p = sub.add_parser('train')
    p.add_argument('--val', required=True)
    p.add_argument('--seed', type=int, required=True)
    p = sub.add_parser('combine')
    p.add_argument('--no-upload', action='store_true')
    p.add_argument('--pr-dry-run', action='store_true')
    p.add_argument('--force', action='store_true')
    p.add_argument('--out', default=os.path.join(BASE, 'submissions'), help='run records (default %(default)s)')
    args = parser.parse_args()
    load_env_file('.env')
    use_fork_settings()

    summary = []
    log(f'===== metrocast_weekly {" ".join(sys.argv[1:])}')
    try:
        if args.cmd == 'prepare':
            cmd_prepare(args)
        elif args.cmd == 'train':
            cmd_train(args)
        else:
            cmd_combine(args, summary)
        status, code = 'OK', 0
        if any(line.startswith('WARNING') for line in summary):
            status = 'OK with warnings'
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
    log(f'===== {args.cmd}: {status}')
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as f:
            f.write(f'## metrocast {args.cmd}: {status}\n\n' + ''.join(f'- {line}\n' for line in summary))
    return code


if __name__ == '__main__':
    sys.exit(main())
