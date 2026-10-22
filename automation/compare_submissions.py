"""Compare two quantile submission files of the same week.

Training is deterministic on a given CPU, but different CPUs (GitHub's runners are AMD EPYC 7763,
AMD EPYC 9V45 or Intel Xeon 8370C) give slightly different results.
Differences are measured relative to the 95% interval width of each (target, horizon, location)
in the original. A rerun is accepted when it has exactly the same rows and, for each target, the
mean difference over all quantiles is at most --mean-tol and the largest median difference at
most --median-tol. On 2026-10-10, three CPUs differed pairwise by at most 0.028 (mean) and 0.048
(medians), while the forecast from the week before (wrong data) differed by 0.125-0.173 and
0.33-0.40. Exit code 0 if accepted, 1 if not.

    python -m automation.compare_submissions <original.csv> <rerun.csv> [--mean-tol 0.05 --median-tol 0.10]
"""
import argparse
import sys

import numpy as np
import pandas as pd

KEYS = ['target', 'horizon', 'location', 'output_type', 'output_type_id']


def compare(original, rerun, mean_tol, median_tol):
    a = pd.read_csv(original, dtype={'location': str})
    b = pd.read_csv(rerun, dtype={'location': str})
    m = a.merge(b, on=KEYS, how='outer', suffixes=('', '_rerun'), indicator=True)
    if not (m['_merge'] == 'both').all() or len(a) != len(b):
        print(f'rows differ: {len(a)} vs {len(b)}, {(m["_merge"] != "both").sum()} unmatched')
        return False
    diff = (m['value_rerun'] - m['value']).abs()
    if diff.max() == 0:
        print(f'identical: {len(a)} rows, max abs difference 0')
        return True
    q = m.pivot_table(index=['target', 'horizon', 'location'], columns='output_type_id', values='value')
    width = (q[0.975] - q[0.025]).rename('width')
    m = m.join(width, on=['target', 'horizon', 'location'])
    m['rel'] = diff / m['width'].where(m['width'] > 0, np.nan)
    m.loc[(diff == 0), 'rel'] = 0.0
    is_median = np.isclose(m['output_type_id'], 0.5)
    stats = pd.DataFrame({'mean': m.groupby('target')['rel'].mean(),
                          'max': m.groupby('target')['rel'].max(),
                          'median_max': m[is_median].groupby('target')['rel'].max()})
    print(f'{len(a)} rows, {(diff == 0).mean():.1%} identical; difference / 95% width by target '
          f'(accepted if mean <= {mean_tol} and median_max <= {median_tol}):')
    print(stats.round(4).to_string())
    return bool((stats['mean'] <= mean_tol).all() and (stats['median_max'] <= median_tol).all())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('original')
    parser.add_argument('rerun')
    parser.add_argument('--mean-tol', type=float, default=0.05)
    parser.add_argument('--median-tol', type=float, default=0.10)
    args = parser.parse_args()
    ok = compare(args.original, args.rerun, args.mean_tol, args.median_tol)
    print('ACCEPTED' if ok else 'NOT ACCEPTED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
