"""CDC COVID-19 Forecast Hub (CDCgov/covid19-forecast-hub): targets and data sources.

Targets: weekly confirmed COVID-19 hospital admissions (NHSN) and the weekly proportion of ED
visits due to COVID-19 (NSSP). Data handling and the submission format are shared with the RSV
hub: see dmaprime/hubs/cdc_resp.py.
"""
import os
import sys

from dmaprime.hubs import cdc_resp as _resp

NAME = 'covid'
DATA_DIR = os.path.join('data', NAME)
REMOTE_BASE_URL = 'https://raw.githubusercontent.com/CDCgov/covid19-forecast-hub/main'

OUTPUT_TYPE = _resp.OUTPUT_TYPE
QUANTILES = _resp.QUANTILES
MEDIAN_IDX = _resp.MEDIAN_IDX
HORIZONS = _resp.HORIZONS
NATIONAL = _resp.NATIONAL

# file: target CSV written by sync(); aux: co-signal target; scale: value multiplier before log1p;
# upper: maximum valid value; exclude: target_end_date ranges left out (NHSN voluntary reporting)
TARGETS = {
    'wk inc covid hosp': {
        'file': 'target-hospital-admissions.csv',
        'aux': 'wk inc covid prop ed visits',
        'scale': 1.0,
        'upper': None,
        'exclude': [('2024-05-01', '2024-10-31')],
    },
    'wk inc covid prop ed visits': {
        'file': 'target-ed-visits-prop.csv',
        'aux': 'wk inc covid hosp',
        'scale': 100.0,   # proportion -> percent before log1p
        'upper': 1.0,
    },
}

_HUB = sys.modules[__name__]


def sync(verbose=True):
    return _resp.sync(_HUB, verbose)


def data_files():
    return _resp.data_files(_HUB)


def load_panel(frames=None, data_dir=None):
    return _resp.load_panel(_HUB, frames, data_dir)


def releases(data_dir=None):
    return _resp.releases(_HUB, data_dir)


def populations(data_dir=None):
    return _resp.populations(_HUB, data_dir)


def write_submission(forecasts, model_id, out_dir, data_dir=None):
    return _resp.write_submission(_HUB, forecasts, model_id, out_dir)
