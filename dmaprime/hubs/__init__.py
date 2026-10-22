"""Forecast hubs. Each hub module defines its targets, data sources and submission format:

    NAME, DATA_DIR, TARGETS, OUTPUT_TYPE, HORIZONS, NATIONAL
    QUANTILES, MEDIAN_IDX                  quantile levels the models are trained on
    N_SAMPLES                              sample hubs: trajectories per location
    sync(verbose=True)                     download the latest target data into DATA_DIR
    data_files()                           local data file names, snapshotted with every run
    load_panel(frames=None, data_dir=None) -> dmaprime.data.panel.Panel
    write_submission(forecasts, model_id, out_dir, data_dir=None) -> path of the submission file
"""
from dmaprime.hubs import flusight, metrocast

HUBS = {flusight.NAME: flusight, metrocast.NAME: metrocast}
