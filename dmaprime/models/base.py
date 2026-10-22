"""Common forecast container shared by all models and hubs.

A model module (dmaprime/models/<model_id>.py) defines:
    OUTPUT_TYPE   'quantile' or 'sample'
    forecast(hub, panel, target, config, verbose=True) -> Forecast
Hubs turn Forecasts into submission files; a hub that needs another output type than the
model provides must convert it (samples -> quantiles is direct; quantiles -> samples needs an
assumption about dependence across horizons).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Forecast:
    target: str
    last_date: pd.Timestamp   # last observed week the forecast starts from
    locations: list
    output_type: str          # 'quantile' or 'sample'
    output_type_ids: np.ndarray   # quantile levels, or sample indices
    values: np.ndarray        # (locations, horizons, output_type_ids), natural scale
