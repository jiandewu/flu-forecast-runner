"""Forecasting models, keyed by model id (see dmaprime/models/base.py for the interface)."""
from dmaprime.models import pooled_hyb, pooled_obshyb

MODELS = {'pooled_hyb': pooled_hyb, 'pooled_obshyb': pooled_obshyb}
