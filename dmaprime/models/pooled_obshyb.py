"""Pooled observation-aware HYB (ObsHYB): one network for every metro area, HSA and state, with
sample trajectories from a Gaussian copula.

Built for hubs with sample output (Flu MetroCast). The hub panel holds the hub's locations
plus every NSSP state and HSA in the country; all of them train one network.

Marginals: like pooled HYB, the network predicts quantiles of z[t+1+h] - z[t] for h = 0..H-1
(z = log1p(value * scale)), trained with pinball loss, seeds averaged, spread calibrated per
horizon.

Inputs per week of the window:
    own series: z relative to the last week, weekly change, level standardized by location history
    parent series (HSA -> its state, state -> nation): z relative to its last week, weekly change,
        availability flag
    nation: same three
    reliability flags of the own series (config flag_inputs): low coverage, anomalous change
    sin/cos of the ISO week

Early stopping (config val): 'recent' validates on the most recent VAL_WEEKS anchor weeks,
'last_season' on the previous season's October-May windows (held out from training), which
are also the windows the copula correlation is estimated on. A list of modes trains every
seed once per mode and averages the forecasts (and copula correlations).

Observation-aware training (config flag_train): 'drop' leaves out training windows whose
label weeks are flagged (low coverage or anomalous); 'none' keeps them.

Trajectories: the dependence across horizons is a Gaussian copula whose correlation matrix is
estimated from the network's own validation errors: each validation label's PIT under the
predicted quantiles, mapped to normal scores, correlated across horizons. For each location,
N correlated normal vectors are drawn; at each horizon the N values are the quantiles at levels
(i + 0.5) / N assigned in the rank order of the draws, so every horizon's marginal is the
forecast distribution exactly and the ranks carry the dependence.
"""
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from dmaprime.data.panel import from_z
from dmaprime.models.base import Forecast
from dmaprime.models.nets import HybridCNNForecaster
from dmaprime.models.pooled_hyb import _level, pinball_loss

OUTPUT_TYPE = 'sample'
INPUT_WINDOW = 10

# Training configuration (same network and optimizer settings as pooled HYB)
HIDDEN_SIZE = 96
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
MAX_EPOCHS = 200
PATIENCE = 15
VAL_WEEKS = 8
INPUT_NOISE = 0.01
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULTS = {'flag_inputs': True, 'flag_train': 'drop', 'batch_size': BATCH_SIZE,
            'copula': 'gaussian', 'sample_seed': 0, 'lag_horizons': 1, 'val': 'recent'}


def _settings(config):
    return {**DEFAULTS, **{k: config[k] for k in DEFAULTS if k in config}}


def build_windows(hub, panel, target, cutoff, anchors, flag_inputs=True, n_out=None, chunk=64):
    """Inputs (and labels where known) for every (anchor, location) with a complete own window.

    Only data at or before `cutoff` is used for location statistics; inputs use data up to each
    anchor, and labels need anchor + horizon <= cutoff. Returns X (N, W, F), Y (N, n_out) with NaN
    where unknown, label flags (N,) (any known label week flagged), and a frame of (anchor, location).
    """
    H, W = n_out or hub.HORIZONS, INPUT_WINDOW
    z = panel.z[target]
    T, L = z.shape
    anchors = np.array([t for t in anchors if t - W + 1 >= 1 and t < T], dtype=int)
    lev = _level(z, cutoff)
    dz = np.vstack([np.full((1, L), np.nan), np.diff(z, axis=0)])
    par = np.array([panel.locations.index(panel.parent[l]) for l in panel.locations])
    nat = np.full(L, panel.locations.index(hub.NATIONAL))
    flagged = panel.flags['low_coverage'] | panel.flags['anomalous']
    names = np.asarray(panel.locations)

    Xs, Ys, Fs, ks = [], [], [], []
    for start in range(0, len(anchors), chunk):
        a_idx = anchors[start:start + chunk]
        win_idx = a_idx[:, None] + np.arange(-W + 1, 1)[None, :]       # (A, W)
        own_rel = z[win_idx] - z[a_idx][:, None, :]                       # (A, W, L)
        own_dz, own_lev = dz[win_idx], lev[win_idx]
        ok = np.isfinite(own_rel).all(axis=1) & np.isfinite(own_dz).all(axis=1) & np.isfinite(own_lev).all(axis=1)
        if not ok.any():
            continue

        def other(cols):
            rel, d = own_rel[:, :, cols], own_dz[:, :, cols]
            good = np.isfinite(rel).all(axis=1) & np.isfinite(d).all(axis=1)  # (A, L)
            g = good[:, None, :]
            return np.where(g, rel, 0.0), np.where(g, d, 0.0), np.broadcast_to(g, rel.shape).astype(float)

        feats = [own_rel, own_dz, own_lev, *other(par), *other(nat)]
        if flag_inputs:
            feats += [panel.flags['low_coverage'][win_idx].astype(float),
                      panel.flags['anomalous'][win_idx].astype(float)]
        season = np.broadcast_to(panel.season[win_idx][:, :, None, :], own_rel.shape + (2,))
        X = np.concatenate([np.stack(feats, axis=-1), season], axis=-1)  # (A, W, L, F)

        Y = np.full((len(a_idx), L, H), np.nan)
        label_flag = np.zeros((len(a_idx), L), dtype=bool)
        for h in range(H):
            tt = a_idx + 1 + h
            known = tt <= min(cutoff, T - 1)
            if not known.any():
                continue
            Y[known, :, h] = z[tt[known]] - z[a_idx[known]]
            label_flag[known] |= flagged[tt[known]]
        a, l = np.nonzero(ok)
        Xs.append(X[a, :, l].astype(np.float32))
        Ys.append(Y[a, l].astype(np.float32))
        Fs.append(label_flag[a, l])
        ks.append(pd.DataFrame({'anchor': a_idx[a], 'location': names[l]}))
    if not Xs:
        F = 11 + 2 * flag_inputs
        return (np.zeros((0, W, F), np.float32), np.zeros((0, H), np.float32), np.zeros(0, bool),
                pd.DataFrame(columns=['anchor', 'location']))
    return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Fs), pd.concat(ks, ignore_index=True)


def validation_mask(panel, cutoff, keys, mode):
    """Windows held out for early stopping and the copula correlation.

    'recent': the most recent VAL_WEEKS anchor weeks; 'last_season': the anchors of the
    previous season's October-May (falls back to 'recent' when there is none)."""
    if mode == 'last_season':
        date = panel.dates[cutoff]
        season = date.year if date.month >= 8 else date.year - 1
        anchor_dates = panel.dates[keys['anchor'].to_numpy()]
        mask = np.asarray((anchor_dates >= pd.Timestamp(season - 1, 10, 1)) & (anchor_dates <= pd.Timestamp(season, 5, 31)))
        if mask.any():
            return mask
    elif mode != 'recent':
        raise ValueError(f'unknown val mode {mode}')
    return (keys['anchor'] > keys['anchor'].max() - VAL_WEEKS).to_numpy()


def masked_pinball_loss(preds, target, quantiles):
    """pinball_loss over the known labels only (NaN labels are ignored)."""
    mask = torch.isfinite(target)
    if mask.all():
        return pinball_loss(preds, target, quantiles)
    errors = torch.where(mask, target, 0.0).unsqueeze(2) - preds
    loss = torch.max((quantiles - 1) * errors, quantiles * errors)
    return (loss * mask.unsqueeze(2)).sum() / (mask.sum() * len(quantiles))


class PooledObshybForecaster:
    """Feature scaler + HybridCNNForecaster trained on all panel locations for one target."""

    def __init__(self, hub, target, seed=0, settings=None):
        self.hub = hub
        self.target = target
        self.seed = seed
        self.settings = settings or dict(DEFAULTS)
        self.model = None
        self.mean = self.std = None
        self.n_out = hub.HORIZONS + self.settings['lag_horizons']
        self.corr = None   # (n_out, n_out) normal-score correlation of validation errors

    def _scale(self, X):
        return (X - self.mean) / self.std

    def _new_model(self, input_size):
        hub = self.hub
        model = HybridCNNForecaster(input_size=input_size, hidden_size=HIDDEN_SIZE,
                                    output_steps=self.n_out, num_quantiles=len(hub.QUANTILES)).to(DEVICE)
        return model, optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    @staticmethod
    def _train_epoch(model, loader, optimizer, q):
        model.train()
        for X_batch, Y_batch in loader:
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
            X_batch = X_batch + torch.randn_like(X_batch) * INPUT_NOISE
            optimizer.zero_grad()
            loss = masked_pinball_loss(model(X_batch), Y_batch, q)
            loss.backward()
            optimizer.step()

    def fit(self, panel, cutoff, verbose=False, max_epochs=None):
        """Train with early stopping on the most recent VAL_WEEKS anchor weeks (all locations),
        then estimate the copula correlation on the same validation windows."""
        hub, s = self.hub, self.settings
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        anchors = range(INPUT_WINDOW, cutoff - hub.HORIZONS + 1)
        X, Y, label_flag, keys = build_windows(hub, panel, self.target, cutoff, anchors, s['flag_inputs'],
                                               n_out=self.n_out)
        ok = np.isfinite(Y[:, :hub.HORIZONS]).all(axis=1)   # extra lag horizons may be unknown
        if s['flag_train'] == 'drop':
            ok &= ~label_flag
        X, Y, keys = X[ok], Y[ok], keys[ok].reset_index(drop=True)

        flat = X.reshape(-1, X.shape[2])
        self.mean, self.std = flat.mean(axis=0), flat.std(axis=0)
        self.std[self.std < 1e-6] = 1.0
        X = self._scale(X)

        val_mask = validation_mask(panel, cutoff, keys, s['val'])
        train_ds = TensorDataset(torch.from_numpy(X[~val_mask]), torch.from_numpy(Y[~val_mask]))
        X_val = torch.from_numpy(X[val_mask]).to(DEVICE)
        Y_val = torch.from_numpy(Y[val_mask]).to(DEVICE)
        loader = DataLoader(train_ds, batch_size=s['batch_size'], shuffle=True)

        q = torch.tensor(hub.QUANTILES, dtype=torch.float32, device=DEVICE)
        model, optimizer = self._new_model(X.shape[2])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best_loss, best_state, best_epoch, bad_epochs = float('inf'), None, 0, 0
        for epoch in range(max_epochs or MAX_EPOCHS):
            self._train_epoch(model, loader, optimizer, q)
            model.eval()
            with torch.no_grad():
                val_loss = masked_pinball_loss(model(X_val), Y_val, q).item()
            scheduler.step(val_loss)
            if val_loss < best_loss - 1e-6:
                best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= PATIENCE:
                    break
        model.load_state_dict(best_state)
        self.model = model
        model.eval()
        with torch.no_grad():
            val_pred = np.sort(model(X_val).cpu().numpy(), axis=2)
        y_val = Y[val_mask]
        full = np.isfinite(y_val).all(axis=1)
        self.corr = error_correlation(val_pred[full], y_val[full], hub.QUANTILES)
        if verbose:
            print(f"{self.target} seed {self.seed}: {len(train_ds)} train / {int(val_mask.sum())} val windows, "
                  f"{epoch + 1} epochs, best val pinball {best_loss:.4f} at epoch {best_epoch + 1}, "
                  f"copula corr h0-h1 {self.corr[0, 1]:.2f} h0-h3 {self.corr[0, -1]:.2f}")
        return self

    def predict_delta(self, panel, origin):
        """Quantiles of z change for every location at origin index: (keys, deltas (N, n_out, Q))."""
        X, _, _, keys = build_windows(self.hub, panel, self.target, origin, [origin], self.settings['flag_inputs'],
                                      n_out=self.n_out)
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.from_numpy(self._scale(X)).to(DEVICE)).cpu().numpy()
        return keys, np.sort(out, axis=2)


def quantile_function(levels, values, p):
    """Piecewise linear interpolation of quantiles on the normal-score scale, linear
    extrapolation beyond the outer levels. values (..., Q), p (P,) -> (..., P)."""
    from scipy.special import ndtri   # here, not at import: FluSight runs import this module without scipy
    x = ndtri(np.asarray(levels))
    xp = ndtri(np.asarray(p))
    out = np.empty(values.shape[:-1] + (len(xp),))
    flat_v = values.reshape(-1, values.shape[-1])
    flat_o = out.reshape(-1, len(xp))
    lo_slope = (flat_v[:, 1] - flat_v[:, 0]) / (x[1] - x[0])
    hi_slope = (flat_v[:, -1] - flat_v[:, -2]) / (x[-1] - x[-2])
    for i in range(len(flat_v)):
        flat_o[i] = np.interp(xp, x, flat_v[i])
    below, above = xp < x[0], xp > x[-1]
    flat_o[:, below] = flat_v[:, :1] + lo_slope[:, None] * (xp[below] - x[0])
    flat_o[:, above] = flat_v[:, -1:] + hi_slope[:, None] * (xp[above] - x[-1])
    return out


def pit(levels, values, y):
    """PIT of y under the quantile function of values: inverse of quantile_function.
    values (N, H, Q), y (N, H) -> (N, H) in (0, 1)."""
    from scipy.special import ndtr, ndtri
    x = ndtri(np.asarray(levels))
    u = np.empty(y.shape)
    for n in range(y.shape[0]):
        for h in range(y.shape[1]):
            v = values[n, h]
            if y[n, h] < v[0]:
                s = (v[1] - v[0]) / (x[1] - x[0])
                score = x[0] + (y[n, h] - v[0]) / s if s > 0 else x[0]
            elif y[n, h] > v[-1]:
                s = (v[-1] - v[-2]) / (x[-1] - x[-2])
                score = x[-1] + (y[n, h] - v[-1]) / s if s > 0 else x[-1]
            else:
                score = np.interp(y[n, h], v, x)
            u[n, h] = ndtr(score)
    return np.clip(u, 1e-4, 1 - 1e-4)


def error_correlation(pred, y, levels):
    """Correlation across horizons of the normal scores of validation PITs."""
    from scipy.special import ndtri
    scores = ndtri(pit(levels, pred, y))
    corr = np.corrcoef(scores, rowvar=False)
    return np.clip(corr, -1, 1)


def copula_samples(levels, deltas, corr, n_samples, rng):
    """Sample trajectories: deltas (N, H, Q) quantiles -> (N, H, n_samples).

    Each horizon's values are the quantiles at (i + 0.5) / n, ordered by the ranks of draws
    from a normal distribution with correlation corr across horizons."""
    N, H, _ = deltas.shape
    p = (np.arange(n_samples) + 0.5) / n_samples
    marg = quantile_function(levels, deltas, p)                          # (N, H, n), sorted
    chol = np.linalg.cholesky(corr + 1e-9 * np.eye(H))
    out = np.empty((N, H, n_samples))
    for n in range(N):
        draws = rng.standard_normal((n_samples, H)) @ chol.T            # (n, H)
        ranks = np.argsort(np.argsort(draws, axis=0), axis=0)            # rank of each draw per horizon
        for h in range(H):
            out[n, h] = marg[n, h, ranks[:, h]]
    return out


def nearest_corr(c):
    """Project a symmetric matrix onto the correlation matrices (clip eigenvalues)."""
    w, v = np.linalg.eigh((c + c.T) / 2)
    c = (v * np.clip(w, 1e-6, None)) @ v.T
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


def fit_seeds(hub, panel, target, config, origin, verbose=True, max_epochs=None):
    """One trained PooledObshybForecaster per seed in config['seeds'] and early-stopping mode
    in config['val'] (a mode or a list of modes; their forecasts are averaged)."""
    s = _settings(config)
    modes = s['val'] if isinstance(s['val'], list) else [s['val']]
    return [PooledObshybForecaster(hub, target, seed=seed, settings={**s, 'val': mode}).fit(
        panel, origin, verbose=verbose, max_epochs=max_epochs) for mode in modes for seed in config['seeds']]


def model_predictions(model, panel, origin, locations=None):
    """One network's predictions at origin, by how late each location's data are.

    Locations with data at origin are forecast from origin with outputs 0..H-1; locations
    whose latest data are `lag` weeks older (lag <= lag_horizons, e.g. NYC) from their own
    latest week with outputs lag..lag+H-1, so both cover the same target weeks.
    locations: limit to these locations (default: all with a complete window).
    Returns a list of (lag, keys, deltas (N, H, Q)).
    """
    H = model.hub.HORIZONS
    preds, done = [], set()
    for lag in range(model.settings['lag_horizons'] + 1):
        k, d = model.predict_delta(panel, origin - lag)
        keep = ~k['location'].isin(done)
        if locations is not None:
            keep &= k['location'].isin(locations)
        k, d = k[keep].reset_index(drop=True), d[keep.to_numpy()]
        if len(k):
            preds.append((lag, k, d[:, lag:lag + H]))
            done |= set(k['location'])
    return preds


def combine_predictions(preds, corrs, origin, horizons):
    """Average the networks' predictions (list of model_predictions outputs, in a fixed order)
    and copula correlations (list of (n_out, n_out)).
    Returns a list of (origin index, keys, mean deltas (N, H, Q), copula correlation (H, H))."""
    corr_all = nearest_corr(np.mean(corrs, axis=0))
    groups = []
    for i, (lag, keys, _) in enumerate(preds[0]):
        for p in preds[1:]:
            if p[i][0] != lag or not p[i][1].equals(keys):
                raise RuntimeError('networks produced different locations')
        deltas = np.mean([p[i][2] for p in preds], axis=0)
        groups.append((origin - lag, keys, deltas,
                       nearest_corr(corr_all[lag:lag + horizons, lag:lag + horizons])))
    return groups


def predict_groups(models, panel, origin, locations=None):
    """Seed-averaged predictions at origin, grouped by how late each location's data are
    (see model_predictions). Returns a list of (origin index, keys, mean deltas (N, H, Q),
    copula correlation (H, H))."""
    return combine_predictions([model_predictions(m, panel, origin, locations) for m in models],
                               [m.corr for m in models], origin, models[0].hub.HORIZONS)


def calibrate(hub, deltas, widen):
    if widen is None:
        return deltas
    med = deltas[:, :, hub.MEDIAN_IDX:hub.MEDIAN_IDX + 1]
    return med + np.asarray(widen).reshape(1, -1, 1) * (deltas - med)


def to_samples(hub, panel, target, origin, keys, deltas, corr, config):
    """Calibrated z-change quantiles -> natural-scale sample trajectories (N, H, S)."""
    s = _settings(config)
    deltas = calibrate(hub, deltas, config.get('calibration', {}).get(target))
    rng = np.random.default_rng(s['sample_seed'])
    if s['copula'] == 'comonotone':   # same quantile level at every horizon
        smp = quantile_function(hub.QUANTILES, deltas, (np.arange(hub.N_SAMPLES) + 0.5) / hub.N_SAMPLES)
    else:
        corr = np.eye(len(corr)) if s['copula'] == 'independent' else corr
        smp = copula_samples(hub.QUANTILES, deltas, corr, hub.N_SAMPLES, rng)
    cols = [panel.locations.index(l) for l in keys['location']]
    z_last = panel.z[target][origin, cols]
    return from_z(z_last.reshape(-1, 1, 1) + smp, hub.TARGETS[target])


def latest_origin(panel, target):
    """Index of the latest week with data for any of the hub's locations."""
    hub_cols = [panel.locations.index(l) for l in panel.hub_locations]
    origin = len(panel.dates) - 1
    while not np.isfinite(panel.z[target][origin, hub_cols]).any():
        origin -= 1
    return origin


def align_to_nssp(panel, target, origin, max_lag=None):
    """When the NSSP data (panel.nssp_through, the national series) end before the hub's data, hide
    the hub's weeks after it, so that every location is forecast from NSSP's last week with the extra
    lag outputs (as NYC is every week) and all inputs stay consistent. Changes panel.z[target] in place.
    Returns the lag in weeks (0: nothing changed); raises ValueError past max_lag (default: the lag
    outputs of DEFAULTS)."""
    through = getattr(panel, 'nssp_through', None)
    if through is None or pd.isna(through) or through >= panel.dates[origin]:
        return 0
    t = panel.date_index(through)
    lag = origin - t
    if lag > (max_lag if max_lag is not None else DEFAULTS['lag_horizons']):
        raise ValueError(f'NSSP data end {lag} weeks before the hub data ({through:%Y-%m-%d})')
    cols = [panel.locations.index(l) for l in panel.hub_locations]
    z = panel.z[target].copy()
    z[t + 1:, cols] = np.nan
    panel.z[target] = z
    return lag


def groups_forecast(hub, panel, target, config, origin, groups, verbose=True):
    """Sample trajectories for every group of predict_groups / combine_predictions -> Forecast."""
    locations, values = [], []
    for o, keys, deltas, corr in groups:
        values.append(to_samples(hub, panel, target, o, keys, deltas, corr, config))
        locations += list(keys['location'])
        if verbose:
            print(f'{target}: {len(keys)} locations from {panel.dates[o]:%Y-%m-%d}, copula corr\n{np.round(corr, 2)}')
    return Forecast(target=target, last_date=panel.dates[origin], locations=locations,
                    output_type=OUTPUT_TYPE, output_type_ids=np.arange(hub.N_SAMPLES),
                    values=np.concatenate(values))


def forecast(hub, panel, target, config, verbose=True):
    """Seed-ensembled, calibrated sample forecast from the latest week of the hub's locations.

    config: {'seeds': [...], 'calibration': {target: [factor per horizon]}, plus the
    optional settings in DEFAULTS}.
    """
    origin = latest_origin(panel, target)
    lag = align_to_nssp(panel, target, origin, _settings(config)['lag_horizons'])
    if lag and verbose:
        print(f'WARNING {target}: NSSP data only through {panel.nssp_through:%Y-%m-%d}; every location is '
              f'forecast from that week (fallback)')
    models = fit_seeds(hub, panel, target, config, origin, verbose=verbose)
    groups = predict_groups(models, panel, origin, locations=panel.hub_locations)
    return groups_forecast(hub, panel, target, config, origin, groups, verbose)
