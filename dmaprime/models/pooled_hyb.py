"""Pooled HYB forecaster: one model per target trained on all locations, public hub data only.

One model per target is trained on every location at once. Each series is transformed to
z = log1p(value * scale) and the model predicts quantiles of the change z[t+1+h] - z[t]
for horizons h = 0..H-1, so the median starts from the last observed week.

Inputs per week of the 10-week window (all leakage-free for a given cutoff):
    target: z relative to the last week, weekly change, level standardized by location history
    aux (the hub's co-signal target, same location): same three + availability flag
    national target: z relative to the last week, weekly change
    sin/cos of the ISO week

Hub-specific settings (targets, quantile levels, horizons, national location) come from the
`hub` module passed in; see dmaprime/hubs/__init__.py.
"""
import warnings

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from dmaprime.data.panel import from_z
from dmaprime.models.base import Forecast
from dmaprime.models.nets import HybridCNNForecaster

OUTPUT_TYPE = 'quantile'
INPUT_WINDOW = 10

# Training configuration
HIDDEN_SIZE = 96
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
MAX_EPOCHS = 200
PATIENCE = 15
VAL_WEEKS = 8          # most recent anchor weeks held out for early stopping
INPUT_NOISE = 0.01
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _level(z, cutoff):
    """Standardize each location by its own history up to and including cutoff."""
    hist = z[:cutoff + 1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # locations without this target are all NaN
        mu = np.nanmean(hist, axis=0)
        sd = np.nanstd(hist, axis=0)
    return (z - mu) / np.where(np.isfinite(sd) & (sd > 1e-3), sd, 1.0)


def build_windows(hub, panel, target, cutoff, anchors):
    """Inputs (and targets where known) for every (anchor, location) with a complete window.

    Only data at or before `cutoff` is used for location statistics; inputs use data up to
    each anchor, and targets need anchor + horizons <= cutoff to be returned as labels.
    Returns X (N, W, F), Y (N, H) with NaN where unknown, and a frame of (anchor, location).
    """
    H = hub.HORIZONS
    z = panel.z[target]
    aux = panel.z[hub.TARGETS[target]['aux']]
    nat_col = panel.locations.index(hub.NATIONAL)
    lev, aux_lev = _level(z, cutoff), _level(aux, cutoff)
    dz = np.vstack([np.full((1, z.shape[1]), np.nan), np.diff(z, axis=0)])
    aux_dz = np.vstack([np.full((1, aux.shape[1]), np.nan), np.diff(aux, axis=0)])

    X, Y, keys = [], [], []
    W = INPUT_WINDOW
    for t in anchors:
        if t - W < 0:
            continue
        win = slice(t - W + 1, t + 1)
        z_win, dz_win = z[win], dz[win]
        nat = z[win, nat_col] - z[t, nat_col]
        nat_dz = dz[win, nat_col]
        if not (np.isfinite(nat).all() and np.isfinite(nat_dz).all()):
            continue
        future = z[t + 1:t + 1 + H] if t + H <= cutoff else None
        for l in range(z.shape[1]):
            if not (np.isfinite(z_win[:, l]).all() and np.isfinite(dz_win[:, l]).all()):
                continue
            a_rel = aux[win, l] - aux[t, l]
            aux_ok = np.isfinite(a_rel).all() and np.isfinite(aux_dz[win, l]).all()
            aux_feats = (np.stack([a_rel, aux_dz[win, l], aux_lev[win, l], np.ones(W)], axis=1)
                         if aux_ok else np.zeros((W, 4)))
            feats = np.column_stack([
                z_win[:, l] - z[t, l], dz_win[:, l], lev[win, l],
                aux_feats, nat, nat_dz, panel.season[win],
            ])
            y = (future[:, l] - z[t, l]) if future is not None and len(future) == H else np.full(H, np.nan)
            X.append(feats)
            Y.append(y)
            keys.append((t, panel.locations[l]))
    keys = pd.DataFrame(keys, columns=['anchor', 'location'])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), keys


def pinball_loss(preds, target, quantiles):
    """Mean pinball loss; preds (B, H, Q), target (B, H). Equals WIS / 2 for a symmetric quantile set."""
    errors = target.unsqueeze(2) - preds
    return torch.max((quantiles - 1) * errors, quantiles * errors).mean()


class PooledHybForecaster:
    """Feature scaler + HybridCNNForecaster trained on all locations for one target of a hub."""

    def __init__(self, hub, target, seed=0):
        self.hub = hub
        self.target = target
        self.seed = seed
        self.model = None
        self.mean = self.std = None

    def _scale(self, X):
        return (X - self.mean) / self.std

    def _new_model(self, input_size):
        hub = self.hub
        model = HybridCNNForecaster(input_size=input_size, hidden_size=HIDDEN_SIZE,
                                    output_steps=hub.HORIZONS, num_quantiles=len(hub.QUANTILES)).to(DEVICE)
        return model, optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    @staticmethod
    def _train_epoch(model, loader, optimizer, q):
        model.train()
        for X_batch, Y_batch in loader:
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
            X_batch = X_batch + torch.randn_like(X_batch) * INPUT_NOISE
            optimizer.zero_grad()
            loss = pinball_loss(model(X_batch), Y_batch, q)
            loss.backward()
            optimizer.step()

    def fit(self, panel, cutoff, verbose=False, refit=False, refit_exclude_weeks=0):
        """Train with early stopping on the most recent VAL_WEEKS anchor weeks.

        refit: afterwards retrain from scratch on all windows (including the validation weeks)
        for the best epoch count, replaying the learning-rate schedule of the first run.
        refit_exclude_weeks: in the refit, drop windows whose targets reach into the last N
        weeks before cutoff (recent real-time data is often still incomplete).
        """
        hub = self.hub
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        anchors = range(INPUT_WINDOW, cutoff - hub.HORIZONS + 1)
        X, Y, keys = build_windows(hub, panel, self.target, cutoff, anchors)
        ok = np.isfinite(Y).all(axis=1)
        X, Y, keys = X[ok], Y[ok], keys[ok]

        flat = X.reshape(-1, X.shape[2])
        self.mean, self.std = flat.mean(axis=0), flat.std(axis=0)
        self.std[self.std < 1e-6] = 1.0
        X = self._scale(X)

        # Time-based holdout: the most recent anchor weeks, across all locations
        val_mask = (keys['anchor'] > keys['anchor'].max() - VAL_WEEKS).to_numpy()
        train_ds = TensorDataset(torch.from_numpy(X[~val_mask]), torch.from_numpy(Y[~val_mask]))
        X_val = torch.from_numpy(X[val_mask]).to(DEVICE)
        Y_val = torch.from_numpy(Y[val_mask]).to(DEVICE)
        loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        q = torch.tensor(hub.QUANTILES, dtype=torch.float32, device=DEVICE)
        model, optimizer = self._new_model(X.shape[2])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best_loss, best_state, best_epoch, bad_epochs, lrs = float('inf'), None, 0, 0, []
        for epoch in range(MAX_EPOCHS):
            lrs.append(optimizer.param_groups[0]['lr'])
            self._train_epoch(model, loader, optimizer, q)
            model.eval()
            with torch.no_grad():
                val_loss = pinball_loss(model(X_val), Y_val, q).item()
            scheduler.step(val_loss)
            if val_loss < best_loss - 1e-6:
                best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= PATIENCE:
                    break
        model.load_state_dict(best_state)

        if refit:
            torch.manual_seed(self.seed)
            model, optimizer = self._new_model(X.shape[2])
            keep = (keys['anchor'] + hub.HORIZONS <= cutoff - refit_exclude_weeks).to_numpy()
            all_loader = DataLoader(TensorDataset(torch.from_numpy(X[keep]), torch.from_numpy(Y[keep])),
                                    batch_size=BATCH_SIZE, shuffle=True)
            for lr in lrs[:best_epoch + 1]:
                for group in optimizer.param_groups:
                    group['lr'] = lr
                self._train_epoch(model, all_loader, optimizer, q)
        self.model = model
        if verbose:
            refit_note = (f', refit on {int(keep.sum())} windows (excluding the last {refit_exclude_weeks} weeks) '
                          f'for {best_epoch + 1} epochs') if refit else ''
            print(f"{self.target} seed {self.seed}: {len(train_ds)} train / {int(val_mask.sum())} val windows, "
                  f"{epoch + 1} epochs, best val pinball {best_loss:.4f} at epoch {best_epoch + 1}{refit_note}")
        return self

    def predict_delta(self, panel, origin):
        """Quantiles of z change for every location at origin index: (keys, deltas (N, H, Q))."""
        X, _, keys = build_windows(self.hub, panel, self.target, origin, [origin])
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.from_numpy(self._scale(X)).to(DEVICE)).cpu().numpy()
        return keys, np.sort(out, axis=2)


def last_z(panel, target, origin, keys):
    cols = [panel.locations.index(l) for l in keys['location']]
    return panel.z[target][origin, cols]


def delta_values(hub, z_last, target, deltas, widen=None):
    """z-change quantiles (N, H, Q) + last observed z (N,) -> natural-scale quantiles.

    widen: array of length H with spread multipliers around the median, or None.
    """
    if widen is not None:
        med = deltas[:, :, hub.MEDIAN_IDX:hub.MEDIAN_IDX + 1]
        deltas = med + np.asarray(widen).reshape(1, -1, 1) * (deltas - med)
    return np.sort(from_z(np.asarray(z_last).reshape(-1, 1, 1) + deltas, hub.TARGETS[target]), axis=2)


def forecast(hub, panel, target, config, verbose=True):
    """Seed-ensembled, calibrated forecast from the latest week with data.

    config: {'seeds': [...], 'calibration': {target: [factor per horizon]},
             'refit': bool, 'refit_exclude_weeks': int (both optional)}.
    """
    origin = panel.latest_origin(target)
    deltas, keys = [], None
    for seed in config['seeds']:
        model = PooledHybForecaster(hub, target, seed=seed).fit(
            panel, origin, verbose=verbose, refit=config.get('refit', False),
            refit_exclude_weeks=config.get('refit_exclude_weeks', 0))
        k, d = model.predict_delta(panel, origin)
        if keys is not None and not k.equals(keys):
            raise RuntimeError(f'{target}: seeds produced different locations')
        keys = k
        deltas.append(d)
    widen = config['calibration'][target]
    values = delta_values(hub, last_z(panel, target, origin, keys), target, np.mean(deltas, axis=0), widen=widen)
    if verbose:
        print(f'{target}: data through {panel.dates[origin]:%Y-%m-%d}, {len(keys)} locations, widen {widen}')
    return Forecast(target=target, last_date=panel.dates[origin], locations=list(keys['location']),
                    output_type=OUTPUT_TYPE, output_type_ids=hub.QUANTILES, values=values)
