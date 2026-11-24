"""Seed spread of the four sequence models in book/Chapter4-DeepLearning/mlgeo_4.4_RNN.ipynb.

Code copied from the notebook (cells 1, 3, 6, 9, 15, 21, 29, 33, 35). Data seed fixed
(gnss_series seed=42). Only the torch seed varies. As in the notebook, the four models are
trained in sequence (RNN, LSTM, Attention, Transformer) after ONE torch.manual_seed call,
so each model's init and shuffling depend on the RNG state the previous model left behind.
Seed 42 is the notebook's own seed.

Written to figs/lec23_rnn/seeds_4.4.json for the leaderboard of lecture 23.
Run from repo root:  pixi run python book/slides/2026/figscripts/lec23_seeds.py [seeds...]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import mlgeo_synth

SEEDS = [int(s) for s in sys.argv[1:]] or [42, 43, 44]
device = torch.device("cpu")

# ---- data (cells 3, 6) -------------------------------------------------------------
eq_day = 1800
gnss = mlgeo_synth.gnss_series(n_years=10.0, eq_day=eq_day, seed=42)

WINDOW, HORIZON = 90, 30
series = gnss["disp_mm"].to_numpy(dtype=np.float32)
n_days = len(series)
split = int(0.8 * n_days)
mu = float(series[:split].mean())
sigma = float(series[:split].std())
z = (series - mu) / sigma
print(f"training-portion mean {mu:.1f} mm, std {sigma:.1f} mm")


def make_windows(z, start, stop, window=WINDOW, horizon=HORIZON):
    X, Y = [], []
    for i in range(start, stop - window - horizon + 1):
        X.append(z[i : i + window])
        Y.append(z[i + window : i + window + horizon])
    X = np.stack(X)[..., np.newaxis]
    Y = np.stack(Y)
    anchor = X[:, -1:, 0].copy()
    X = X - anchor[:, :, np.newaxis]
    Y = Y - anchor
    return (torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(anchor))


X_train, Y_train, anc_train = make_windows(z, 0, split)
X_val, Y_val, anc_val = make_windows(z, split, n_days)
print("train:", tuple(X_train.shape), "val:", tuple(X_val.shape))


# ---- training helpers (cell 9) -----------------------------------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters())


def val_mae_mm(model):
    model.eval()
    with torch.no_grad():
        pred = model(X_val.to(device)).cpu()
    return (pred - Y_val).abs().mean().item() * sigma


def train_model(model, name, n_epochs=20, lr=1e-3, batch_size=64):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    n = X_train.shape[0]
    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        for j in range(0, n, batch_size):
            idx = perm[j : j + batch_size]
            xb, yb = X_train[idx].to(device), Y_train[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        # notebook evaluates val MSE every epoch; no RNG use, kept for identical timing
        model.eval()
        with torch.no_grad():
            loss_fn(model(X_val.to(device)), Y_val.to(device)).item()
    elapsed = time.time() - t0
    mae = val_mae_mm(model)
    print(f"[{name}] {count_params(model):,} params, {elapsed:.1f} s, val MAE {mae:.4f} mm",
          flush=True)
    return mae, elapsed, count_params(model)


# ---- models (cells 15, 21, 29, 33) -------------------------------------------------
class VanillaRNN(torch.nn.Module):
    def __init__(self, hidden_size=32, horizon=HORIZON):
        super().__init__()
        self.rnn = torch.nn.RNN(1, hidden_size, batch_first=True)
        self.head = torch.nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class LSTMForecaster(torch.nn.Module):
    def __init__(self, hidden_size=32, horizon=HORIZON):
        super().__init__()
        self.lstm = torch.nn.LSTM(1, hidden_size, batch_first=True)
        self.head = torch.nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class TinyAttention(torch.nn.Module):
    def __init__(self, d_model=32, horizon=HORIZON):
        super().__init__()
        self.embed = torch.nn.Linear(1, d_model)
        self.Wq = torch.nn.Linear(d_model, d_model, bias=False)
        self.Wk = torch.nn.Linear(d_model, d_model, bias=False)
        self.Wv = torch.nn.Linear(d_model, d_model, bias=False)
        self.head = torch.nn.Linear(d_model, horizon)
        self.scale = d_model ** 0.5

    def forward(self, x):
        h = self.embed(x)
        Q, K, V = self.Wq(h), self.Wk(h), self.Wv(h)
        scores = Q @ K.transpose(1, 2) / self.scale
        weights = torch.softmax(scores, dim=-1)
        context = weights @ V
        return self.head(context.mean(dim=1))


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1).float()
        freq = torch.exp(torch.arange(0, d_model, 2).float()
                         * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * freq)
        pe[:, 1::2] = torch.cos(pos * freq)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.shape[1]]


class TinyTransformer(torch.nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=2, horizon=HORIZON):
        super().__init__()
        self.embed = torch.nn.Linear(1, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64,
            dropout=0.0, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False)
        self.head = torch.nn.Linear(d_model, horizon)

    def forward(self, x):
        h = self.encoder(self.pos(self.embed(x)))
        return self.head(h.mean(dim=1))


MODELS = [("RNN", VanillaRNN), ("LSTM", LSTMForecaster),
          ("Attention", TinyAttention), ("Transformer", TinyTransformer)]

# ---- baselines (cell 35) -----------------------------------------------------------
mae_persistence = Y_val.abs().mean().item() * sigma
starts = np.arange(split, n_days - WINDOW - HORIZON + 1)
seas_pred = np.stack([z[i + WINDOW - 365 : i + WINDOW + HORIZON - 365] for i in starts])
true_z = Y_val.numpy() + anc_val.numpy()
mae_seasonal = float(np.abs(seas_pred - true_z).mean() * sigma)
print(f"persistence {mae_persistence:.4f} mm, seasonal naive {mae_seasonal:.4f} mm")

# ---- seed loop ---------------------------------------------------------------------
runs = {name: {"val_mae_mm": [], "train_time_s": [], "n_params": None} for name, _ in MODELS}
for seed in SEEDS:
    print(f"=== seed {seed} ===", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    for name, cls in MODELS:
        mae, t, npar = train_model(cls(), name)
        runs[name]["val_mae_mm"].append(mae)
        runs[name]["train_time_s"].append(t)
        runs[name]["n_params"] = npar

out = {
    "notebook": "book/Chapter4-DeepLearning/mlgeo_4.4_RNN.ipynb",
    "seeds": SEEDS,
    "seed_protocol": ("torch.manual_seed(seed) once, then RNN, LSTM, Attention, Transformer "
                      "trained in sequence (same as notebook); data seed fixed at 42"),
    "std_type": "sample (ddof=1)",
    "torch_version": torch.__version__,
    "torch_num_threads": torch.get_num_threads(),
    "baselines": {"persistence": mae_persistence, "seasonal_naive": mae_seasonal},
    "models": {},
}
for name, r in runs.items():
    v = np.array(r["val_mae_mm"])
    out["models"][name] = {
        "val_mae_mm": [round(x, 4) for x in v.tolist()],
        "mean": round(float(v.mean()), 4),
        "std": round(float(v.std(ddof=1)), 4) if len(v) > 1 else None,
        "n_params": r["n_params"],
        "mean_train_time_s": round(float(np.mean(r["train_time_s"])), 2),
    }
path = Path(__file__).resolve().parent.parent / "figs" / "lec23_rnn" / "seeds_4.4.json"
path.write_text(json.dumps(out, indent=2))
print(json.dumps(out["models"], indent=2))
print("wrote", path)
