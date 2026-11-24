"""Numbers for lecture 23b (graph networks), from notebook 4.5 (mlgeo_4.5_GNN.ipynb).

Code copied from the notebook, section by section, with the same seeds and settings, so the
numbers equal the notebook's printed outputs (training times excepted: they are wall-clock
seconds and change from run to run). Writes figs/lec23b_gnn/numbers.json.

Data: mlgeo_synth.event_station_table(n_events=800, n_stations=50, n_clusters=20, seed=0),
synthetic peak ground acceleration (PGA) with a known attenuation relation, event terms and
station terms. Task: predict log10 PGA at the stations of an earthquake that have no
recording (20-30% of the network) from the stations that recorded it.

Run from the repository root:
    pixi run python book/slides/2026/figscripts/lec23b_gnn_numbers.py
Options (for exploration only; the defaults are the notebook's settings):
    --layers N      message-passing and graph-convolution layers (notebook: 2)
    --out PATH      where to write the JSON
    --subnetworks   the optional exercise of notebook section 12: graph networks trained on a random
                    sub-network at every batch (up to 10 of the 40 stations removed); writes
                    figs/lec23b_gnn/numbers_subnetworks.json unless --out is given
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import mlgeo_synth

T0 = time.time()
ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--out", default=None)
ap.add_argument("--subnetworks", action="store_true")
args = ap.parse_args()
FIGS = Path(__file__).resolve().parents[1] / "figs" / "lec23b_gnn"
args.out = args.out or str(FIGS / ("numbers_subnetworks.json" if args.subnetworks else "numbers.json"))

# ---- setup (notebook section 0) -------------------------------------------------------
torch.set_num_threads(1)  # tiny models: one thread is fastest, and sums are always added in the same order
N_LAYERS = args.layers
SEEDS = [0, 1, 2]
out = {"source": "book/slides/2026/figscripts/lec23b_gnn_numbers.py, copy of book/Chapter4-DeepLearning/mlgeo_4.5_GNN.ipynb",
       "torch": torch.__version__, "threads": torch.get_num_threads(), "subnetworks": args.subnetworks}

# ---- 1. data (notebook section 1) -----------------------------------------------------
df, truth = mlgeo_synth.event_station_table(n_events=800, n_stations=50, n_clusters=20, seed=0)
Y = df.pivot(index="event_id", columns="station_id", values="log10_pga").to_numpy()
dist = df.pivot(index="event_id", columns="station_id", values="dist_km").to_numpy()
xy = df.groupby("station_id")[["station_x_km", "station_y_km"]].first().to_numpy()
events = df.groupby("event_id")[["cluster_id", "time_days", "magnitude", "event_x_km", "event_y_km"]].first()

c = truth["coeffs"]
median = (c["a_mag"] * events["magnitude"].to_numpy()[:, None]
          + c["b_dist"] * np.log10(dist + c["c_km"]) + c["intercept"])
event_term = truth["event_terms"].to_numpy()[:, None]
station_term = truth["station_terms"].to_numpy()
station_plus_noise = Y - (median + event_term)
noise_only = Y - (median + event_term + station_term[None, :])

out["data"] = dict(
    n_events=int(Y.shape[0]), n_stations=int(Y.shape[1]), n_clusters=int(events["cluster_id"].nunique()),
    log10_pga_min=float(Y.min()), log10_pga_max=float(Y.max()),
    magnitude_min=float(events["magnitude"].min()), magnitude_max=float(events["magnitude"].max()),
    coeffs=c,
    station_term_std=float(station_term.std(ddof=1)),
    station_term_min=float(station_term.min()), station_term_max=float(station_term.max()),
    event_term_std=float(truth["event_terms"].std(ddof=1)),
)

# ---- 2. task, split, new stations, masks (notebook section 2) --------------------------
cluster_time = events.groupby("cluster_id")["time_days"].median().sort_values()
order = cluster_time.index.to_numpy()
clusters = {"train": order[:14], "validation": order[14:17], "test": order[17:]}
ev = {k: np.flatnonzero(events["cluster_id"].isin(v).to_numpy()) for k, v in clusters.items()}

rng = np.random.default_rng(100)
new = np.sort(rng.choice(50, 10, replace=False))
old = np.setdiff1d(np.arange(50), new)


def draw_masks(rng, n_samples, n_stations, lo=0.2, hi=0.3):
    n_masked = rng.integers(int(np.ceil(lo * n_stations)), int(np.floor(hi * n_stations)) + 1, n_samples)
    rank = rng.random((n_samples, n_stations)).argsort(1).argsort(1)
    return rank < n_masked[:, None]


mu, sd = float(Y[ev["train"]][:, old].mean()), float(Y[ev["train"]][:, old].std())
z = (Y - mu) / sd


def node_inputs(z, mask):
    return torch.tensor(np.stack([np.where(mask, 0.0, z), mask.astype(float)], -1), dtype=torch.float32)


rng_eval = np.random.default_rng(7)
va = np.repeat(ev["validation"], 4)
te = np.repeat(ev["test"], 4)
m_va = draw_masks(rng_eval, len(va), 40)
m_te = draw_masks(rng_eval, len(te), 40)
X_va, X_te = node_inputs(z[va][:, old], m_va), node_inputs(z[te][:, old], m_te)
y_te = Y[te][:, old]

out["split"] = dict(
    clusters={k: int(len(v)) for k, v in clusters.items()},
    events={k: int(len(v)) for k, v in ev.items()},
    samples=dict(train_per_epoch=int(len(ev["train"])), validation=int(len(va)), test=int(len(te))),
    masks_per_event_validation_test=4, masked_fraction=[0.2, 0.3], masked_per_sample_40=[8, 12],
    n_masked_test_values=int(m_te.sum()),
    training_network_stations=int(len(old)), new_stations=new.tolist(),
    standardization=dict(mu=mu, sd=sd),
)


# ---- 3. baselines and floors (notebook section 3) --------------------------------------
def mae(pred, y, mask):
    return float(np.abs(pred - y)[mask].mean())


def mean_of_recorded(y, mask):
    m = np.where(mask, 0.0, y).sum(1) / (~mask).sum(1)
    return np.repeat(m[:, None], y.shape[1], 1)


def idw(y, mask, xy, power=2.0):
    D = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    pred = np.empty_like(y)
    for s in range(len(y)):
        w = np.where(mask[s][None, :], 0.0, 1.0 / np.maximum(D, 1e-9) ** power)
        np.fill_diagonal(w, 0.0)
        pred[s] = w @ np.where(mask[s], 0.0, y[s]) / w.sum(1)
    return pred


st_true = station_term[old]
rf_te = station_plus_noise[te][:, old]


def site_term_estimate(pred):
    part = np.where(m_te, pred - y_te + rf_te, np.nan)
    return np.nanmean(part, 0)


def site_term_r(pred):
    return float(np.corrcoef(site_term_estimate(pred), st_true)[0, 1])


near = m_te & (dist[te][:, old] < 40.0)

baselines = {
    "mean of recorded stations": mean_of_recorded(y_te, m_te),
    "inverse-distance weighting": idw(y_te, m_te, xy[old]),
}
floors = {
    "floor: station term + noise left": np.abs(rf_te)[m_te].mean(),
    "floor: noise only left": np.abs(noise_only[te][:, old])[m_te].mean(),
}
out["baselines"] = {k: dict(mae=mae(p, y_te, m_te), factor=10 ** mae(p, y_te, m_te),
                            mae_within_40km=mae(p, y_te, near), site_term_r=site_term_r(p))
                    for k, p in baselines.items()}
out["floors"] = {k: dict(mae=float(v), factor=float(10 ** v)) for k, v in floors.items()}
out["floors_within_40km"] = {
    "floor: station term + noise left": float(np.abs(rf_te)[near].mean()),
    "floor: noise only left": float(np.abs(noise_only[te][:, old])[near].mean())}
out["split"]["n_masked_test_within_40km"] = int(near.sum())


# ---- 4. training loop (notebook section 4) ---------------------------------------------
def n_weights(model):
    return sum(p.numel() for p in model.parameters())


def predict(model, X, g):
    model.eval()
    with torch.no_grad():
        return model(X, g).numpy() * sd + mu


def masked_mse(pred, obs, mask):
    return ((pred - obs) ** 2 * mask).sum() / mask.sum()


z_tr = torch.tensor(z[ev["train"]][:, old], dtype=torch.float32)
z_va, mva_t = torch.tensor(z[va][:, old], dtype=torch.float32), torch.tensor(m_va, dtype=torch.float32)


def train(model, g, seed, max_epochs=200, patience=20, lr=1e-3, batch_size=32):
    rng = np.random.default_rng(seed + 1000)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(z_tr)
    best, best_state, bad, history = np.inf, None, 0, {"train": [], "validation": []}
    t0 = time.time()
    for epoch in range(max_epochs):
        mask = draw_masks(rng, n, 40)
        X, M = node_inputs(z_tr.numpy(), mask), torch.tensor(mask, dtype=torch.float32)
        order = rng.permutation(n)
        model.train()
        total = 0.0
        for j in range(0, n, batch_size):
            idx = order[j:j + batch_size]
            if args.subnetworks and g is not None:   # notebook section 12, exercise solution
                n_drop = rng.integers(0, 11)
                keep = np.sort(rng.choice(40, 40 - n_drop, replace=False))
                loss = masked_mse(model(X[idx][:, keep], build_graph(xy[old][keep])),
                                  z_tr[idx][:, keep], M[idx][:, keep])
            else:
                loss = masked_mse(model(X[idx], g), z_tr[idx], M[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_loss = masked_mse(model(X_va, g), z_va, mva_t).item()
        history["train"].append(total / n)
        history["validation"].append(val_loss)
        if val_loss < best - 1e-5:
            best, bad = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad == patience:
                break
    model.load_state_dict(best_state)
    return dict(time_s=time.time() - t0, epochs=epoch + 1,
                best_epoch=int(np.argmin(history["validation"])) + 1, history=history)


results = {}


def run(name, make_model, g):
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = make_model()
        info = train(model, g, seed)
        pred = predict(model, X_te, g)
        results.setdefault(name, []).append(dict(seed=seed, model=model, pred=pred, weights=n_weights(model), **info,
                                                 mae=mae(pred, y_te, m_te)))
        print(f"{name}, seed {seed}: test MAE {results[name][-1]['mae']:.3f} after {info['epochs']} epochs "
              f"({info['time_s']:.0f} s)", flush=True)


# ---- 5. fully connected (notebook section 5) -------------------------------------------
class FullyConnected(torch.nn.Module):
    def __init__(self, n_stations=40, hidden=128):
        super().__init__()
        self.n_stations = n_stations
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2 * n_stations, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, n_stations))

    def forward(self, x, g=None):
        if x.shape[1] != self.n_stations:
            raise ValueError(f"built for {self.n_stations} stations, got {x.shape[1]}")
        return self.net(x.flatten(1))


run("fully connected", FullyConnected, None)


# ---- 6. the graph (notebook section 6) -------------------------------------------------
# Your turn (1): five stations, two nearest neighbors each
yt1_xy = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [30.0, 5.0], [40.0, 0.0]])
yt1_D = np.hypot(yt1_xy[:, None, 0] - yt1_xy[None, :, 0], yt1_xy[:, None, 1] - yt1_xy[None, :, 1])
yt1_nn = np.argsort(yt1_D + np.diag(np.full(5, np.inf)), axis=1)[:, :2]
A5 = np.zeros((5, 5))
for i in range(5):
    A5[i, yt1_nn[i]] = 1
A5 = np.maximum(A5, A5.T)
A5_tilde = A5 + np.eye(5)
deg5 = A5_tilde.sum(1)
A5_hat = A5_tilde / np.sqrt(deg5[:, None] * deg5[None, :])
out["your_turn_1"] = dict(xy_km=yt1_xy.tolist(), nearest_two=yt1_nn.tolist(), A=A5.astype(int).tolist(),
                          degree_with_self=deg5.tolist(), A_hat=np.round(A5_hat, 3).tolist())


def build_graph(xy, k=6, scale_km=50.0):
    n = len(xy)
    D = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    nn = np.argsort(D + np.diag(np.full(n, np.inf)), axis=1)[:, :k]
    A = np.zeros((n, n))
    A[np.repeat(np.arange(n), k), nn.ravel()] = 1
    A = np.maximum(A, A.T)
    A_tilde = A + np.eye(n)
    deg = A_tilde.sum(1)
    A_hat = A_tilde / np.sqrt(deg[:, None] * deg[None, :])
    dst, src = np.nonzero(A)
    edge = np.stack([D[dst, src], xy[src, 0] - xy[dst, 0], xy[src, 1] - xy[dst, 1]], 1) / scale_km
    S = np.zeros((n, len(src)))
    S[dst, np.arange(len(src))] = 1.0 / A.sum(1)[dst]
    f = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    return dict(A=A, A_hat=f(A_hat), src=torch.from_numpy(src), edge=f(edge), S=f(S),
                n_edges=len(src), degree=A.sum(1))


g_old = build_graph(xy[old])
out["graph"] = dict(k=6, n_nodes=40, n_edges_directed=int(g_old["n_edges"]),
                    n_station_pairs=int(g_old["n_edges"] // 2),
                    degree_mean=float(g_old["degree"].mean()), degree_min=int(g_old["degree"].min()),
                    degree_max=int(g_old["degree"].max()),
                    neighbor_distance_km_median=float(np.median(
                        np.hypot(*(xy[old][g_old["src"].numpy()] - xy[old][np.nonzero(g_old["A"])[0]]).T))),
                    edge_feature_scale_km=50.0)


# ---- 7. graph convolution (notebook section 7) -----------------------------------------
class GraphConv(torch.nn.Module):
    def __init__(self, n_layers=N_LAYERS, hidden=32):
        super().__init__()
        dims = [2] + [hidden] * n_layers
        self.layers = torch.nn.ModuleList(torch.nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x, g):
        h = x
        for layer in self.layers:
            h = torch.relu(g["A_hat"] @ layer(h))
        return self.head(h).squeeze(-1)


run("graph convolution", GraphConv, g_old)


# ---- 8. message passing (notebook section 8) -------------------------------------------
# Your turn (2): one update for one station, by hand
yt2_h_i = np.array([0.0, 1.0])
yt2_h_nb = np.array([[0.8, 0.0], [0.2, 0.0], [-0.6, 0.0]])
yt2_e = np.array([[0.4, 0.4, 0.0], [0.2, 0.0, -0.2], [0.2, -0.12, 0.16]])
W_m = np.array([[1.0, 0.0, -0.5, 0.0, 0.0], [-1.0, 0.0, -0.5, 0.0, 0.0]])
W_u = np.array([[0.0, 0.0, 1.0, -1.0], [0.0, 1.0, 0.0, 0.0]])
yt2_messages = np.maximum(0, np.concatenate([yt2_h_nb, yt2_e], axis=1) @ W_m.T)
yt2_mean = yt2_messages.mean(axis=0)
yt2_h_new = np.maximum(0, W_u @ np.concatenate([yt2_h_i, yt2_mean]))
out["your_turn_2"] = dict(h_i=yt2_h_i.tolist(), h_neighbors=yt2_h_nb.tolist(), edge_d_dx_dy_per_50km=yt2_e.tolist(),
                          W_m=W_m.tolist(), W_u=W_u.tolist(), messages=np.round(yt2_messages, 3).tolist(),
                          mean=np.round(yt2_mean, 3).tolist(), h_new=np.round(yt2_h_new, 3).tolist())


class MessagePassingLayer(torch.nn.Module):
    def __init__(self, d_in, hidden):
        super().__init__()
        self.message = torch.nn.Linear(d_in + 3, hidden)
        self.update = torch.nn.Linear(d_in + hidden, hidden)

    def forward(self, h, g):
        e = g["edge"].expand(h.shape[0], -1, -1)
        m = torch.relu(self.message(torch.cat([h[:, g["src"], :], e], -1)))
        mean = g["S"] @ m
        return torch.relu(self.update(torch.cat([h, mean], -1)))


class MessagePassing(torch.nn.Module):
    def __init__(self, n_layers=N_LAYERS, hidden=32):
        super().__init__()
        dims = [2] + [hidden] * n_layers
        self.layers = torch.nn.ModuleList(MessagePassingLayer(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x, g):
        h = x
        for layer in self.layers:
            h = layer(h, g)
        return self.head(h).squeeze(-1)


run("message passing", MessagePassing, g_old)

# ---- 9. leaderboard (notebook section 9) -----------------------------------------------
rows = []
for name, p in baselines.items():
    rows.append(dict(model=name, mae=mae(p, y_te, m_te), std=np.nan, near=mae(p, y_te, near),
                     r=site_term_r(p), weights=0, time_s=0.0, epochs=np.nan))
for name, runs in results.items():
    rows.append(dict(model=name, mae=np.mean([r["mae"] for r in runs]),
                     std=np.std([r["mae"] for r in runs], ddof=1),
                     near=np.mean([mae(r["pred"], y_te, near) for r in runs]),
                     r=np.mean([site_term_r(r["pred"]) for r in runs]),
                     weights=runs[0]["weights"], time_s=np.mean([r["time_s"] for r in runs]),
                     epochs=np.mean([r["epochs"] for r in runs])))
for name, v in floors.items():
    rows.append(dict(model=name, mae=float(v), std=np.nan, near=out["floors_within_40km"][name], r=np.nan,
                     weights=np.nan, time_s=np.nan, epochs=np.nan))
board = pd.DataFrame(rows).sort_values("mae").set_index("model")
board["factor"] = 10 ** board["mae"]
print(board.round(3).to_string())

out["leaderboard"] = {}
for name, runs in results.items():
    out["leaderboard"][name] = dict(
        mae_mean=float(np.mean([r["mae"] for r in runs])), mae_std=float(np.std([r["mae"] for r in runs], ddof=1)),
        factor=float(10 ** np.mean([r["mae"] for r in runs])),
        mae_within_40km_mean=float(np.mean([mae(r["pred"], y_te, near) for r in runs])),
        site_term_r_mean=float(np.mean([site_term_r(r["pred"]) for r in runs])),
        site_term_slope_seed0=float(np.polyfit(st_true, site_term_estimate(runs[0]["pred"]), 1)[0]),
        weights=int(runs[0]["weights"]), train_time_s_mean=float(np.mean([r["time_s"] for r in runs])),
        per_seed=[dict(seed=r["seed"], mae=r["mae"], site_term_r=site_term_r(r["pred"]), epochs=r["epochs"],
                       best_epoch=r["best_epoch"], train_time_s=r["time_s"]) for r in runs])
out["layers_graph_models"] = N_LAYERS
# per-station terms learned by each method (networks: seed 0), for the scatter of lecture 23b
# (figscripts/lec23b_gnn_figs.py); same estimate as the site-term r column
out["site_terms_seed0"] = dict(
    true=np.round(st_true, 4).tolist(),
    **{name: np.round(site_term_estimate(p), 4).tolist() for name, p in baselines.items()},
    **{name: np.round(site_term_estimate(runs[0]["pred"]), 4).tolist() for name, runs in results.items()})

# ---- 10. test 1: station order shuffled (notebook section 10) ---------------------------
perm = np.random.default_rng(99).permutation(40)
X_perm = X_te[:, perm, :]
g_perm = build_graph(xy[old][perm])
t1 = {}
for name, runs in results.items():
    g = g_perm if name != "fully connected" else None
    changes = [float(np.abs(predict(r["model"], X_perm, g) - r["pred"][:, perm]).max()) for r in runs]
    maes = [mae(predict(r["model"], X_perm, g), y_te[:, perm], m_te[:, perm]) for r in runs]
    t1[name] = dict(mae_original=float(np.mean([r["mae"] for r in runs])), mae_shuffled=float(np.mean(maes)),
                    mae_shuffled_std=float(np.std(maes, ddof=1)), largest_output_change=float(max(changes)))
out["test1_station_order"] = t1
print(pd.DataFrame(t1).T.to_string())

# ---- 11. test 2: ten new stations (notebook section 11) --------------------------------
g_all = build_graph(xy)
m_all = np.zeros((len(te), 50), dtype=bool)
m_all[:, old] = m_te
m_all[:, new] = True
X_all, y_all = node_inputs(z[te], m_all), Y[te]
at_new = np.zeros_like(m_all)
at_new[:, new] = True
at_old = np.zeros_like(m_all)
at_old[:, old] = m_te

t2 = {
    "floor: station term + noise left": dict(new=float(np.abs(station_plus_noise[te])[at_new].mean())),
    "floor: noise only left": dict(new=float(np.abs(noise_only[te])[at_new].mean())),
    "mean of recorded stations": dict(new=mae(mean_of_recorded(y_all, m_all), y_all, at_new),
                                      old=mae(mean_of_recorded(y_all, m_all), y_all, at_old)),
    "inverse-distance weighting": dict(new=mae(idw(y_all, m_all, xy), y_all, at_new),
                                       old=mae(idw(y_all, m_all, xy), y_all, at_old)),
}
site_new = {}
for name in ("graph convolution", "message passing"):
    preds = [predict(r["model"], X_all, g_all) for r in results[name]]
    new_maes = [mae(p, y_all, at_new) for p in preds]
    old_maes = [mae(p, y_all, at_old) for p in preds]
    t2[name] = dict(new=float(np.mean(new_maes)), new_std=float(np.std(new_maes, ddof=1)),
                    old=float(np.mean(old_maes)), old_std=float(np.std(old_maes, ddof=1)),
                    old_before_growth=out["leaderboard"][name]["mae_mean"])
    applied = (preds[0] - y_all + station_plus_noise[te])[:, new].mean(0)
    site_new[name] = dict(applied_site_term_seed0=np.round(applied, 3).tolist(),
                          std_applied=float(applied.std(ddof=1)),
                          r_with_true=float(np.corrcoef(applied, station_term[new])[0, 1]))
t2["fully connected"] = "cannot run: its first layer expects 40 stations"
out["test2_new_stations"] = t2
out["test2_site_terms"] = dict(true_new=np.round(station_term[new], 3).tolist(),
                               rms_true_new=float(np.sqrt((station_term[new] ** 2).mean())),
                               rms_true_old=float(np.sqrt((station_term[old] ** 2).mean())), **site_new)
out["test2_masked_old_values"] = int(at_old.sum())
out["test2_new_values"] = int(at_new.sum())
print(json.dumps(t2, indent=1))

out["total_wall_time_s"] = time.time() - T0
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(out, indent=1, default=float))
print(f"wrote {args.out} in {out['total_wall_time_s']:.0f} s")
