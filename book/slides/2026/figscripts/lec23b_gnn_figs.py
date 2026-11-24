"""Figures for lecture 23b (graph networks), tied to notebook 4.5 (mlgeo_4.5_GNN.ipynb).

Sources of the numbers on the figures:
- figs/lec23b_gnn/numbers.json, written by figscripts/lec23b_gnn_numbers.py (a section-by-section copy of
  the notebook's code, seeds 0, 1, 2): the Your-turn values, the graph statistics, test 1 (station
  order), the per-station terms learned by each method (seed 0) and their correlations.
- Recomputed here with the notebook's own code, no training: the synthetic data
  (mlgeo_synth.event_station_table(n_events=800, n_stations=50, n_clusters=20, seed=0)), the split by
  sequence, the 10 stations installed later (default_rng(100)), the test masks (default_rng(7)), the
  6-nearest-neighbor graph, the stations reached after 1, 2 and 3 layers, and the weight counts of the
  three networks (the notebook's classes, instantiated). Each recomputed number that numbers.json also
  holds is checked against it (assert). The new numbers are written to figs/lec23b_gnn/figs_numbers.json.
- Drawn, no data: the network anatomy, the three kinds of graph (grid, all-to-all, stations) and the
  three heads.

Figures (figs/lec23b_gnn/):
 1. task_map.png          the region, epicenters by split, 40 + 10 stations; the graph and one test earthquake
 2. graph_family.png      a grid (convolution), all-to-all (attention), a station network (graph network)
 3. gnn_anatomy.png       node features -> two message-passing layers -> one head per station
 4. adjacency_by_hand.png Your turn (1): five stations, A and the normalized adjacency
 5. message_by_hand.png   Your turn (2): three messages, their mean, the update
 6. hops.png              one layer = one hop: the stations one station hears from after 1, 2, 3 layers
 7. weights_count.png     weights of the three networks, for 40 and 50 stations
 8. station_order.png     test 1: test MAE with the stations in the original and in a shuffled order
 9. heads.png             node-level, graph-level and edge-level heads
10. site_terms.png        station terms learned against the true ones (IDW and the three networks, seed 0)

Regenerate (from the repository root, a few seconds):
    pixi run python book/slides/2026/figscripts/lec23b_gnn_figs.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

import mlgeo_synth  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / "figs" / "lec23b_gnn"
FIGS.mkdir(parents=True, exist_ok=True)
N = json.loads((FIGS / "numbers.json").read_text())
plt.rcParams.update({"font.size": 20, "axes.titlesize": 22, "axes.titleweight": "bold"})
INK, MUTED, LIGHT = "#26241f", "#6e675c", "#e8e4dc"
GRAPH, FC_C, GCN_C, IDW_C, OUT_C, RED = "#CC79A7", "#0072B2", "#E69F00", "#8a857c", "#009E73", "#D55E00"
SPLIT_C = {"train": "#BBBBBB", "validation": "#E69F00", "test": "#D55E00"}
TAG = "Synthetic network (mlgeo_synth)"
out = {}


def save(fig, name):
    fig.savefig(FIGS / name, dpi=110, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def arrow(ax, p, q, color=INK, lw=2.5, ms=22, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=4))


def box(ax, x, y, w, h, text, fc, fs=20, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.25", fc=fc, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, weight="bold")


def tri(ax, x, y, fc, ec=INK, s=200, lw=1.2, z=3, **kw):
    ax.scatter([x], [y], marker="^", s=s, color=fc, edgecolor=ec, lw=lw, zorder=z, **kw)


# ------------------------------------------------------------------ data, split, masks, graph (notebook sections 1, 2, 6)
df, truth = mlgeo_synth.event_station_table(n_events=800, n_stations=50, n_clusters=20, seed=0)
Y = df.pivot(index="event_id", columns="station_id", values="log10_pga").to_numpy()
xy = df.groupby("station_id")[["station_x_km", "station_y_km"]].first().to_numpy()
events = df.groupby("event_id")[["cluster_id", "time_days", "magnitude", "event_x_km", "event_y_km"]].first()
order = events.groupby("cluster_id")["time_days"].median().sort_values().index.to_numpy()
clusters = {"train": order[:14], "validation": order[14:17], "test": order[17:]}
ev = {k: np.flatnonzero(events["cluster_id"].isin(v).to_numpy()) for k, v in clusters.items()}
new = np.sort(np.random.default_rng(100).choice(50, 10, replace=False))
old = np.setdiff1d(np.arange(50), new)
assert new.tolist() == N["split"]["new_stations"]
assert {k: len(v) for k, v in ev.items()} == N["split"]["events"]


def draw_masks(rng, n_samples, n_stations, lo=0.2, hi=0.3):
    n_masked = rng.integers(int(np.ceil(lo * n_stations)), int(np.floor(hi * n_stations)) + 1, n_samples)
    rank = rng.random((n_samples, n_stations)).argsort(1).argsort(1)
    return rank < n_masked[:, None]


rng_eval = np.random.default_rng(7)
va, te = np.repeat(ev["validation"], 4), np.repeat(ev["test"], 4)
m_va, m_te = draw_masks(rng_eval, len(va), 40), draw_masks(rng_eval, len(te), 40)
assert int(m_te.sum()) == N["split"]["n_masked_test_values"]
y_te = Y[te][:, old]


def knn_adjacency(xy, k=6):
    n = len(xy)
    D = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    nn = np.argsort(D + np.diag(np.full(n, np.inf)), axis=1)[:, :k]
    A = np.zeros((n, n))
    A[np.repeat(np.arange(n), k), nn.ravel()] = 1
    return np.maximum(A, A.T), D


A, D = knn_adjacency(xy[old])
assert int(A.sum()) == N["graph"]["n_edges_directed"]
pts = xy[old]

# the test earthquake of the notebook's graph figure (cell 27): the largest test earthquake, its first mask
e_show = ev["test"][np.argmax(events["magnitude"].to_numpy()[ev["test"]])]
s_show = np.flatnonzero(te == e_show)[0]
hid = m_te[s_show]
mag = float(events["magnitude"].iloc[e_show])
out["map"] = dict(event_index=int(e_show), magnitude=round(mag, 2), n_hidden=int(hid.sum()),
                  log10_pga_range=[round(float(y_te[s_show].min()), 2), round(float(y_te[s_show].max()), 2)])

# ------------------------------------------------------------------ 1. task map
fig, axes = plt.subplots(1, 2, figsize=(17, 8.2), gridspec_kw=dict(wspace=0.32))
ax = axes[0]
for k in ("train", "validation", "test"):
    idx = ev[k]
    ax.scatter(events["event_x_km"].iloc[idx], events["event_y_km"].iloc[idx], s=10, color=SPLIT_C[k],
               label=f"{k} ({len(idx)})", zorder=1)
ax.scatter(*xy[old].T, marker="^", s=150, color=INK, zorder=3, label="40 stations")
ax.scatter(*xy[new].T, marker="^", s=150, facecolor="white", edgecolor=FC_C, lw=2, zorder=3,
           label="10 installed later")
ax.set_title("800 earthquakes, 20 sequences", fontsize=21)
ax.legend(fontsize=16, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False,
          markerscale=1.6, handletextpad=0.2, columnspacing=0.8)
ax = axes[1]
for i, j in zip(*np.nonzero(np.triu(A))):
    ax.plot(*pts[[i, j]].T, color=GRAPH, lw=1.3, alpha=0.7, zorder=1)
sc = ax.scatter(*pts[~hid].T, c=y_te[s_show, ~hid], cmap="viridis", vmin=y_te[s_show].min(),
                vmax=y_te[s_show].max(), marker="^", s=230, edgecolor="k", zorder=3, label="recorded")
ax.scatter(*pts[hid].T, marker="^", s=230, facecolor="white", edgecolor=RED, lw=2.5, zorder=3,
           label=f"hidden, to predict ({hid.sum()})")
ax.scatter(events["event_x_km"].iloc[e_show], events["event_y_km"].iloc[e_show], marker="*", s=600,
           color=RED, edgecolor="k", zorder=4, label="epicenter")
ax.plot([], [], color=GRAPH, lw=2, label="edge: 6 nearest")
cax = ax.inset_axes([1.03, 0.05, 0.035, 0.9])
cb = fig.colorbar(sc, cax=cax)
cb.set_label("log10 PGA (g)", fontsize=17)
cb.ax.tick_params(labelsize=15)
ax.set_title(f"The graph; one test earthquake, M{mag:.1f}", fontsize=21)
ax.legend(fontsize=16, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False,
          handletextpad=0.2, columnspacing=0.8)
for ax in axes:
    ax.set_aspect("equal")
    ax.set_xlim(-15, 215); ax.set_ylim(-15, 215)
    ax.set_xlabel("east (km)", fontsize=17)
    ax.tick_params(labelsize=15)
axes[0].set_ylabel("north (km)", fontsize=17)
fig.text(0.5, 0.97, TAG + ": a region of 200 km × 200 km", ha="center", fontsize=17, color=MUTED)
save(fig, "task_map.png")

# ------------------------------------------------------------------ 2. three kinds of graph
fig, axes = plt.subplots(1, 3, figsize=(17, 6.6))
ax = axes[0]
g = np.arange(5)
for a in g:
    for b in g:
        if a < 4:
            ax.plot([a, a + 1], [b, b], color=GCN_C, lw=2.5, zorder=1)
        if b < 4:
            ax.plot([a, a], [b, b + 1], color=GCN_C, lw=2.5, zorder=1)
        ax.add_patch(Rectangle((a - 0.28, b - 0.28), 0.56, 0.56, fc="#fbe3b0", ec=INK, lw=1.2, zorder=2))
ax.add_patch(Rectangle((2 - 0.28, 2 - 0.28), 0.56, 0.56, fc=GCN_C, ec=INK, lw=1.5, zorder=3))
ax.set_xlim(-0.8, 4.8); ax.set_ylim(-1.6, 4.8)
ax.set_title("pixels: convolution (4.3)")
ax.text(2, -1.1, "same neighbors everywhere,\nin a fixed layout", ha="center", va="center", fontsize=18, color=MUTED)
ax = axes[1]
n = 6
xs = np.arange(n)
for i in range(n):
    for j in range(i + 1, n):
        h = 0.35 * (j - i)
        t = np.linspace(0, np.pi, 50)
        ax.plot((xs[i] + xs[j]) / 2 + (xs[j] - xs[i]) / 2 * np.cos(t), 0.2 + h * np.sin(t), color=FC_C, lw=1.6, alpha=0.7)
for i in range(n):
    ax.add_patch(Circle((xs[i], 0), 0.22, fc="#cfe3f3", ec=INK, lw=1.5, zorder=3))
    ax.text(xs[i], -0.6, f"{i + 1}", ha="center", fontsize=17)
ax.set_xlim(-0.6, n - 0.4); ax.set_ylim(-2.2, 2.4)
ax.set_title("days: attention (4.4)")
ax.text((n - 1) / 2, -1.6, "6 days: every node linked\nto every other", ha="center", va="center", fontsize=18, color=MUTED)
ax = axes[2]
for i, j in zip(*np.nonzero(np.triu(A))):
    ax.plot(*pts[[i, j]].T, color=GRAPH, lw=1.3, alpha=0.8, zorder=1)
ax.scatter(*pts.T, marker="^", s=110, color=GRAPH, edgecolor=INK, lw=1, zorder=3)
ax.set_xlim(-10, 210); ax.set_ylim(-75, 210)
ax.set_title("stations: graph network (4.5)")
ax.text(100, -45, f"irregular: {int(A.sum(1).min())} to {int(A.sum(1).max())} neighbors,\nno order", ha="center",
        va="center", fontsize=18, color=MUTED)
for ax in axes:
    ax.set_aspect("equal")
    ax.axis("off")
save(fig, "graph_family.png")

# ------------------------------------------------------------------ 3. anatomy
fig, ax = plt.subplots(figsize=(17, 7.2))
ax.set_aspect("equal"); ax.axis("off")
ax.set_xlim(-0.5, 34.5); ax.set_ylim(-0.4, 12.2)
sub = np.array([[1.0, 8.0], [3.2, 9.6], [2.6, 6.4], [5.0, 7.8], [4.6, 5.0], [1.2, 4.8], [6.4, 9.4]])
sub_e = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (2, 5), (0, 5), (3, 4), (3, 6), (1, 6), (4, 5)]
hidden_sub = {2, 6}


def mini_graph(ax, off, colors, labels=None):
    for i, j in sub_e:
        ax.plot(*(sub[[i, j]] + off).T, color=GRAPH, lw=2, zorder=1)
    for k, (x, y) in enumerate(sub + off):
        c = colors[k]
        empty = isinstance(c, str) and c == "white"
        tri(ax, x, y, c, ec=RED if empty else INK, s=320, lw=2.5 if empty else 1.2)


vals = plt.cm.viridis(np.linspace(0.15, 0.9, 7))
cin = [vals[k] if k not in hidden_sub else "white" for k in range(7)]
mini_graph(ax, np.array([0.0, 0.0]), cin)
ax.text(3.7, 11.4, "each station: 2 numbers", ha="center", fontsize=19, weight="bold")
ax.text(3.7, 3.2, "value (0 if hidden)\nmask flag (1 if hidden)", ha="center", va="top", fontsize=16, color=MUTED)
arrow(ax, (7.6, 7.3), (9.3, 7.3))
box(ax, 9.5, 5.6, 4.4, 3.4, "message\npassing 1", GRAPH, fs=19)
arrow(ax, (14.0, 7.3), (15.4, 7.3))
box(ax, 15.6, 5.6, 4.4, 3.4, "message\npassing 2", GRAPH, fs=19)
ax.text(14.75, 10.2, "each station: 32 numbers after each layer;\nthe same weights at every station",
        ha="center", fontsize=16, color=GRAPH)
arrow(ax, (20.1, 7.3), (21.5, 7.3))
box(ax, 21.7, 5.6, 3.9, 3.4, "linear\n32 → 1", OUT_C, fs=19)
arrow(ax, (25.7, 7.3), (27.0, 7.3))
cout = list(plt.cm.viridis(np.linspace(0.15, 0.9, 7)))
mini_graph(ax, np.array([26.6, 0.0]), cout)
ax.text(30.3, 11.4, "log10 PGA at every station", ha="center", fontsize=19, weight="bold")
ax.text(30.3, 3.2, "the loss: the hidden\nstations only", ha="center", va="top", fontsize=16, color=MUTED)
ax.text(17, 0.2, "tensor shapes: (batch, 40, 2) in  →  (batch, 40, 32)  →  (batch, 40, 32)  →  (batch, 40) out",
        ha="center", fontsize=17, color=MUTED)
save(fig, "gnn_anatomy.png")

# ------------------------------------------------------------------ 4. adjacency by hand (Your turn 1)
yt1 = N["your_turn_1"]
x5 = np.array(yt1["xy_km"])
D5 = np.hypot(x5[:, None, 0] - x5[None, :, 0], x5[:, None, 1] - x5[None, :, 1])
nn5 = np.argsort(D5 + np.diag(np.full(5, np.inf)), axis=1)[:, :2]
A5 = np.zeros((5, 5))
for i in range(5):
    A5[i, nn5[i]] = 1
A5 = np.maximum(A5, A5.T)
deg5 = (A5 + np.eye(5)).sum(1)
A5_hat = (A5 + np.eye(5)) / np.sqrt(deg5[:, None] * deg5[None, :])
assert nn5.tolist() == yt1["nearest_two"] and A5.astype(int).tolist() == yt1["A"]
assert np.allclose(np.round(A5_hat, 3), yt1["A_hat"]) and deg5.tolist() == yt1["degree_with_self"]

fig = plt.figure(figsize=(17, 6.4))
ax = fig.add_axes([0.0, 0.08, 0.36, 0.84])
for i, j in zip(*np.nonzero(np.triu(A5))):
    ax.plot(*x5[[i, j]].T, color=GRAPH, lw=3, zorder=1)
lab_off = {0: (-2.4, -0.3, "right"), 1: (0, 2.6, "center"), 2: (-2.4, 0.3, "right"), 3: (0, 2.6, "center"),
           4: (0, 2.6, "center")}
deg_off = {0: (0, -4.4, "center"), 1: (0, -4.4, "center"), 2: (2.4, 1.6, "left"), 3: (0, 8.0, "center"),
           4: (0, -4.4, "center")}
for k, (x, y) in enumerate(x5):
    tri(ax, x, y, GRAPH, s=700)
    dx, dy, ha = lab_off[k]
    ax.text(x + dx, y + dy, str(k), ha=ha, va="bottom" if dy > 1 else "center", fontsize=24, weight="bold")
    dx, dy, ha = deg_off[k]
    ax.text(x + dx, y + dy, f"d̃ = {deg5[k]:.0f}", ha=ha, va="center", fontsize=18, color=MUTED)
ax.set_aspect("equal"); ax.axis("off")
ax.set_xlim(-8, 46); ax.set_ylim(-9, 16)
ax.set_title("2 nearest each, kept if either chose", fontsize=20)
ax.text(20, -12.5, "d̃: neighbors + the station itself", ha="center", fontsize=16, color=MUTED)


def matrix(ax, M, fmt, title, hl=None, cmap=None):
    n = len(M)
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            fc = "white" if v == 0 else ("#f3d6e6" if cmap is None else plt.cm.RdPu(0.08 + 0.32 * v / M.max()))
            ec = RED if hl == (i, j) else LIGHT
            ax.add_patch(Rectangle((j, n - 1 - i), 1, 1, fc=fc, ec=ec, lw=4 if hl == (i, j) else 1.5,
                                   zorder=3 if hl == (i, j) else 2))
            ax.text(j + 0.5, n - 0.5 - i, fmt(v), ha="center", va="center", fontsize=17 if "." in fmt(1.0) else 20,
                    color=INK if v else MUTED)
        ax.text(-0.3, n - 0.5 - i, str(i), ha="right", va="center", fontsize=17, color=MUTED)
        ax.text(i + 0.5, n + 0.15, str(i), ha="center", va="bottom", fontsize=17, color=MUTED)
    ax.set_xlim(-0.8, n + 0.1); ax.set_ylim(-0.2, n + 1.4)
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(n / 2, n + 0.85, title, ha="center", va="bottom", fontsize=20, weight="bold")


ax = fig.add_axes([0.37, 0.05, 0.25, 0.9])
matrix(ax, A5, lambda v: f"{v:.0f}", "adjacency A")
ax = fig.add_axes([0.66, 0.05, 0.33, 0.9])
matrix(ax, A5_hat, lambda v: "0" if v == 0 else f"{v:.3f}", r"$\hat{A} = \tilde{D}^{-1/2}\,(A + I)\,\tilde{D}^{-1/2}$", hl=(1, 3), cmap=True)
fig.text(0.825, 0.0, f"row 1, column 3: 1/√(5 × 3) = {A5_hat[1, 3]:.3f}", ha="center", fontsize=19, color=RED)
save(fig, "adjacency_by_hand.png")

# ------------------------------------------------------------------ 5. one message-passing update (Your turn 2)
yt2 = N["your_turn_2"]
h_i, h_nb, e = np.array(yt2["h_i"]), np.array(yt2["h_neighbors"]), np.array(yt2["edge_d_dx_dy_per_50km"])
W_m, W_u = np.array(yt2["W_m"]), np.array(yt2["W_u"])
msg = np.maximum(0, np.concatenate([h_nb, e], axis=1) @ W_m.T)
mean = msg.mean(0)
h_new = np.maximum(0, W_u @ np.concatenate([h_i, mean]))
assert np.allclose(np.round(msg, 3), yt2["messages"]) and np.allclose(np.round(mean, 3), yt2["mean"])
assert np.allclose(np.round(h_new, 3), yt2["h_new"])

fig, ax = plt.subplots(figsize=(17, 7.4))
ax.set_aspect("equal"); ax.axis("off")
ax.set_xlim(-18, 66); ax.set_ylim(-17, 18.5)
nb_km = e[:, 1:] * 50.0     # east, north offsets in km
ax.text(-17, 17.2, "message = ReLU(W_m [h_j, d, Δx, Δy]);  with these weights: [ReLU(v − 0.5 d), ReLU(−v − 0.5 d)]",
        ha="left", fontsize=19, color=INK)
for k, ((dx, dy), (v, _)) in enumerate(zip(nb_km, h_nb)):
    ax.plot([0, dx], [0, dy], color=GRAPH, lw=3, zorder=1)
    tri(ax, dx, dy, plt.cm.viridis(0.5 + 0.45 * v), s=900)
    vx, vy, ha = [(dx, dy - 3.6, "center"), (dx + 2.4, dy, "left"), (dx - 2.4, dy + 0.5, "right")][k]
    ax.text(vx, vy, f"v = {v:g}".replace("-", "−"), ha=ha, va="center", fontsize=20, weight="bold")
    mx, my = dx / 2, dy / 2
    ax.text(mx + [0, 1.2, 1.0][k], my + [1.4, 0, 0.6][k], f"d = {e[k, 0]:g} ({e[k, 0] * 50:.0f} km)",
            fontsize=17, color=MUTED, ha=["center", "left", "left"][k], va="bottom" if k == 0 else "center")
tri(ax, 0, 0, "white", ec=RED, s=1100, lw=3)
ax.text(-2.6, -1.2, "station i, hidden\nh_i = [0, 1]", ha="right", va="top", fontsize=19, color=RED)
x0 = 33
ax.text(x0, 11.4, "messages", fontsize=20, weight="bold")
for k in range(3):
    ax.text(x0, 8.2 - 3.2 * k, f"from v = {h_nb[k, 0]:g}:".replace("-", "−") + f"   [{msg[k, 0]:.1f}, {msg[k, 1]:.1f}]",
            fontsize=19)
ax.plot([x0, x0 + 23], [-1.6, -1.6], color=INK, lw=1.5)
ax.text(x0, -4.6, f"mean:   [{mean[0]:.3f}, {mean[1]:.3f}]", fontsize=20, weight="bold", color=GRAPH)
ax.text(x0, -9.0, "update = ReLU(W_u [h_i, mean])", fontsize=18, color=MUTED)
ax.text(x0, -13.4, f"new state of i:   [{h_new[0]:.3f}, {h_new[1]:.1f}]", fontsize=21, weight="bold", color=OUT_C)
save(fig, "message_by_hand.png")

# ------------------------------------------------------------------ 6. hops: what one station hears from
reach = []
P = np.eye(40)
A1 = A + np.eye(40)
for hops in (1, 2, 3):
    P = (P @ A1 > 0).astype(float)
    reach.append(P.sum(1))
reach = np.array(reach)          # (3, 40): stations reached after 1, 2, 3 layers, the station itself included
centre = np.array([100.0, 100.0])
s0 = int(np.argmin(np.hypot(*(pts - centre).T)))
hop_of = np.full(40, np.inf)
hop_of[s0] = 0
frontier = {s0}
for hcount in (1, 2, 3):
    nxt = {j for i in frontier for j in np.flatnonzero(A[i]) if hop_of[j] == np.inf}
    for j in nxt:
        hop_of[j] = hcount
    frontier = nxt
far = [float(D[s0, hop_of <= h].max()) for h in (1, 2, 3)]
out["hops"] = dict(station=s0, reached_by_station=[int((hop_of <= h).sum()) for h in (1, 2, 3)],
                   farthest_km_by_station=[round(f, 1) for f in far],
                   reached_mean_all_stations=[round(float(r.mean()), 1) for r in reach],
                   reached_min_all_stations=[int(r.min()) for r in reach],
                   reached_max_all_stations=[int(r.max()) for r in reach],
                   median_neighbor_distance_km=round(N["graph"]["neighbor_distance_km_median"], 1))
fig, axes = plt.subplots(1, 3, figsize=(17, 6.6))
shade = {0: RED, 1: GRAPH, 2: "#e3a9c9", 3: "#f3d6e6"}
for L, ax in zip((1, 2, 3), axes):
    for i, j in zip(*np.nonzero(np.triu(A))):
        on = hop_of[i] <= L and hop_of[j] <= L and min(hop_of[i], hop_of[j]) < L
        ax.plot(*pts[[i, j]].T, color=GRAPH if on else LIGHT, lw=2 if on else 1, zorder=1)
    for k in range(40):
        hk = hop_of[k]
        c = shade[int(hk)] if hk <= L else "white"
        tri(ax, *pts[k], c, ec=INK if hk <= L else "#b9b3a8", s=170 if k != s0 else 380)
    ax.set_title(f"{L} layer{'s' if L > 1 else ''}: {int((hop_of <= L).sum())} of 40", fontsize=22)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-10, 210); ax.set_ylim(-30, 210)
    ax.text(100, -22, f"farthest: {far[L - 1]:.0f} km", ha="center", fontsize=18, color=MUTED)
save(fig, "hops.png")


# ------------------------------------------------------------------ 7. weights (the notebook's classes, instantiated)
class FullyConnected(torch.nn.Module):
    def __init__(self, n_stations=40, hidden=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2 * n_stations, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, n_stations))


class GraphConv(torch.nn.Module):
    def __init__(self, n_layers=2, hidden=32):
        super().__init__()
        dims = [2] + [hidden] * n_layers
        self.layers = torch.nn.ModuleList(torch.nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.head = torch.nn.Linear(hidden, 1)


class MessagePassingLayer(torch.nn.Module):
    def __init__(self, d_in, hidden):
        super().__init__()
        self.message = torch.nn.Linear(d_in + 3, hidden)
        self.update = torch.nn.Linear(d_in + hidden, hidden)


class MessagePassing(torch.nn.Module):
    def __init__(self, n_layers=2, hidden=32):
        super().__init__()
        dims = [2] + [hidden] * n_layers
        self.layers = torch.nn.ModuleList(MessagePassingLayer(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.head = torch.nn.Linear(hidden, 1)


def nw(m):
    return sum(p.numel() for p in m.parameters())


w = {"fc40": nw(FullyConnected(40)), "fc50": nw(FullyConnected(50)), "gcn": nw(GraphConv()), "mp": nw(MessagePassing())}
assert w["fc40"] == N["leaderboard"]["fully connected"]["weights"]
assert w["gcn"] == N["leaderboard"]["graph convolution"]["weights"]
assert w["mp"] == N["leaderboard"]["message passing"]["weights"]
mp_l = MessagePassing()
w["mp_layer1"], w["mp_layer2"] = nw(mp_l.layers[0]), nw(mp_l.layers[1])
out["weights"] = w
fig, ax = plt.subplots(figsize=(16, 6.4))
rows = [("fully connected, 40 stations", w["fc40"], FC_C, "80 → 128 → 128 → 40"),
        ("fully connected, 50 stations", w["fc50"], FC_C, "100 → 128 → 128 → 50: retrain"),
        ("graph convolution, any number", w["gcn"], GCN_C, "2 → 32 → 32, head 32 → 1"),
        ("message passing, any number", w["mp"], GRAPH, f"layer 1 {w['mp_layer1']:,}, layer 2 {w['mp_layer2']:,}, head 33")]
for k, (lab, v, c, note) in enumerate(rows):
    ax.barh(k, v, color=c, height=0.62)
    ax.text(v + 400, k, f"{v:,}   {note}", va="center", fontsize=19)
ax.set_yticks(range(len(rows)), [r[0] for r in rows])
ax.invert_yaxis()
ax.set_xlim(0, 62000)
ax.set_xlabel("number of weights")
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("the graph networks' count does not depend on the number of stations", loc="left")
save(fig, "weights_count.png")

# ------------------------------------------------------------------ 8. test 1: station order
t1 = N["test1_station_order"]
fig, ax = plt.subplots(figsize=(16, 6.6))
names = ["fully connected", "graph convolution", "message passing"]
cols = [FC_C, GCN_C, GRAPH]
XP = [0, 1.3, 2.6]
for k, (nm, c) in enumerate(zip(names, cols)):
    k = XP[k]
    ax.bar(k - 0.19, t1[nm]["mae_original"], 0.36, color=c)
    ax.bar(k + 0.19, t1[nm]["mae_shuffled"], 0.36, color=c, alpha=0.45, hatch="//", edgecolor="white")
    ax.text(k - 0.19, t1[nm]["mae_original"] + 0.006, f"{t1[nm]['mae_original']:.3f}", ha="center", fontsize=18)
    ax.text(k + 0.19, t1[nm]["mae_shuffled"] + 0.006, f"{t1[nm]['mae_shuffled']:.3f}", ha="center", fontsize=18)

for nm, ls in [("inverse-distance weighting", "--"), ("mean of recorded stations", ":")]:
    v = N["baselines"][nm]["mae"]
    ax.axhline(v, color=IDW_C, ls=ls, lw=2)
    ax.text(3.55, v + 0.004, f"{'IDW' if 'inverse' in nm else 'mean of recorded'} {v:.3f}", ha="right",
            va="bottom", fontsize=17, color=MUTED)
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(color=IDW_C, label="original order"),
                   Patch(facecolor=IDW_C, alpha=0.45, hatch="//", edgecolor="white", label="stations shuffled")],
          loc="upper right", fontsize=17, frameon=False)
chg = [t1[nm]["largest_output_change"] for nm in names]
ax.set_xticks(XP, [f"{nm}\nlargest change: " + (f"{c:.1e}" if c < 1e-3 else f"{c:.1f}") for nm, c in zip(names, chg)])
ax.set_xlim(-0.6, 3.6)
ax.tick_params(axis="x", labelsize=18)
ax.set_ylim(0, 0.42)
ax.set_ylabel("test MAE (log10 PGA)")
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.5, -0.08, TAG + "; mean of seeds 0, 1, 2; new order: default_rng(99)", ha="center", fontsize=15, color=MUTED)
save(fig, "station_order.png")

# ------------------------------------------------------------------ 9. heads
fig, axes = plt.subplots(1, 3, figsize=(17, 6.6))
small = np.array([[0.0, 2.0], [1.6, 3.2], [1.4, 1.0], [3.2, 2.4], [3.0, 0.2], [0.2, 0.0]])
small_e = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (2, 5), (0, 5), (4, 5)]
vcols = plt.cm.viridis([0.2, 0.8, 0.5, 0.65, 0.35, 0.9])
titles = ["node level (today)", "graph level", "edge level"]
subs = ["one value per station:\nshaking at each station", "one value per earthquake:\nlocation, magnitude",
        "one value per pair of stations:\nsuch as a time lag"]
for k, ax in enumerate(axes):
    for a, b in small_e:
        lw, c = (5, OUT_C) if (k == 2 and (a, b) in [(0, 1), (2, 3), (4, 5)]) else (2.2, GRAPH)
        ax.plot(*small[[a, b]].T, color=c, lw=lw, zorder=1)
    for i, (x, y) in enumerate(small):
        tri(ax, x, y, vcols[i] if k == 0 else GRAPH, s=420)
    if k == 1:
        ax.add_patch(FancyBboxPatch((-0.5, -0.6), 4.2, 4.4, boxstyle="round,pad=0.05,rounding_size=0.4",
                                    fill=False, ec=OUT_C, lw=3, ls="--"))
        ax.add_patch(FancyArrowPatch((1.6, -0.75), (1.6, -1.45), arrowstyle="-|>", mutation_scale=22, color=OUT_C, lw=2.5))
        box(ax, 0.55, -2.35, 2.1, 0.8, "M 4.8", OUT_C, fs=18)
    ax.set_title(titles[k])
    ax.text(1.6, -3.05, subs[k], ha="center", va="top", fontsize=17, color=MUTED)
    ax.set_xlim(-1, 4.2); ax.set_ylim(-4.6, 3.8)
    ax.set_aspect("equal"); ax.axis("off")
save(fig, "heads.png")

# ------------------------------------------------------------------ 10. station terms learned (seed 0)
st = N["site_terms_seed0"]
true = np.array(st["true"])
fig, axes = plt.subplots(1, 4, figsize=(17, 5.6), sharex=True, sharey=True)
panels = [("inverse-distance weighting", "IDW", IDW_C), ("graph convolution", "graph convolution", GCN_C),
          ("fully connected", "fully connected", FC_C), ("message passing", "message passing", GRAPH)]
for ax, (key, lab, c) in zip(axes, panels):
    est = np.array(st[key])
    r = float(np.corrcoef(est, true)[0, 1])
    ax.plot([-0.4, 0.4], [-0.4, 0.4], color=MUTED, lw=1.2, ls="--")
    ax.scatter(true, est, s=70, color=c, edgecolor=INK, lw=0.6)
    ax.set_title(f"{lab}\nr = {r:.2f}", fontsize=20)
    ax.set_xlim(-0.4, 0.4); ax.set_ylim(-0.4, 0.4)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=15)
    ax.set_xticks([-0.3, 0, 0.3])
    ax.set_yticks([-0.3, 0, 0.3])
    ax.set_xlabel("true station term", fontsize=17)
axes[0].set_ylabel("station term learned", fontsize=17)
fig.text(0.5, 0.0, TAG + ": 40 stations, hidden test values, networks of seed 0 (log10 PGA)",
         ha="center", fontsize=15, color=MUTED)
save(fig, "site_terms.png")

(FIGS / "figs_numbers.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
