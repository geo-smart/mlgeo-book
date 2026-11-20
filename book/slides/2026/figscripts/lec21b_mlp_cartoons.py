"""Cartoons of the fully connected network (multilayer perceptron, MLP) for lecture 21b.

One concept per figure, written to figs/lec21b_mlp/, every number computed here
(numbers.json):

 1. mlp_anatomy.png       notebook 4.1's network: 61 features -> 32 hidden -> 4 classes
 2. dense_by_hand.png     one dense layer by hand: 3 inputs, 2 neurons, z = W x + b, ReLU
 3. width_boundary.png    what hidden neurons do: 1, 3 and 16 neurons on two synthetic features
 4. weights_count.png     weights per layer of the 4.1 and 4.2 networks: n_out x (n_in + 1)
 5. width_depth.png       wide and shallow against deep and narrow, at about the same weights
 6. no_order.png          an MLP does not see order: waveform samples shuffled, same accuracy
 7. scaling.png           why inputs are scaled: gradient descent on unscaled and scaled inputs
 8. dropout.png           dropout: random neurons off in each training batch, all on at test
 9. heads.png             the same hidden layers, a classification head or a regression head

Regenerate: pixi run python book/slides/2026/figscripts/lec21b_mlp_cartoons.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))  # book/, for mlgeo_synth
import mlgeo_synth  # noqa: E402

FIGS = HERE.parent / "figs" / "lec21b_mlp"
FIGS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 20, "axes.titlesize": 22, "axes.titleweight": "bold"})
INK, MUTED, LIGHT = "#26241f", "#6e675c", "#e8e4dc"
IN_C, HID_C, OUT_C, POS_C, NEG_C, K_C = "#8a857c", "#0072B2", "#009E73", "#009E73", "#D55E00", "#E69F00"
numbers = {}


def canvas(w=16, h=7.5):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def arrow(ax, p, q, color=INK, lw=2.5, ms=22):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw))


def box(ax, x, y, w, h, text, fc, fs=20, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.25", fc=fc, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, weight="bold")


def column(ax, x, ys, color, r=0.32, crossed=()):
    for i, y in enumerate(ys):
        ax.add_patch(Circle((x, y), r, fc=color if i not in crossed else LIGHT, ec=INK, lw=1.2))
        if i in crossed:
            ax.plot([x - r, x + r], [y - r, y + r], color=NEG_C, lw=2.5)
            ax.plot([x - r, x + r], [y + r, y - r], color=NEG_C, lw=2.5)


def connect(ax, x0, ys0, x1, ys1, r=0.32, color=MUTED, lw=0.5, skip=()):
    for i, a in enumerate(ys0):
        for j, b in enumerate(ys1):
            if i in skip or ("to", j) in skip:
                continue
            ax.plot([x0 + r, x1 - r], [a, b], color=color, lw=lw, zorder=0)


def save(fig, name):
    fig.savefig(FIGS / name, dpi=110, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def dense_weights(sizes):
    return [n_out * (n_in + 1) for n_in, n_out in zip(sizes[:-1], sizes[1:])]


# ------------------------------------------------------------------ 1. anatomy (notebook 4.1, NN1)
w41 = dense_weights([61, 32, 4])
numbers["anatomy"] = {"sizes": [61, 32, 4], "weights_per_layer": w41, "total": sum(w41)}
fig, ax = canvas(17, 8)
ax.set_xlim(-1, 33); ax.set_ylim(-2.8, 12.5)
yin = np.linspace(1.0, 9.0, 9)
yhid = np.linspace(1.6, 8.4, 7)
yout = np.linspace(3.0, 7.0, 4)
connect(ax, 4, yin, 12, yhid)
connect(ax, 12, yhid, 20, yout)
column(ax, 4, yin, IN_C)
column(ax, 12, yhid, HID_C)
column(ax, 20, yout, OUT_C)
ax.text(4, 9.9, "⋮", ha="center", fontsize=22); ax.text(12, 9.2, "⋮", ha="center", fontsize=22)
ax.text(4, 11.2, "input: 61 features\nof one seismic event", ha="center", fontsize=19, weight="bold")
ax.text(12, 11.2, "hidden layer\n32 neurons, ReLU", ha="center", fontsize=19, weight="bold", color=HID_C)
ax.text(20, 11.2, "output layer\n4 scores", ha="center", fontsize=19, weight="bold", color=OUT_C)
labels = ["earthquake", "explosion", "noise", "surface event"]
probs = [0.07, 0.81, 0.08, 0.04]
for y, lab, p in zip(yout[::-1], labels, probs):
    ax.text(21.0, y, lab, va="center", fontsize=17)
    ax.add_patch(Rectangle((25.6, y - 0.3), 6 * p, 0.6, fc=OUT_C))
    ax.text(25.7 + 6 * p, y, f" {p:.2f}", va="center", fontsize=16)
ax.text(28.0, 8.3, "softmax, for example", ha="center", fontsize=16, color=MUTED)
ax.text(8, -0.6, f"61 × 32 + 32 = {w41[0]:,}", ha="center", fontsize=18)
ax.text(16, -0.6, f"32 × 4 + 4 = {w41[1]}", ha="center", fontsize=18)
ax.text(12, -2.0, f"every input connects to every neuron: {sum(w41):,} weights (notebook 4.1)", ha="center", fontsize=18, color=MUTED)
save(fig, "mlp_anatomy.png")

# ------------------------------------------------------------------ 2. one dense layer by hand
x = np.array([1.0, -0.5, 2.0])
W = np.array([[0.5, -1.0, 0.25], [-1.0, 0.5, -0.25]])
b = np.array([0.1, 0.2])
z = W @ x + b
a = np.maximum(z, 0)
numbers["dense_by_hand"] = {"x": x.tolist(), "W": W.tolist(), "b": b.tolist(), "z": z.tolist(), "relu": a.tolist()}
fig, ax = canvas(17, 7.5)
ax.set_xlim(-1, 33); ax.set_ylim(-1.5, 12)
yin = [8.5, 5.5, 2.5]
yh = [7.5, 3.5]
for i, yi in enumerate(yin):
    for j, yj in enumerate(yh):
        c = HID_C if j == 0 else NEG_C
        ax.plot([2.4, 8.6], [yi, yj], color=c, lw=2, zorder=0)
for i, yi in enumerate(yin):
    ax.add_patch(Circle((2, yi), 0.45, fc=IN_C, ec=INK))
    ax.text(0.9, yi, f"x{i + 1} = {x[i]:g}", ha="right", va="center", fontsize=19)
for j, yj in enumerate(yh):
    ax.add_patch(Circle((9, yj), 0.55, fc=HID_C if j == 0 else NEG_C, ec=INK))
    ax.text(9, yj, f"z{j + 1}", ha="center", va="center", fontsize=17, color="white", weight="bold")
ax.text(-0.5, 10.6, "one weight per link (blue: to z1, red: to z2); biases b1 = 0.1, b2 = 0.2", fontsize=19)
rows = [f"z1 = 0.5×1 − 1×(−0.5) + 0.25×2 + 0.1 = {z[0]:.2f}   → ReLU → {a[0]:.2f}",
        f"z2 = −1×1 + 0.5×(−0.5) − 0.25×2 + 0.2 = {z[1]:.2f}   → ReLU → {a[1]:.2f}"]
ax.text(11.5, 7.5, rows[0], va="center", fontsize=19, color=HID_C)
ax.text(11.5, 3.5, rows[1], va="center", fontsize=19, color=NEG_C)
ax.text(11.5, 0.4, "in matrix form: z = W x + b, then a = ReLU(z).   W is 2 × 3: one row per neuron.",
        fontsize=19, color=INK)
save(fig, "dense_by_hand.png")

# ------------------------------------------------------------------ 3. what hidden neurons do
rng = np.random.default_rng(4)
n = 600
r = np.r_[rng.normal(0.9, 0.35, n // 2), rng.normal(2.4, 0.35, n // 2)]
th = rng.uniform(0, 2 * np.pi, n)
X2 = np.c_[r * np.cos(th), 0.7 * r * np.sin(th)]
y2 = np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)
gx, gy = np.meshgrid(np.linspace(-3.6, 3.6, 300), np.linspace(-2.8, 2.8, 240))
G = np.c_[gx.ravel(), gy.ravel()]
fig, axes = plt.subplots(1, 3, figsize=(17, 6.2))
acc = {}
for ax, h in zip(axes, [1, 3, 16]):
    clf = MLPClassifier(hidden_layer_sizes=(h,), activation="relu", solver="lbfgs", max_iter=3000, random_state=0).fit(X2, y2)
    acc[h] = float(clf.score(X2, y2))
    ax.contourf(gx, gy, clf.predict_proba(G)[:, 1].reshape(gx.shape), levels=[0, 0.5, 1], colors=["#d6e6f4", "#fbe0cf"])
    ax.scatter(*X2[y2 == 0].T, s=8, color=HID_C); ax.scatter(*X2[y2 == 1].T, s=8, color=NEG_C)
    if h <= 3:
        for k in range(h):
            w1, w2 = clf.coefs_[0][:, k]; b0 = clf.intercepts_[0][k]
            xx = np.linspace(-3.6, 3.6, 10)
            ax.plot(xx, -(w1 * xx + b0) / w2, color=INK, lw=2, ls="--")
    ax.set_xlim(-3.6, 3.6); ax.set_ylim(-2.8, 2.8); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{h} hidden neuron{'s' if h > 1 else ''}: {100 * acc[h]:.0f}% correct", fontsize=20)
axes[0].set_xlabel("feature 1 (scaled)"); axes[0].set_ylabel("feature 2 (scaled)")
fig.text(0.5, -0.02, "dashed: where each ReLU neuron switches on. Each neuron cuts the plane with one straight line; "
         "the output layer combines the cuts.", ha="center", fontsize=17, color=MUTED)
numbers["width_boundary_training_accuracy"] = acc
fig.tight_layout()
save(fig, "width_boundary.png")

# ------------------------------------------------------------------ 4. counting weights
w42 = dense_weights([6, 32, 32, 1])
wdeep = dense_weights([61, 128, 64, 32, 4])
numbers["weights"] = {"4.1 NN1": w41, "4.2 MLP": w42, "4.1 DeepMLP (without batch norm)": wdeep}
fig, ax = plt.subplots(figsize=(16, 7))
rows = [("4.1: 61 → 32", 61, 32, w41[0]), ("4.1: 32 → 4 classes", 32, 4, w41[1]),
        ("4.2: 6 → 32", 6, 32, w42[0]), ("4.2: 32 → 32", 32, 32, w42[1]), ("4.2: 32 → 1 number", 32, 1, w42[2])]
yy = np.arange(len(rows))[::-1]
ax.barh(yy, [r[3] for r in rows], color=[HID_C, HID_C, OUT_C, OUT_C, OUT_C], height=0.6)
for y, (lab, ni, no, w) in zip(yy, rows):
    ax.text(w + 25, y, f"{w:,} = {no} × ({ni} + 1)", va="center", fontsize=19)
ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows], fontsize=19)
ax.set_xlim(0, 3000); ax.set_xlabel("number of weights")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title(f"weights = outputs × (inputs + 1):  4.1 has {sum(w41):,} in all, 4.2 has {sum(w42):,}", loc="left", fontsize=22)
save(fig, "weights_count.png")

# ------------------------------------------------------------------ 5. width against depth
wide = [61, 256, 4]
deep = [61, 64, 64, 64, 64, 4]
ww, wd = sum(dense_weights(wide)), sum(dense_weights(deep))
numbers["width_depth"] = {"wide": [wide, ww], "deep": [deep, wd]}
fig, axes = plt.subplots(1, 2, figsize=(17, 6.8))
for ax, sizes, name, tot in [(axes[0], wide, "wide and shallow", ww), (axes[1], deep, "deep and narrow", wd)]:
    ax.set_xlim(-0.6, len(sizes) - 0.4); ax.set_ylim(-2.0, 10.5); ax.axis("off")
    prev = None
    for k, s in enumerate(sizes):
        m = min(s, 12) if k not in (0, len(sizes) - 1) else min(s, 6)
        ys = np.linspace(5 - 0.38 * m, 5 + 0.38 * m, m)
        if prev is not None:
            for a_ in prev:
                for b_ in ys:
                    ax.plot([k - 1, k], [a_, b_], color=MUTED, lw=0.3, zorder=0)
        ax.scatter([k] * m, ys, s=120, color=IN_C if k == 0 else OUT_C if k == len(sizes) - 1 else HID_C, zorder=2, edgecolor=INK)
        ax.text(k, -0.5, str(s), ha="center", fontsize=18, weight="bold")
        prev = ys
    ax.text((len(sizes) - 1) / 2, 9.9, name, ha="center", fontsize=23, weight="bold")
    ax.text((len(sizes) - 1) / 2, -1.7, f"{tot:,} weights", ha="center", fontsize=20)
fig.text(0.5, 0.02, "numbers: neurons per layer (dots drawn: at most 12)", ha="center", fontsize=16, color=MUTED)
save(fig, "width_depth.png")

# ------------------------------------------------------------------ 6. no order
Xs, ys_, _ = mlgeo_synth.seismogram_dataset(n_events=400, n_noise=400, seed=11)
Xd = Xs[:, ::10]                                  # 300 samples per window, 10 Hz
Xd = Xd / Xd.std(axis=1, keepdims=True)
perm = np.random.default_rng(0).permutation(Xd.shape[1])
Xtr, Xte, ytr, yte = train_test_split(Xd, ys_, test_size=0.3, random_state=0, stratify=ys_)
acc_o, acc_p = [], []
for seed in range(3):
    m = MLPClassifier(hidden_layer_sizes=(32,), max_iter=400, random_state=seed)
    acc_o.append(m.fit(Xtr, ytr).score(Xte, yte))
    m = MLPClassifier(hidden_layer_sizes=(32,), max_iter=400, random_state=seed)
    acc_p.append(m.fit(Xtr[:, perm], ytr).score(Xte[:, perm], yte))
numbers["no_order"] = {"accuracy_as_recorded": acc_o, "accuracy_shuffled": acc_p}
fig, axes = plt.subplots(2, 1, figsize=(16, 6.5), sharex=True)
i_ev = np.where(ys_ == 1)[0][0]
t = np.arange(300) / 10
axes[0].plot(t, Xd[i_ev], color=IN_C, lw=1.2)
axes[0].set_title(f"as recorded: MLP test accuracy {100 * np.mean(acc_o):.1f}% ± {100 * np.std(acc_o):.1f}", loc="left", fontsize=20)
axes[1].plot(t, Xd[i_ev, perm], color=NEG_C, lw=1.0)
axes[1].set_title(f"samples shuffled (the same shuffle for every window): {100 * np.mean(acc_p):.1f}% ± {100 * np.std(acc_p):.1f}",
                  loc="left", fontsize=20)
for a_ in axes:
    a_.set_yticks([])
    for s in ("top", "right", "left"):
        a_.spines[s].set_visible(False)
axes[1].set_xlabel("sample (30 s window at 10 Hz); synthetic event windows against noise windows, 3 seeds")
save(fig, "no_order.png")

# ------------------------------------------------------------------ 7. scaling
rng = np.random.default_rng(2)
nS = 200
dist = rng.uniform(5, 150, nS)                    # km
mag = rng.uniform(1, 4, nS)
yS = 0.02 * dist - 0.8 * mag + rng.normal(0, 0.3, nS)
yS = yS - yS.mean()


def gd_path(F, lr, steps=60):
    w = np.zeros(2); path = [w.copy()]
    for _ in range(steps):
        g = 2 * F.T @ (F @ w - yS) / len(yS)
        w = w - lr * g; path.append(w.copy())
    return np.array(path)


Fu = np.c_[dist - dist.mean(), mag - mag.mean()]
Fs = Fu / Fu.std(axis=0)
Hu, Hs = 2 * Fu.T @ Fu / nS, 2 * Fs.T @ Fs / nS
lr_u, lr_s = 1.9 / np.linalg.eigvalsh(Hu).max(), 1.9 / np.linalg.eigvalsh(Hs).max() / 2
pu, ps = gd_path(Fu, lr_u), gd_path(Fs, lr_s)
wu_opt, ws_opt = np.linalg.lstsq(Fu, yS, rcond=None)[0], np.linalg.lstsq(Fs, yS, rcond=None)[0]
numbers["scaling"] = {"condition_unscaled": float(np.linalg.cond(Hu)), "condition_scaled": float(np.linalg.cond(Hs)),
                      "distance_weight_after_60": float(pu[-1, 0]), "magnitude_weight_after_60_unscaled": float(pu[-1, 1]),
                      "magnitude_weight_optimum_unscaled": float(wu_opt[1]), "scaled_final_error": float(np.linalg.norm(ps[-1] - ws_opt))}
fig, axes = plt.subplots(1, 2, figsize=(16, 6.6))
for ax, F, p, wopt, title, labs in [
        (axes[0], Fu, pu, wu_opt, "raw inputs: km and magnitude units", ("weight of distance (per km)", "weight of magnitude")),
        (axes[1], Fs, ps, ws_opt, "each input scaled to unit variance", ("weight of distance (scaled)", "weight of magnitude (scaled)"))]:
    span0 = max(abs(wopt[0]) * 2.2, 0.05); span1 = max(abs(wopt[1]) * 1.6, 0.5)
    A, B = np.meshgrid(np.linspace(-0.2 * span0, span0, 200), np.linspace(-span1, 0.4 * span1, 200))
    Wg = np.c_[A.ravel(), B.ravel()]
    L = ((Wg @ F.T - yS) ** 2).mean(axis=1).reshape(A.shape)
    ax.contour(A, B, np.log(L), levels=18, colors=MUTED, linewidths=0.8)
    ax.plot(p[:, 0], p[:, 1], "o-", color=NEG_C, ms=4, lw=1.5)
    ax.plot(*wopt, "*", color=OUT_C, ms=22)
    ax.set_title(title, loc="left", fontsize=20); ax.set_xlabel(labs[0]); ax.set_ylabel(labs[1])
fig.text(0.5, -0.03, "contours: the loss for two weights; red: 60 steps of gradient descent from zero; green star: the lowest loss",
         ha="center", fontsize=17, color=MUTED)
fig.tight_layout()
save(fig, "scaling.png")

# ------------------------------------------------------------------ 8. dropout
fig, axes = plt.subplots(1, 3, figsize=(17, 6.2))
rngd = np.random.default_rng(7)
for ax, title, p in zip(axes, ["training, batch 1", "training, batch 2", "validation and test"], [0.3, 0.3, 0.0]):
    ax.set_xlim(-0.6, 3.6); ax.set_ylim(-0.5, 7.5); ax.axis("off"); ax.set_aspect("equal")
    layers = [np.linspace(1, 6, 5), np.linspace(0.5, 6.5, 7), np.linspace(1.5, 5.5, 5), np.linspace(2.5, 4.5, 2)]
    off = [set(), set(np.where(rngd.random(7) < p)[0]) if p else set(), set(np.where(rngd.random(5) < p)[0]) if p else set(), set()]
    for k in range(3):
        for i, a_ in enumerate(layers[k]):
            for j, b_ in enumerate(layers[k + 1]):
                if i in off[k] or j in off[k + 1]:
                    continue
                ax.plot([k, k + 1], [a_, b_], color=MUTED, lw=0.6, zorder=0)
    for k, ys in enumerate(layers):
        for i, yv in enumerate(ys):
            on = i not in off[k]
            ax.add_patch(Circle((k, yv), 0.28, fc=(IN_C if k == 0 else OUT_C if k == 3 else HID_C) if on else LIGHT, ec=INK))
            if not on:
                ax.plot([k - 0.2, k + 0.2], [yv - 0.2, yv + 0.2], color=NEG_C, lw=2.5)
                ax.plot([k - 0.2, k + 0.2], [yv + 0.2, yv - 0.2], color=NEG_C, lw=2.5)
    ax.set_title(title, fontsize=21)
fig.text(0.5, 0.0, "dropout 0.3: in each training batch, each hidden neuron is switched off with probability 0.3; "
         "at evaluation all neurons work", ha="center", fontsize=17, color=MUTED)
save(fig, "dropout.png")

# ------------------------------------------------------------------ 9. heads
fig, ax = canvas(17, 7.5)
ax.set_xlim(0, 34); ax.set_ylim(-1, 14)
box(ax, 0.3, 5.0, 8.5, 3.6, "hidden layers\nlinear → ReLU\n× 1 or 2", HID_C, fs=20)
ax.text(4.55, 3.9, "the same for both tasks", ha="center", fontsize=17, color=MUTED)
arrow(ax, (8.9, 6.8), (10.4, 6.8))
for i in range(8):
    ax.add_patch(Rectangle((10.6, 4.9 + i * 0.5), 0.6, 0.5, fc="#c6dbef", ec=INK, lw=0.8))
ax.text(10.9, 3.9, "32", ha="center", fontsize=18, weight="bold")
arrow(ax, (11.4, 8.0), (14.3, 10.6), color=POS_C)
box(ax, 14.5, 9.6, 5.6, 2.2, "linear 32 → 4\nscores (logits)", POS_C, fs=18)
arrow(ax, (20.3, 10.7), (21.8, 10.7), color=POS_C)
box(ax, 22.0, 9.6, 3.6, 2.2, "softmax", "#7fc6a4", fs=19)
ax.text(26.2, 10.7, "→ source type\n(notebook 4.1)", fontsize=17, va="center")
ax.text(14.5, 12.7, "classification: which class?", fontsize=21, weight="bold", color=POS_C)
ax.text(14.5, 8.6, "loss: cross-entropy (softmax inside)", fontsize=17, color=MUTED)
arrow(ax, (11.4, 5.4), (14.3, 2.8), color=NEG_C)
box(ax, 14.5, 1.6, 5.6, 2.2, "linear 32 → 1\nno activation", NEG_C, fs=18)
arrow(ax, (20.3, 2.7), (21.8, 2.7), color=NEG_C)
ax.text(22.0, 2.7, "one number: P395's east position\nin 30 days, mm (notebook 4.2)", fontsize=17, va="center")
ax.text(14.5, 4.6, "regression: how much?", fontsize=21, weight="bold", color=NEG_C)
ax.text(14.5, 0.5, "loss: mean squared, absolute or Huber error", fontsize=17, color=MUTED)
save(fig, "heads.png")

(FIGS / "numbers.json").write_text(json.dumps(numbers, indent=2, default=float))
print(json.dumps(numbers, indent=1, default=float)[:2500])
