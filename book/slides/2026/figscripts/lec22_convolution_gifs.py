"""Animated slide figures for lecture 22: what a convolution does to data.

Five animations (GIF, looping) and the last frame of each as a PNG, written to
figs/lec22_convolution/, plus every number quoted on a slide in numbers.json.

1. conv_slide_multiply_sum.gif   one kernel (the P wavelet of the course generator,
                                 0.6 s) slid along a synthetic record: at each
                                 position, multiply weight by sample, add, write one
                                 output. Synthetic (mlgeo_synth), 30 km, SNR 4.
2. conv_same_weights_everywhere.gif
                                 the same event moved through the window: the feature
                                 map moves with it, its maximum does not; a fully
                                 connected neuron (weights tied to positions, the
                                 template at the first position) loses it.
                                 Synthetic (mlgeo_synth), 20 km, SNR 5.
3. conv_kernel_bank_real.gif     three kernels on a real record: the M3.9 earthquake
                                 9 km NNE of Snoqualmie, WA (USGS ComCat uw61939452,
                                 2023-08-08 10:17:23.87 UTC, depth 15.9 km), at UW.SP2
                                 (Seward Park, Seattle; HHZ, 100 Hz; 36 km away).
                                 Waveform from the EarthScope FDSN web service, raw
                                 counts, mean removed.
4. conv_2d_temperature_front.gif a 3x3 kernel slid over NCEP/NCAR Reanalysis 1 air
                                 temperature (sigma 0.995; Kalnay et al., 1996) on
                                 2014-01-07 00 UTC, the January 2014 Arctic outbreak;
                                 file pydata/xarray-data air_temperature.nc @2baf0c2,
                                 same file and checksum as notebook 2.5.
5. conv_receptive_field.gif      how far one unit of the notebook 4.3 detector sees,
                                 layer by layer (three blocks of convolution, kernel 7,
                                 and max pooling by 4), against the S-P times of the
                                 generator's 5-80 km events.

PyTorch's Conv1d and Conv2d compute this sliding weighted sum without flipping the
kernel (a cross-correlation); the animations do the same.

Regenerate: pixi run python book/slides/2026/figscripts/lec22_convolution_gifs.py   (about 3 minutes)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pooch  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from obspy import UTCDateTime  # noqa: E402
from obspy.clients.fdsn import Client  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))  # book/, for mlgeo_synth
from mlgeo_synth.seismic import _wavelet, synthetic_seismogram  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 22, "axes.titleweight": "bold",
    "axes.labelsize": 19, "xtick.labelsize": 16, "ytick.labelsize": 16,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, ORANGE, GREEN, RED, INK, MUTED, GRAY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#26241f", "#6e675c", "#8a857c"
CACHE = pooch.os_cache("mlgeo-figscripts")
FIGS = HERE.parent / "figs" / "lec22_convolution"
FIGS.mkdir(parents=True, exist_ok=True)
DPI = 80          # 16 x 7.5 in -> 1280 x 600 px; the slide stretches it
FS = 100.0
numbers = {}


def save(anim, fig, name, fps):
    anim.save(FIGS / f"{name}.gif", writer=PillowWriter(fps=fps), dpi=DPI)
    fig.savefig(FIGS / f"{name}_last.png", dpi=110)
    plt.close(fig)
    print(f"wrote {name}.gif ({(FIGS / f'{name}.gif').stat().st_size / 1e6:.1f} MB)")


def p_kernel():
    """The generator's P wavelet (5 Hz, decay 0.5 s), first 0.6 s, unit norm."""
    tk = np.arange(int(0.6 * FS)) / FS
    w = _wavelet(tk, 5.0, 0.5)
    return w / np.linalg.norm(w)


def slide_sum(x, w):
    """Output k = sum_j w[j] x[k+j]: what Conv1d computes ('valid' positions)."""
    return np.correlate(x, w, mode="valid")


def standardize(x):
    return (x - x.mean()) / x.std()


# ============================================================== 1. multiply, sum, slide
w = p_kernel()
nk = w.size
t, x, meta = synthetic_seismogram(duration_s=24.0, fs=FS, magnitude=2.5, distance_km=30.0, f0=5.0, snr=4.0, seed=7)
x = standardize(x)
y = slide_sum(x, w)
ty = (np.arange(y.size) + nk / 2) / FS               # output plotted at the window centre
numbers["gif1"] = {"distance_km": 30.0, "snr": 4.0, "t_p_s": meta["t_p"], "t_s_s": meta["t_s"],
                   "kernel_samples": int(nk), "kernel_s": nk / FS,
                   "output_max": float(y.max()), "t_output_max_s": float(ty[y.argmax()]),
                   "output_std_before_p": float(y[ty < meta["t_p"] - 1].std())}

fig = plt.figure(figsize=(16, 7.5))
gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 1.0, 1.1], hspace=0.65, left=0.07, right=0.98, top=0.93, bottom=0.09)
a1, a2, a3 = (fig.add_subplot(gs[i]) for i in range(3))
a1.plot(t, x, color=GRAY, lw=0.8)
for tt, lab in [(meta["t_p"], "P"), (meta["t_s"], "S")]:
    a1.axvline(tt, color=INK, ls=":", lw=1.5)
    a1.text(tt + 0.1, 3.6, lab, fontsize=18, weight="bold", color=INK)
win = a1.add_patch(plt.Rectangle((0, -4.5), nk / FS, 9, color=ORANGE, alpha=0.25))
kline, = a1.plot([], [], color=ORANGE, lw=2.5)
a1.set_xlim(0, t[-1]); a1.set_ylim(-4.5, 4.5); a1.set_yticks([])
a1.set_title("1  the kernel (orange, 60 weights = 0.6 s) sits on the record", loc="left")
a1.set_xlabel("time (s)")

tz = np.arange(nk) / FS
a2.axhline(0, color=MUTED, lw=0.8)
seg_line, = a2.plot([], [], color=GRAY, lw=2, label="record under the kernel")
k_line, = a2.plot(tz, w * 12, color=ORANGE, lw=2.5, label="kernel weights")
bars = a2.bar(tz, np.zeros(nk), width=1 / FS, color=GREEN)
a2.set_xlim(0, nk / FS); a2.set_ylim(-4.5, 4.5); a2.set_yticks([])
a2.set_xticks([0, 0.2, 0.4, 0.6]); a2.set_xticklabels(["0", "0.2", "0.4", "0.6 s"])
a2.legend(loc="upper right", fontsize=15, ncol=2, frameon=False)
title2 = a2.set_title("", loc="left")

a3.axhline(0, color=MUTED, lw=0.8)
out_line, = a3.plot([], [], color=BLUE, lw=2)
dot, = a3.plot([], [], "o", color=BLUE, ms=10)
a3.set_xlim(0, t[-1]); pad = 0.15 * np.abs(y).max()
a3.set_ylim(y.min() - pad, y.max() + pad); a3.set_yticks([0])
a3.set_title("3  slide, repeat: one output per position (the feature map)", loc="left")
a3.set_xlabel("time (s)")
for tt in (meta["t_p"], meta["t_s"]):
    a3.axvline(tt, color=INK, ls=":", lw=1.5)

steps = np.arange(0, y.size, 40)
frames = list(steps) + [steps[-1]] * 12


scale = 3.5 / np.abs(w * x[y.argmax():y.argmax() + nk]).max()   # products drawn on a fixed scale


def draw1(k):
    seg = x[k:k + nk]
    prod = w * seg
    win.set_x(k / FS)
    kline.set_data(t[k:k + nk], w * 12)
    seg_line.set_data(tz, seg)
    for b, p in zip(bars, prod):
        b.set_height(p * scale)
        b.set_color(GREEN if p >= 0 else RED)
    title2.set_text(f"2  multiply each weight by the sample under it, add: sum = {prod.sum():+.2f}")
    out_line.set_data(ty[:k + 1], y[:k + 1])
    dot.set_data([ty[k]], [y[k]])
    return []


anim = FuncAnimation(fig, draw1, frames=frames, blit=False)
save(anim, fig, "conv_slide_multiply_sum", fps=10)

# ============================================================== 2. same weights everywhere
dur = 30.0
n = int(dur * FS)
_, clean, m2 = synthetic_seismogram(duration_s=dur, fs=FS, magnitude=2.5, distance_km=20.0, f0=5.0, snr=np.inf, seed=1)
_, noise, _ = synthetic_seismogram(duration_s=dur, fs=FS, magnitude=-10, snr=1e-12, seed=11)
noise = noise / noise.std() * (np.abs(clean).max() / 5.0)        # SNR 5, the generator's definition
tp0 = m2["t_p"]
fc_w = clean / np.linalg.norm(clean)                             # the neuron of lecture 21: template at its first position
shifts = np.arange(0.0, 13.01, 0.25)                             # P from 8.3 s (trained position) to 21.3 s
t2 = np.arange(n) / FS


base = clean + noise


def shifted(s):
    """The whole record (event and noise) moved later by s seconds, wrapping at the end."""
    return np.roll(base, int(round(s * FS)))


def fc_out(rec):
    return float(fc_w @ (rec / np.linalg.norm(rec)))


def conv_max(rec):
    return float(np.maximum(slide_sum(standardize(rec), w), 0).max())


fc_ref, cv_ref = fc_out(shifted(0.0)), conv_max(shifted(0.0))
fc_all = np.array([fc_out(shifted(s)) for s in shifts]) / fc_ref
cv_all = np.array([conv_max(shifted(s)) for s in shifts]) / cv_ref
numbers["gif2"] = {"distance_km": 20.0, "snr": 5.0, "t_p_original_s": tp0,
                   "shift_range_s": [float(shifts[0]), float(shifts[-1])],
                   "fc_relative_min_max": [float(fc_all.min()), float(fc_all.max())],
                   "fc_relative_at_shift_0.5s": float(fc_all[np.argmin(np.abs(shifts - 0.5))]),
                   "conv_max_relative_min_max": [float(cv_all.min()), float(cv_all.max())]}

fig = plt.figure(figsize=(16, 7.5))
gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 1], hspace=0.55, wspace=0.25, left=0.05, right=0.97, top=0.9, bottom=0.15)
b1, b2, b3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
rec_line, = b1.plot([], [], color=GRAY, lw=0.8)
p_mark = b1.axvline(0, color=INK, ls=":", lw=1.5)
b1.set_xlim(0, dur); b1.set_ylim(-5, 5); b1.set_yticks([])
b1.set_title("the same record, shifted later and later", loc="left")
fm_line, = b2.plot([], [], color=BLUE, lw=2)
fm_peak, = b2.plot([], [], "v", color=BLUE, ms=14)
b2.set_xlim(0, dur); b2.set_ylim(0, 12); b2.set_yticks([0])
b2.set_title("feature map: same 60 weights at every position, then ReLU", loc="left")
b2.set_xlabel("time (s)")
bar_art = b3.bar([0, 1], [1, 1], color=[RED, BLUE], width=0.6)
b3.set_xticks([0, 1]); b3.set_xticklabels(["fully\nconnected", "convolution\n+ max pool"], fontsize=16)
b3.set_ylim(0, 1.25); b3.set_ylabel("output / output at first position")
b3.axhline(1, color=MUTED, ls="--", lw=1)
b3.set_title("one number out", loc="left")
txt = b3.text(0.5, 1.15, "", ha="center", fontsize=17)
frames2 = [0] * 10 + list(range(len(shifts))) + [len(shifts) - 1] * 10


def draw2(i):
    rec = shifted(shifts[i])
    rs = standardize(rec)
    fm = np.maximum(slide_sum(rs, w), 0)
    tf = (np.arange(fm.size) + nk / 2) / FS
    rec_line.set_data(t2, rs)
    p_mark.set_xdata([tp0 + shifts[i]] * 2)
    fm_line.set_data(tf, fm)
    fm_peak.set_data([tf[fm.argmax()]], [fm.max() + 0.6])
    bar_art[0].set_height(max(fc_all[i], 0)); bar_art[1].set_height(cv_all[i])
    txt.set_text(f"P at {tp0 + shifts[i]:.1f} s")
    return []


anim = FuncAnimation(fig, draw2, frames=frames2, blit=False)
save(anim, fig, "conv_same_weights_everywhere", fps=8)

# ============================================================== 3. a bank of kernels on a real record
EVENT_ID = "uw61939452"
ev = Client("USGS").get_events(eventid=EVENT_ID)[0]
origin = ev.preferred_origin()
st = Client("EARTHSCOPE").get_waveforms("UW", "SP2", "--", "HHZ", origin.time - 5, origin.time + 40)
tr = st[0]
assert tr.stats.sampling_rate == FS
r = tr.data.astype(float)
r -= r.mean()
r /= np.abs(r).max()
tr_t = np.arange(r.size) / FS - 5.0                              # seconds after origin time
numbers["gif3"] = {"event": EVENT_ID, "origin_utc": str(origin.time), "magnitude": ev.preferred_magnitude().mag,
                   "depth_km": origin.depth / 1000, "station": "UW.SP2..HHZ", "fs_hz": FS,
                   "peak_counts": int(np.abs(tr.data - tr.data.mean()).max())}

kernels = {
    "smooth: 1 s average": np.ones(100) / 100,
    "change: sample minus sample 0.05 s earlier": np.r_[-1.0, np.zeros(4), 1.0],
    "5 Hz wavelet, like a P wave": np.cos(2 * np.pi * 5 * (np.arange(61) / FS - 0.3)) * np.exp(-((np.arange(61) / FS - 0.3) ** 2) / (2 * 0.1**2)),
}
outs = {}
for name, kern in kernels.items():
    o = np.correlate(r, kern, mode="same")
    outs[name] = o / np.abs(o).max()
colors3 = [GREEN, RED, ORANGE]

fig = plt.figure(figsize=(16, 7.5))
gs = fig.add_gridspec(4, 2, width_ratios=[1, 6.5], hspace=0.75, wspace=0.08, left=0.03, right=0.98, top=0.92, bottom=0.09)
c0 = fig.add_subplot(gs[0, 1])
c0.plot(tr_t, r, color=GRAY, lw=0.7)
cursor0 = c0.axvline(tr_t[0], color=INK, lw=2)
c0.set_xlim(tr_t[0], tr_t[-1]); c0.set_yticks([])
c0.set_title("M3.9 Snoqualmie earthquake at UW.SP2, Seattle (real record, 36 km)", loc="left", fontsize=20)
fig.add_subplot(gs[0, 0]).axis("off")
lines3, curs3 = [], []
for j, (name, kern) in enumerate(kernels.items()):
    ak = fig.add_subplot(gs[j + 1, 0])
    ak.plot(np.arange(kern.size) / FS, kern / np.abs(kern).max(), color=colors3[j], lw=3,
            marker="o" if kern.size < 10 else None, ms=6)
    ak.axhline(0, color=MUTED, lw=0.6); ak.set_xticks([]); ak.set_yticks([]); ak.set_ylim(-1.2, 1.2)
    for s in ("left", "bottom"):
        ak.spines[s].set_visible(False)
    ao = fig.add_subplot(gs[j + 1, 1], sharex=c0)
    ln, = ao.plot([], [], color=colors3[j], lw=0.9)
    ao.set_ylim(-1.1, 1.1); ao.set_yticks([])
    ao.set_title(name, loc="left", fontsize=19, color=colors3[j])
    lines3.append(ln)
    curs3.append(ao.axvline(tr_t[0], color=INK, lw=2))
ao.set_xlabel("time after origin (s)")
steps3 = np.arange(0, r.size, 40)
frames3 = list(steps3) + [r.size - 1] * 12


def draw3(k):
    cursor0.set_xdata([tr_t[k]] * 2)
    for ln, cu, o in zip(lines3, curs3, outs.values()):
        ln.set_data(tr_t[:k + 1], o[:k + 1])
        cu.set_xdata([tr_t[k]] * 2)
    return []


anim = FuncAnimation(fig, draw3, frames=frames3, blit=False)
save(anim, fig, "conv_kernel_bank_real", fps=10)

# ============================================================== 4. 2-D: a 3x3 kernel on a temperature map
grid = pooch.retrieve(
    "https://github.com/pydata/xarray-data/raw/2baf0c22d9671a2058415f17e71b3fe06239a3dd/air_temperature.nc",
    known_hash="sha256:c606b89c35970a2983b914b76df4adbb409003ef34aa7cfd7f582e41f307482b",
    fname="air_temperature_raw.nc", path=CACHE, progressbar=False)
coast = gpd.read_file(pooch.retrieve(
    "https://naciscdn.org/naturalearth/50m/physical/ne_50m_coastline.zip",
    known_hash="sha256:640f805509b822f57f4840a2e18d9ff2412cf1cf6976124701c2789436166fde",
    fname="ne_50m_coastline.zip", path=CACHE, progressbar=False))
WHEN = "2014-01-07T00:00"
field = xr.open_dataset(grid)["air"].sel(time=WHEN) - 273.15
lat, lon = field.lat.values, field.lon.values - 360.0          # lat runs north to south
T = field.values
K = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 4.0          # south row minus north row, 5 degrees apart
ny, nx = T.shape[0] - 2, T.shape[1] - 2
G = np.full((ny, nx), np.nan)
Gfull = np.array([[np.sum(K * T[i:i + 3, j:j + 3]) for j in range(nx)] for i in range(ny)])
imax = np.unravel_index(np.argmax(Gfull), Gfull.shape)
numbers["gif4"] = {"time_utc": WHEN, "grid_deg": 2.5, "kernel": K.tolist(),
                   "max_K_per_5deg": float(Gfull.max()),
                   "max_at_lat_lon": [float(lat[imax[0] + 1]), float(lon[imax[1] + 1])],
                   "min_temperature_c": float(T.min()), "max_temperature_c": float(T.max())}

fig = plt.figure(figsize=(16, 7.5))
gs = fig.add_gridspec(1, 3, width_ratios=[2.4, 1.0, 2.4], wspace=0.12, left=0.04, right=0.97, top=0.88, bottom=0.12)
d1, d2, d3 = (fig.add_subplot(gs[i]) for i in range(3))
ext = [lon[0] - 1.25, lon[-1] + 1.25, lat[-1] - 1.25, lat[0] + 1.25]
d1.imshow(T, extent=ext, cmap="RdYlBu_r", vmin=-40, vmax=30, aspect="auto")
coast.plot(ax=d1, color=INK, lw=0.6)
d1.set_aspect("auto"); d1.set_xlabel(""); d1.set_ylabel("")
d1.set_xlim(ext[0], ext[1]); d1.set_ylim(ext[2], ext[3])
d1.set_title("air temperature, 7 Jan 2014 (°C)", loc="left", fontsize=19)
box = plt.Rectangle((0, 0), 7.5, 7.5, fill=False, ec="k", lw=3)
d1.add_patch(box)
im3 = d3.imshow(G, extent=[lon[1] - 1.25, lon[-2] + 1.25, lat[-2] - 1.25, lat[1] + 1.25],
                cmap="PuOr_r", vmin=-25, vmax=25, aspect="auto")
coast.plot(ax=d3, color=INK, lw=0.6)
d3.set_aspect("auto"); d3.set_xlabel(""); d3.set_ylabel("")
d3.set_xlim(ext[0], ext[1]); d3.set_ylim(ext[2], ext[3])
d3.set_title("output: south minus north (°C per 5°)", loc="left", fontsize=19)
cb = fig.colorbar(im3, ax=d3, orientation="horizontal", fraction=0.05, pad=0.08)
cb.ax.tick_params(labelsize=14)
dot3 = d3.add_patch(plt.Rectangle((0, 0), 2.5, 2.5, fill=False, ec="k", lw=3))
for a in (d1, d3):
    a.set_xticks([]); a.set_yticks([])
d2.axis("off")
d2.set_xlim(0, 3); d2.set_ylim(0, 8.6)
d2.text(1.5, 8.3, "under the box (°C)", ha="center", fontsize=16, weight="bold")
d2.text(1.5, 4.9, "× weights", ha="center", fontsize=16, weight="bold")
cells_T = [[d2.text(c + 0.5, 7.4 - rr * 0.8, "", ha="center", va="center", fontsize=17) for c in range(3)] for rr in range(3)]
for rr in range(3):
    for c in range(3):
        v = K[rr, c]
        d2.text(c + 0.5, 4.0 - rr * 0.8, f"{v:+.2g}" if v else "0", ha="center", va="center", fontsize=17,
                color=BLUE if v > 0 else (RED if v < 0 else MUTED))
sum_txt = d2.text(1.5, 0.9, "", ha="center", fontsize=19, weight="bold")
order = [(i, j) for i in range(2) for j in range(0, nx, 5)] + [(i, nx - 1) for i in range(2, ny)]
frames4 = order + [order[-1]] * 12


def draw4(ij):
    i, j = ij
    if j == nx - 1 or (i, j) == order[-1]:
        G[: i + 1] = Gfull[: i + 1]
    else:
        G[:i] = Gfull[:i]
        G[i, : j + 1] = Gfull[i, : j + 1]
    im3.set_data(G)
    box.set_xy((lon[j] - 1.25, lat[i + 2] - 1.25))
    dot3.set_xy((lon[j + 1] - 1.25, lat[i + 1] - 1.25))
    for rr in range(3):
        for c in range(3):
            cells_T[rr][c].set_text(f"{T[i + rr, j + c]:.0f}")
    sum_txt.set_text(f"sum = {Gfull[i, j]:+.1f}")
    return []


anim = FuncAnimation(fig, draw4, frames=frames4, blit=False)
save(anim, fig, "conv_2d_temperature_front", fps=6)

# ============================================================== 5. receptive field of the 4.3 detector
layers = [("input", None), ("convolution 1, kernel 7", ("conv", 7, 1)), ("max pooling 1, by 4", ("pool", 4, 4)),
          ("convolution 2, kernel 7", ("conv", 7, 1)), ("max pooling 2, by 4", ("pool", 4, 4)),
          ("convolution 3, kernel 7", ("conv", 7, 1)), ("max pooling 3, by 4", ("pool", 4, 4))]
rf, jump, rfs = 1, 1, []
for name, spec in layers:
    if spec is not None:
        _, k, s = spec
        rf += (k - 1) * jump
        jump *= s
    rfs.append(rf)
rf_s = np.array(rfs) / FS
sp_min, sp_max = 5 * (1 / 3.5 - 1 / 6), 80 * (1 / 3.5 - 1 / 6)
numbers["gif5"] = {"layers": [l[0] for l in layers], "receptive_field_samples": rfs,
                   "receptive_field_s": rf_s.tolist(), "s_minus_p_range_s": [sp_min, sp_max]}

_, ev5, m5 = synthetic_seismogram(duration_s=30.0, fs=FS, magnitude=2.5, distance_km=50.0, f0=5.0, snr=6.0, seed=3)
ev5 = standardize(ev5)
t5 = np.arange(ev5.size) / FS
fig = plt.figure(figsize=(16, 7.5))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5], hspace=0.45, left=0.27, right=0.97, top=0.9, bottom=0.1)
e1, e2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
e1.plot(t5, ev5, color=GRAY, lw=0.8)
for tt, lab in [(m5["t_p"], "P"), (m5["t_s"], "S")]:
    e1.axvline(tt, color=INK, ls=":", lw=1.5)
    e1.text(tt + 0.15, 3.8, lab, fontsize=18, weight="bold")
span = e1.add_patch(plt.Rectangle((m5["t_p"], -5), 0, 10, color=BLUE, alpha=0.3))
e1.set_xlim(8, 24); e1.set_ylim(-5, 5); e1.set_yticks([])
e1.set_title(f"synthetic event at 50 km: S − P = {m5['t_s'] - m5['t_p']:.1f} s", loc="left")
e1.set_xlabel("time (s)")
e2.axvspan(sp_min, sp_max, color=ORANGE, alpha=0.18)
e2.text(sp_max - 0.1, 0, f"S − P in the training set: {sp_min:.1f} to {sp_max:.1f} s",
        ha="right", va="center", fontsize=17, color="#9a6a00", weight="bold")
bars5 = e2.barh(range(len(layers)), [0] * len(layers), color=BLUE, height=0.6)
labs5 = [e2.text(0, i, "", va="center", fontsize=16) for i in range(len(layers))]
e2.set_yticks(range(len(layers))); e2.set_yticklabels([l[0] for l in layers], fontsize=16)
e2.invert_yaxis(); e2.set_xlim(0, 10); e2.set_xlabel("span of input one unit sees (s)")
e2.set_title("one unit of each layer sees ...", loc="left")
frames5 = [i for i in range(len(layers)) for _ in range(7)] + [len(layers) - 1] * 14


def draw5(i):
    for q in range(len(layers)):
        on = q <= i
        bars5[q].set_width(rf_s[q] if on else 0)
        labs5[q].set_x(rf_s[q] + 0.1)
        labs5[q].set_text(f"{rf_s[q]:.2f} s ({rfs[q]} samples)" if on else "")
    span.set_x(m5["t_p"] - rf_s[i] / 2)
    span.set_width(rf_s[i])
    return []


anim = FuncAnimation(fig, draw5, frames=frames5, blit=False)
save(anim, fig, "conv_receptive_field", fps=6)

(FIGS / "numbers.json").write_text(json.dumps(numbers, indent=2, default=float))
print(json.dumps(numbers, indent=2, default=float))
