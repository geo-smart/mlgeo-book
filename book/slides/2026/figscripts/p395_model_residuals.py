"""Slide figures: P395 observation, NGL's trajectory model, a synthetic twin, and the residual.

Builds intuition for measurement error versus model error. Everything is
fetched from the Nevada Geodetic Laboratory at run time:
  daily positions     IGS20 tenv3 (east component)
  trajectory model    IGS20 .modfit: rate, annual and semi-annual terms, steps,
                      and the residual RMS NGL reports for its own fit
  step catalogue      steps.txt: code 1 = equipment change, code 2 = possible
                      earthquake step (with USGS event ID)
The model's absolute intercept convention is not documented in the file, so
the model is aligned to the data by its mean residual (one constant); the
residual RMS then reproduces NGL's reported value.

The synthetic twin is the model plus independent Gaussian noise with NGL's
residual RMS (fixed seed): what the record would look like if the model were
complete and the errors were independent from day to day.

Regenerate: pixi run python book/slides/2026/figscripts/p395_model_residuals.py
"""
import datetime as dt
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pooch  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 22, "axes.titleweight": "bold",
    "axes.labelsize": 19, "xtick.labelsize": 16, "ytick.labelsize": 16,
})

STA = "P395"
NGL = "https://geodesy.unr.edu"
CACHE = pooch.os_cache("mlgeo-figscripts")
INK, MUTED, GRID = "#26241f", "#6e675c", "#d9d4ca"
OBS, MODEL, SYN = "#2a78d6", "#eb6834", "#1baf7a"     # dataviz reference slots 1-3
OUT = Path(__file__).resolve().parent.parent / "figs" / "1.7_get_geodetic_gnss"


def fetch(url, name):
    return pooch.retrieve(url, known_hash=None, fname=name, path=CACHE, progressbar=False)


def modfit():
    text = open(fetch(f"{NGL}/gps_timeseries/IGS20/models/IGS20/original/model/{STA}.modfit",
                      f"{STA}.IGS20.modfit")).read()
    num = r"(-?\d+\.\d+)"
    rate = float(re.search(rf"Vel East\s*:\s*{num}", text).group(1))
    c1, s1 = map(float, re.search(rf"CS1 East\s*:\s*{num}\+/-\s*\S+\s+{num}", text).groups())
    c2, s2 = map(float, re.search(rf"CS2 East\s*:\s*{num}\+/-\s*\S+\s+{num}", text).groups())
    steps = [(float(t), float(e)) for t, e in
             re.findall(rf"Step: \d+ {num}\s*\nStep East\s*:\s*{num}", text)]
    rms = float(re.search(rf"RMS East\s*:\s*{num}", text).group(1))
    return rate, (c1, s1, c2, s2), steps, rms


def step_catalogue():
    out = []
    for line in open(fetch(f"{NGL}/NGLStationPages/steps.txt", "ngl_steps.txt")):
        p = line.split()
        if p and p[0] == STA:
            day = dt.datetime.strptime(p[1], "%y%b%d")
            year = day.year + (day.timetuple().tm_yday - 0.5) / (366 if day.year % 4 == 0 else 365)
            out.append(dict(t=year, code=int(p[2]), what=p[3] if p[2] == "1" else f"M{p[5]} at {float(p[4]):.0f} km, USGS {p[6]}"))
    return out


def observations():
    d = pd.read_csv(fetch(f"{NGL}/gps_timeseries/IGS20/tenv3/IGS20/{STA}.tenv3", f"{STA}.IGS20.tenv3"), sep=r"\s+")
    return d["yyyy.yyyy"].to_numpy(), d["__east(m)"].to_numpy() * 1000.0


rate, (c1, s1, c2, s2), steps, rms = modfit()
catalogue = step_catalogue()
t, east = observations()


def model(tt, with_steps):
    m = (rate * (tt - tt[0]) + c1 * np.cos(2 * np.pi * tt) + s1 * np.sin(2 * np.pi * tt)
         + c2 * np.cos(4 * np.pi * tt) + s2 * np.sin(4 * np.pi * tt))
    for ts, amp in steps:
        if with_steps(ts):
            m = m + amp * (tt >= ts)
    return m


def kind(ts):
    """Catalogue code of the step nearest a model step epoch (within ~3 days)."""
    near = [c for c in catalogue if abs(c["t"] - ts) < 0.01]
    return near[0]["code"] if near else None


full = model(t, lambda ts: True)
offset = np.mean(east - full)                      # the one alignment constant
full += offset
no_equipment = model(t, lambda ts: kind(ts) != 1) + offset
rng = np.random.default_rng(395)
synthetic = full + rng.normal(0.0, rms, t.size)
resid_obs = east - no_equipment
resid_syn = synthetic - full
print(f"residual RMS vs full model: {np.std(east - full):.3f} mm (NGL reports {rms})")
retrieved = dt.date.today().isoformat()
tag = (f"GNSS station {STA}, east component · NGL IGS20 daily positions and trajectory model "
       f"(.modfit), retrieved {retrieved}")

# ---------- figure 1: observation, model, synthetic twin
fig = plt.figure(figsize=(16, 8.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.45, wspace=0.12)
a = fig.add_subplot(gs[0, :])
a.plot(t, east, ".", ms=2.5, color=OBS, alpha=0.55, rasterized=True, label="observation")
a.plot(t, full, color=MODEL, lw=2.2, label="NGL model: trend + seasonal + steps")
a.set_title("Observation and NGL's trajectory model; shaded window below", loc="left")
a.set_ylabel("east (mm)")
a.legend(loc="upper right", fontsize=15, markerscale=5)
win = (2015.5, 2017.5)
a.axvspan(*win, color=GRID, alpha=0.6, lw=0)
k = (t >= win[0]) & (t <= win[1])
lo, hi = np.percentile(np.r_[east[k], synthetic[k]], [0.5, 99.5])
# Neutral titles: the slide asks students which series is synthetic, then reveals it.
for col, (y, color, name) in enumerate([(synthetic, OBS, "Series 1"), (east, OBS, "Series 2")]):
    b = fig.add_subplot(gs[1, col])
    b.plot(t[k], y[k], ".", ms=4, color=color, alpha=0.8)
    b.plot(t[k], full[k], color=MODEL, lw=1.6)
    b.set_ylim(lo - 1, hi + 1)
    b.set_title(name, loc="left", fontsize=18)
    b.set_xlabel("year")
    if col == 0:
        b.set_ylabel("east (mm)")
    else:
        b.set_yticklabels([])
for ax in fig.axes:
    ax.grid(color=GRID, lw=0.8); ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(tag, x=0.01, ha="left", fontsize=13, color=MUTED)
fig.savefig(OUT / "p395_model_synthetic_slide.png", dpi=130, bbox_inches="tight", pad_inches=0.08)

# ---------- figure 2: residuals with lettered features
def running_median(y, n=61):
    return pd.Series(y).rolling(n, center=True, min_periods=n // 2).median().to_numpy()


fig, (a, b) = plt.subplots(2, 1, figsize=(16, 8.4), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1.0], hspace=0.25))
a.plot(t, resid_obs, ".", ms=2.5, color=OBS, alpha=0.45, rasterized=True)
a.plot(t, running_median(resid_obs), color=INK, lw=1.8)
b.plot(t, resid_syn, ".", ms=2.5, color=SYN, alpha=0.45, rasterized=True)
b.plot(t, running_median(resid_syn), color=INK, lw=1.8)
ylim = np.percentile(resid_obs, [0.3, 99.7]) + np.array([-2, 4])
a.set_ylim(*ylim); b.set_ylim(*ylim)
for c in catalogue:
    style = dict(color=MODEL, lw=1.6, ls="--") if c["code"] == 1 else dict(color=MUTED, lw=1.6, ls=":")
    a.axvline(c["t"], **style)
labels = {"A": (2007.6, None), "B": (2010.46, None), "C": (2012.83, None), "D": (2016.37, None), "E": (2021.5, None)}
for letter, (x, _) in labels.items():
    a.text(x, ylim[1] - 0.5, letter, ha="center", va="top", fontsize=24, fontweight="bold", color=INK,
           bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec=INK, lw=1.5))
a.set_title("Observation minus model (equipment steps left in)", loc="left")
b.set_title("Synthetic minus model", loc="left", fontsize=18)
a.set_ylabel("residual (mm)"); b.set_ylabel("(mm)"); b.set_xlabel("year")
for ax in (a, b):
    ax.grid(color=GRID, lw=0.8); ax.spines[["top", "right"]].set_visible(False)
a.plot([], [], color=MODEL, ls="--", label="equipment change (NGL steps.txt)")
a.plot([], [], color=MUTED, ls=":", label="possible earthquake step (NGL steps.txt)")
a.plot([], [], color=INK, lw=1.8, label="61-day running median")
a.legend(loc="lower left", fontsize=14, ncol=3, frameon=True, framealpha=0.95)
fig.suptitle(tag, x=0.01, ha="left", fontsize=13, color=MUTED)
fig.savefig(OUT / "p395_residuals_slide.png", dpi=130, bbox_inches="tight", pad_inches=0.08)

print("steps in model (epoch, east mm, catalogue code):", [(ts, amp, kind(ts)) for ts, amp in steps])
print("catalogue:", catalogue)
print(f"wrote {OUT / 'p395_model_synthetic_slide.png'} and {OUT / 'p395_residuals_slide.png'}")
