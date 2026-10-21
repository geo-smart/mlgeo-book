"""Slide figure (lecture 9): the shapes values take, synthetic and real.

Top: 20,000 samples (fixed seed) from five distributions, each with mean and
median marked and its sample excess kurtosis (0 for a Gaussian) written, with
the population value in brackets (Gaussian 0, Laplace 3, Student t with 3
degrees of freedom infinite, log-normal with sigma = 1 about 111, Pareto with
alpha = 1.5 undefined because its variance is infinite); the long-tailed ones
are shown to their 99.5th percentile, log-normal and Pareto on a log x axis.
  Gaussian            N(0, 1)
  Laplace             scale 1/sqrt(2) (unit variance), heavier tails than Gaussian
  Student t           3 degrees of freedom (finite mean, finite variance,
                      infinite kurtosis)
  log-normal          exp(N(0, 1)): skewed, all positive
  Pareto (power law)  x_min = 1, tail exponent alpha = 1.5: P(X > x) = x^-1.5,
                      finite mean, infinite variance
Bottom left: complementary cumulative distribution P(|X| > x) of each sample on
log-log axes; a power law is a straight line there.
Bottom middle: seismic moment of earthquakes in western Washington and northern
Oregon (45.5-49 N, 121-124 W, Portland to the Canadian border, Mount St. Helens
included), 2000-2025,
from the course copy of ComCat (snapshots/comcat_puget_2000-2025.csv), above the
magnitude of completeness Mc. Maximum curvature fails here (the box holds the
2004-2008 Mount St. Helens swarm and mixes ml and md), so Mc is the smallest
magnitude, on a 0.1 grid, above which the Aki-Utsu b-value changes by less than
0.03 over the next 0.4 magnitude units (b-value stability, after Cao and Gao
2002 and Woessner and Wiemer 2005). Moment is computed from
the catalog magnitude with M0 = 10^(1.5 M + 9.1) N m (Hanks and Kanamori 1979);
the catalog magnitudes are local (ml) and duration (md) magnitudes, so this is
the moment those magnitudes imply, not a measured moment. The Gutenberg-Richter
b-value (Aki-Utsu maximum likelihood) gives the expected CCDF slope -2b/3.
Bottom right: daily precipitation at Seattle-Tacoma airport (GHCN-Daily
USW00024233, course copy snapshots/ghcnd_USW00024233_1948-2024.csv) on wet
days (>= 0.3 mm), with an exponential fitted to the mean for comparison.

Regenerate: pixi run python book/slides/2026/figscripts/lec09_distributions.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

plt.rcParams.update({"font.size": 17, "axes.titlesize": 19, "axes.titleweight": "bold"})
INK, MUTED = "#26241f", "#6e675c"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # dataviz reference slots 1-5
SEED, N = 2026, 20_000
ALPHA = 1.5
SNAP = Path(__file__).resolve().parents[3] / "Chapter2-DataManipulation" / "snapshots"

rng = np.random.default_rng(SEED)
DISTS = [
    ("Gaussian", rng.standard_normal(N)),
    ("Laplace", rng.laplace(0, 1 / np.sqrt(2), N)),
    ("Student t, ν = 3", rng.standard_t(3, N)),
    ("log-normal", np.exp(rng.standard_normal(N))),
    ("Pareto, α = 1.5", (1 - rng.random(N)) ** (-1 / ALPHA)),
]


def hill(x, frac=0.02):
    """Hill estimator of the tail exponent from the largest frac of |x|."""
    a = np.sort(np.abs(x))[::-1]
    k = max(int(frac * a.size), 10)
    return 1 / np.mean(np.log(a[:k] / a[k]))


def describe(x):
    s = x.std()
    return dict(mean=x.mean(), median=np.median(x), std=s, skew=stats.skew(x), kurt=stats.kurtosis(x),
                beyond3=np.mean(np.abs(x - x.mean()) > 3 * s), hill=hill(x - np.median(x)))


desc = {name: describe(x) for name, x in DISTS}
POP_KURT = ["0", "3", "∞", f"{np.exp(4) + 2 * np.exp(3) + 3 * np.exp(2) - 6:.0f}", "undef."]   # population values

# ---------------------------------------------------------------- real data
cat = pd.read_csv(SNAP / "comcat_puget_2000-2025.csv", usecols=["time", "mag", "magType"]).dropna(subset=["mag"])
mags = cat["mag"].to_numpy()
bins = np.arange(np.floor(mags.min() * 10) / 10, mags.max() + 0.1, 0.1)
counts, _ = np.histogram(mags, bins)
mc_maxc = round(bins[np.argmax(counts)] + 0.2, 1)
dm = 0.01                                            # magnitudes are served to 0.01


def b_aki(mc):
    return np.log10(np.e) / (mags[mags >= mc - 1e-9].mean() - (mc - dm / 2))


grid = np.round(np.arange(0.5, 3.01, 0.1), 1)
b_grid = np.array([b_aki(g) for g in grid])
mc = next(g for i, g in enumerate(grid[:-4]) if np.ptp(b_grid[i:i + 5]) < 0.03)
m_ok = mags[mags >= mc - 1e-9]
b = b_aki(mc)
m0 = 10 ** (1.5 * m_ok + 9.1)

wx = pd.read_csv(SNAP / "ghcnd_USW00024233_1948-2024.csv", usecols=["date", "PRCP"])
prcp_mm = wx["PRCP"].dropna().to_numpy() / 10.0
WET = 0.3
wet = prcp_mm[prcp_mm >= WET]


def ccdf(x):
    a = np.sort(x)
    return a, 1 - np.arange(a.size) / a.size


# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(16, 9.4))
gs = fig.add_gridspec(2, 5, height_ratios=[0.8, 1.15], hspace=0.36, wspace=0.28)
gsb = gs[1, :].subgridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.32)
for j, ((name, x), c) in enumerate(zip(DISTS, COLORS)):
    ax = fig.add_subplot(gs[0, j])
    d = desc[name]
    lo, hi = np.percentile(x, [0.25 if x.min() < 0 else 0, 99.5])
    logx = j >= 3                                       # log-normal and Pareto: log x axis
    edges = np.logspace(np.log10(lo), np.log10(hi), 61) if logx else 60
    ax.hist(x[(x >= lo) & (x <= hi)], bins=edges, color=c, edgecolor="white", lw=0.3)
    if logx:
        ax.set_xscale("log")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)                # room for the kurtosis label
    ax.axvline(d["mean"], color=INK, lw=2)
    ax.axvline(d["median"], color=INK, lw=2, ls=":")
    ax.set_yticks([])
    ax.set_title(name, loc="left", fontsize=16)
    ax.text(0.97, 0.95, (f"kurtosis {d['kurt']:.0f}" if abs(d['kurt']) >= 10 else f"kurtosis {d['kurt']:.1f}")
            + f"\n(population {POP_KURT[j]})",
            transform=ax.transAxes, ha="right", va="top", fontsize=13, color=INK,
            bbox=dict(facecolor="white", edgecolor="none", pad=1))
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    if j == 4:
        ax.legend(handles=[plt.Line2D([], [], color=INK, lw=2, label="mean"),
                           plt.Line2D([], [], color=INK, lw=2, ls=":", label="median")],
                  frameon=False, fontsize=13, loc="center right")

ax = fig.add_subplot(gsb[0])
for (name, x), c in zip(DISTS, COLORS):
    xs, p = ccdf(np.abs(x - (0 if x.min() >= 0 else np.median(x))))
    ax.loglog(xs, p, color=c, lw=2.4, label=name)
xr = np.logspace(0.3, 3, 10)
ax.loglog(xr, 2.2 * xr ** -ALPHA, color=INK, lw=1, ls="--")
ax.text(25, 2.2 * 25 ** -ALPHA * 2.5, "slope −1.5", fontsize=13, color=INK, ha="left", rotation=-33)
ax.set_xlim(1e-2, 3e3)
ax.set_ylim(1 / N, 1.3)
ax.set_xlabel("x (|x| for the symmetric ones)")
ax.set_ylabel("P(X > x)")
ax.legend(frameon=False, fontsize=12, loc="lower left", bbox_to_anchor=(0.0, 0.0), handlelength=1.2, labelspacing=0.3)
ax.set_title("synthetic, log-log", loc="left", fontsize=16)

axm = fig.add_subplot(gsb[1])
xs, p = ccdf(m0)
axm.loglog(xs, p, color=INK, lw=2.4)
axm.loglog(xs, (xs / xs[0]) ** (-2 * b / 3), color=MUTED, lw=1.2, ls="--")
axm.set_xlabel("seismic moment (N m)")
axm.set_ylabel("P(M₀ > x)")
axm.set_title("western WA and\nnorthern OR quakes", loc="left", fontsize=15)
axm.set_ylim(1 / m_ok.size / 2, 1.3)
axm.text(0.04, 0.04, f"M ≥ {mc:.1f}, n = {m_ok.size:,}\nb = {b:.2f}\nslope −2b/3 = {-2 * b / 3:.2f}",
         transform=axm.transAxes, ha="left", va="bottom", fontsize=13, color=INK)

axp = fig.add_subplot(gsb[2])
xs, p = ccdf(wet)
axp.loglog(xs, p, color=INK, lw=2.4, label="observed")
lam = 1 / (wet.mean() - WET)
axp.loglog(xs, np.exp(-lam * (xs - WET)), color=MUTED, lw=1.2, ls="--", label="exponential, same mean")
axp.set_ylim(1 / wet.size / 2, 1.3)
axp.set_xlabel("daily precipitation on wet days (mm)")
axp.set_ylabel("P(R > x)")
axp.legend(frameon=False, fontsize=13, loc="lower left")
axp.set_title("SeaTac rain, 1948–2024", loc="left", fontsize=16)
for a in (ax, axm, axp):
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)

out = Path(__file__).resolve().parent.parent / "figs" / "lec09_noise" / "distributions.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight", pad_inches=0.1)
print(f"wrote {out}")
print(f"seed {SEED}, n = {N} per distribution; Hill tail exponent from the largest 2 % of |x - median|")
print("distribution | mean | median | std | skewness | excess kurtosis | fraction beyond 3 std | Hill alpha")
for name, d in desc.items():
    print(f"{name:<17} | {d['mean']:.3f} | {d['median']:.3f} | {d['std']:.3f} | {d['skew']:.2f} | {d['kurt']:.2f} | "
          f"{100 * d['beyond3']:.2f} % | {d['hill']:.2f}")
print(f"Gaussian expectation beyond 3 std: {100 * 2 * stats.norm.sf(3):.2f} %")
pareto = DISTS[4][1]
print(f"Pareto alpha 1.5: true mean {ALPHA / (ALPHA - 1):.2f}, sample mean {pareto.mean():.3f}; running std at "
      f"n = 1000 / 5000 / 20000: {pareto[:1000].std():.2f} / {pareto[:5000].std():.2f} / {pareto.std():.2f} "
      "(no finite variance to converge to)")
print(f"ComCat Puget: {len(mags)} events with magnitude, magType counts {cat['magType'].value_counts().to_dict()}; "
      f"max curvature + 0.2 = {mc_maxc:.1f} (b there {b_aki(mc_maxc):.3f}); b-stability Mc = {mc:.1f}; "
      f"b on the grid 0.5-3.0: {dict(zip(grid.tolist(), np.round(b_grid, 3).tolist()))}; {m_ok.size} events >= Mc; Aki-Utsu b = {b:.3f} "
      f"(dm = {dm}); expected moment CCDF slope -2b/3 = {-2 * b / 3:.3f}; Hill alpha on moment (top 2 %) "
      f"{hill(m0):.3f}")
mom = describe(m0)
print(f"moment: mean {mom['mean']:.3e} N m, median {mom['median']:.3e} N m, mean/median {mom['mean'] / mom['median']:.1f}, "
      f"largest event carries {100 * m0.max() / m0.sum():.1f} % of the summed moment (M {m_ok.max():.2f})")
wd = describe(wet)
print(f"SeaTac PRCP: {prcp_mm.size} days with a value, {wet.size} wet days >= {WET} mm "
      f"({100 * wet.size / prcp_mm.size:.1f} %); wet-day mean {wd['mean']:.2f} mm, median {wd['median']:.2f} mm, "
      f"std {wd['std']:.2f}, skewness {wd['skew']:.2f}, excess kurtosis {wd['kurt']:.2f}, max {wet.max():.1f} mm on "
      f"{wx.loc[wx['PRCP'] == wx['PRCP'].max(), 'date'].iloc[0]}; Hill alpha (top 2 %) {wd['hill']:.2f}; "
      f"exponential (same mean) P(R > max) = {np.exp(-lam * (wet.max() - WET)):.2e} vs observed 1/{wet.size}")
