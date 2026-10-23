"""Slide figure: how much of each discipline's raw archive exists as AI-ready
(machine-learning benchmark) datasets, by volume.

Every number below is hard-coded from a cited source or from a size query made
on 2026-10-23 (Zenodo records API file sizes, Hugging Face usedStorage, HTTP
Content-Length on the SeisBench GFZ mirror and on PANGAEA, Google Cloud Storage
object listing, DVC size fields, zarr .zmetadata shapes). Nothing is fetched at
run time. These are order-of-magnitude estimates: archive sizes come from
different years and count different things (some include SAR or backups, some
do not), and the AI-ready sum covers only the named datasets, not every
benchmark in the discipline. The archive and the AI-ready datasets are not
always the same source data (e.g. SeisBench holds Italian and Chilean data
that are not in the EarthScope archive).

Units: TB = 1e12 bytes, PB = 1e15 bytes.

Writes figs/lec10_ai_ready/ai_ready_vs_archive.png
Regenerate: pixi run python book/slides/2026/figscripts/ai_ready_vs_archive.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
})
OUT = Path(__file__).resolve().parents[1] / "figs" / "lec10_ai_ready"
OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED, ACCENT, GOOD = "#26241f", "#6e675c", "#c0392b", "#1f6f6b"
ARCH_C = "#b9b2a6"

# ---------------------------------------------------------------------------
# Sourced numbers. Volumes in TB.
#
# discipline, archive label, archive TB, archive source (as-of),
# AI-ready label, AI-ready TB, number of datasets summed, AI-ready sources
#
# SEISMOLOGY
#   archive 1.3 PB: EarthScope + NCEDC + SCEDC continuous waveforms, 47,354
#     stations, 2002-01-01 to 2025-03-31. Ni et al. (2025) arXiv:2505.18874,
#     https://arxiv.org/abs/2505.18874 (2025). EarthScope alone: "over a
#     petabyte", https://www.earthscope.org/news/cloud-case-study-mining-the-
#     seismic-archive-for-a-new-earthquake-catalog/ (3 Jul 2025).
#   AI-ready 5.59 TB: all 37 non-dummy datasets on the SeisBench mirror,
#     https://seisbench.gfz.de/mirror/datasets/ , summed Content-Length,
#     2026-10-23 (STEAD 91.5 GB, INSTANCE counts 168 GB, CREW 1.15 TB, SCEDC
#     685 GB, CEED 617 GB, ...). Includes counts and ground-motion duplicates.
# GNSS GEODESY
#   archive 0.30 PB: NSF GAGE "archive of over 300 terabytes" (GPS, laser
#     scanning, SAR, borehole), NSF FY2021 budget request, https://nsf-gov-
#     resources.nsf.gov/about/budget/fy2021/pdf/40e_fy2021.pdf (2020).
#   AI-ready 0.75 GB: ChangePointCNN-GNSS archive (code, model, ~6,000 labeled
#     time-series plots), https://zenodo.org/records/17179942 (2025). The only
#     GNSS machine-learning training set found; a lower bound.
# REANALYSIS (ERA5)
#   archive ~5 PB: "The total ERA5 dataset is about 5 petabytes in size",
#     ARCO-ERA5 README, https://github.com/google-research/arco-era5
#     (read 2026-10-23). Context: ECMWF MARS held ~600 PB primary data in 2025,
#     https://www.ecmwf.int/node/31124 (23 Mar 2026).
#   AI-ready 91.6 TB: WeatherBench 2 ERA5 stores 1959-2023 6-hourly at 0.25 deg
#     (89.0 TB), 1.5 deg (2.49 TB) and 64x32 (0.18 TB); uncompressed size from
#     zarr .zmetadata, gs://weatherbench2/datasets/era5/, 2026-10-23.
#     https://weatherbench2.readthedocs.io/en/latest/data-guide.html
# CLIMATE MODELS (CMIP6)
#   archive 20 PB: "a total 20 Petabytes produced across the project", CEDA,
#     https://www.ceda.ac.uk/news/updates/2021/2021-10-05-cmip-and-esgf/ (2021).
#   AI-ready 0.097 TB: ClimateBench 2.39 GB, https://zenodo.org/records/7064308
#     (2021); ClimateSet 94.1 GB, https://huggingface.co/datasets/climateset/
#     climateset (2024).
# SATELLITE REMOTE SENSING (Sentinel)
#   archive 78 PB: "More than 78 PB of online data", Copernicus Data Space
#     Ecosystem annual report 2024, https://dataspace.copernicus.eu/node/2872
#     (2024). Context: NASA EOSDIS 178.7 PB, FY2025,
#     https://earthdata.nasa.gov/about/data-metrics
#   AI-ready 63.3 TB: Major TOM Core S2L2A 22.96 + S2L1C 23.55 + S1RTC 15.69 +
#     DEM 0.97 TB (Hugging Face usedStorage, huggingface.co/Major-TOM, 2024);
#     BigEarthNet v2 118 GB, https://zenodo.org/records/10891137 (2024);
#     Sen1Floods11 36.9 GB, gs://sen1floods11 listing (2026-10-23).
# OCEANOGRAPHY
#   archive 0.315 PB: PO.DAAC "nearly 315 terabytes of data at the end of
#     FY2018", https://www.earthdata.nasa.gov/sites/default/files/2022-01/
#     PODAAC05302019_FactSheet.pdf (FY2018). Before SWOT (>20 PB expected).
#   AI-ready 42.0 GB: OceanBench SSH edition, DVC size fields for ose/ and
#     osse_natl60/, https://github.com/quentinf00/oceanbench-data-registry
#     (2023).
# CRYOSPHERE
#   archive 0.619 PB: NSIDC DAAC "More than 619 terabytes of data in the
#     archive at the end of FY2018", https://www.earthdata.nasa.gov/s3fs-
#     public/2022-01/NSIDCDAAC22719_FactSheet.pdf (FY2018).
#   AI-ready 65.6 GB: AI4Arctic ready-to-train 60.2 GB + test 2.55 GB,
#     https://doi.org/10.11583/DTU.21316608 , https://doi.org/10.11583/
#     DTU.21762830 (2023); CaFFe 2.86 GB, https://doi.org/10.1594/
#     PANGAEA.940950 (2022).
# HYDROLOGY
#   archive: no published byte size found for USGS NWIS (unverified).
#   AI-ready 55.8 GB: CAMELS v1.2 14.57 GB, https://zenodo.org/records/15529996
#     (2022); Caravan v1.6 24.78 GB, https://zenodo.org/records/15529786
#     (2025); GRDC-Caravan 16.42 GB, https://zenodo.org/records/15349031 (2025).
# ---------------------------------------------------------------------------
ROWS = [
    # name,                      archive label,           arch TB, as-of,   AI-ready TB, n
    ("Seismology",               "EarthScope+NCEDC+SCEDC", 1300.0,  "2025",   5.59,    37),
    ("Reanalysis\n(ERA5)",       "ERA5",                   5000.0,  "README", 91.6,    3),
    ("Satellite RS\n(Sentinel)", "Copernicus CDSE",        78000.0, "2024",   63.3,    6),
    ("Ocean",                    "NASA PO.DAAC",           315.0,   "FY2018", 0.0420,  1),
    ("Cryosphere",               "NASA NSIDC DAAC",        619.0,   "FY2018", 0.0656,  2),
    ("Climate models\n(CMIP6)",  "CMIP6 / ESGF",           20000.0, "2021",   0.0965,  2),
    ("GNSS\ngeodesy",            "NSF GAGE",               300.0,   "2020",   0.00075, 1),
    ("Hydrology",                "USGS NWIS",              np.nan,  "",       0.0558,  3),
]


def fmt_vol(tb):
    if tb >= 1000:
        return f"{tb / 1000:.3g} PB"
    if tb >= 1:
        return f"{tb:.3g} TB"
    return f"{tb * 1000:.3g} GB"


def fmt_ratio(r):
    pct = 100 * r
    if pct >= 0.1:
        return f"{pct:.2g}%"
    return f"{pct:.1g}%"


names = [r[0] for r in ROWS]
arch = np.array([r[2] for r in ROWS])
ready = np.array([r[4] for r in ROWS])
x = np.arange(len(ROWS))
w = 0.38

fig, ax = plt.subplots(figsize=(24, 8.9))
ax.bar(x - w / 2, np.nan_to_num(arch, nan=0) + 1e-12, w, color=ARCH_C, edgecolor=MUTED,
       label="raw public archive")
ax.bar(x + w / 2, ready, w, color=GOOD, label="AI-ready / benchmark datasets (sum)")
ax.set_yscale("log")
ax.set_ylim(3e-4, 4e6)
ticks = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5]
ax.set_yticks(ticks)
ax.set_yticklabels(["1 GB", "10 GB", "100 GB", "1 TB", "10 TB", "100 TB", "1 PB", "10 PB", "100 PB"])
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=17)
ax.set_ylabel("volume (log scale)")
ax.grid(axis="y", color="#e4e0d8", lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

for i, r in enumerate(ROWS):
    a, b, n = r[2], r[4], r[5]
    ax.text(x[i] + w / 2, b * 1.35, f"{fmt_vol(b)}\n{n} set{'s' if n > 1 else ''}",
            ha="center", va="bottom", fontsize=13.5, color=GOOD)
    if np.isfinite(a):
        ax.text(x[i] - w / 2, a * 1.35, f"{fmt_vol(a)}\n{r[3]}", ha="center", va="bottom",
                fontsize=13.5, color=MUTED)
        ax.text(x[i], 1.3e6, fmt_ratio(b / a), ha="center", va="center", fontsize=21,
                fontweight="bold", color=ACCENT)
    else:
        ax.text(x[i] - w / 2, 1e-3, "archive size\nnot published", ha="center", va="bottom",
                fontsize=13.5, color=MUTED, rotation=90)
        ax.text(x[i], 1.3e6, "?", ha="center", va="center", fontsize=21, fontweight="bold",
                color=ACCENT)
ax.text(-0.75, 1.3e6, "AI-ready\n/ archive", ha="right", va="center", fontsize=15, color=ACCENT)
ax.set_xlim(-0.9, len(ROWS) - 0.5)
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86), fontsize=16, frameon=False)
ax.set_title("Raw archive volume and AI-ready dataset volume, by discipline", loc="left", pad=14)

fig.text(0.01, -0.03,
         "Order-of-magnitude estimates from the cited sources (URLs and dates in the script header). "
         "Archive sizes come from different years and scopes; the AI-ready bar sums only the named "
         "datasets.\nSeismology: SeisBench mirror (37 sets). Reanalysis: WeatherBench 2 ERA5 subsets. "
         "Satellite: Major TOM Core, BigEarthNet v2, Sen1Floods11. Ocean: OceanBench SSH. "
         "Cryosphere: AI4Arctic, CaFFe.\nClimate models: ClimateBench, ClimateSet. "
         "GNSS: ChangePointCNN-GNSS (only set found). Hydrology: CAMELS, Caravan, GRDC-Caravan. "
         "Volumes in decimal units (1 PB = 1000 TB).",
         fontsize=13, color=MUTED, ha="left", va="top")
fig.savefig(OUT / "ai_ready_vs_archive.png", dpi=110, bbox_inches="tight", facecolor="white")
print("wrote", OUT / "ai_ready_vs_archive.png")
for r in ROWS:
    ratio = r[4] / r[2] if np.isfinite(r[2]) else np.nan
    print(f"{r[0].replace(chr(10), ' '):28s} archive {r[2]:>9} TB  ai-ready {r[4]:>9} TB  ratio {ratio:.2e}")
