"""Slide figure: how often "AI-ready" appears in the literature, and how many
Earth-science datasets three public hosts return for a keyword search.

All counts are fetched at run time from public APIs, with no API key, no email
and no personal information in any request. They are keyword counts on the run
date, not a census: a phrase or keyword search misses works that use other
words and includes some that use the words in another sense.

  A  OpenAlex works API, phrase "AI-ready" in title or abstract
     (filter title_and_abstract.search), grouped by publication_year, 2015-2026.
     All fields, Earth and Planetary Sciences (primary_topic.field.id:19) and
     Environmental Science (field 23). The current year is partial.
     Also: works in OpenAlex citing the ESIP Data Readiness Cluster (2022)
     AI-readiness checklist, doi:10.6084/m9.figshare.19983722.
  B  Hugging Face datasets API (search=<term>, all pages counted),
     Zenodo records API (q=<term>, type=dataset, hits.total),
     GitHub repository search API (total_count for topic:<t> plus the word dataset).

Writes figs/lec10_ai_ready/bibliometrics.png and bibliometrics_counts.csv.
Regenerate: pixi run python book/slides/2026/figscripts/ai_ready_bibliometrics.py
"""
import datetime as dt
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pooch  # noqa: E402
import requests  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
})
CACHE = Path(pooch.os_cache("mlgeo-figscripts"))
TODAY = dt.date.today().isoformat()
OUT = Path(__file__).resolve().parents[1] / "figs" / "lec10_ai_ready"
OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED, ACCENT, GOOD, THIRD = "#26241f", "#6e675c", "#c0392b", "#1f6f6b", "#8a6d1f"
YEARS = list(range(2015, 2027))
ESIP_DOI = "10.6084/m9.figshare.19983722"
TERMS = ["seismic", "earthquake", "climate", "weather", "satellite", "ocean", "hydrology"]
GH_TOPICS = ["earth-science", "geoscience", "remote-sensing", "seismology", "climate", "hydrology"]
HDR = {"User-Agent": "mlgeo-figscripts (course figure script)"}


def cached_json(key, url, params=None):
    """GET a JSON document once per day and cache it under the run date."""
    path = CACHE / "ai_ready" / TODAY / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    r = requests.get(url, params=params, headers=HDR, timeout=120)
    r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(r.text)
    return r.json()


# ---------------- Panel A: OpenAlex ----------------
def openalex_by_year(field=None):
    flt = 'title_and_abstract.search:"AI-ready",publication_year:2015-2026'
    if field:
        flt += f",primary_topic.field.id:{field}"
    d = cached_json(f"openalex_field{field}", "https://api.openalex.org/works",
                    dict(filter=flt, group_by="publication_year"))
    got = {int(g["key"]): g["count"] for g in d["group_by"]}
    return [got.get(y, 0) for y in YEARS]


lit = pd.DataFrame({"year": YEARS,
                    "all_fields": openalex_by_year(),
                    "earth_planetary_sci": openalex_by_year(19),
                    "environmental_sci": openalex_by_year(23)})
esip = cached_json("openalex_esip", f"https://api.openalex.org/works/doi:{ESIP_DOI}")
esip_cites = cached_json("openalex_esip_cites", "https://api.openalex.org/works",
                         dict(filter=f"cites:{esip['id'].split('/')[-1]}", per_page=50))
n_esip = esip_cites["meta"]["count"]
print(f"OpenAlex, phrase 'AI-ready' in title/abstract, retrieved {TODAY}:")
print(lit.to_string(index=False))
print("totals 2015-2026:", lit[["all_fields", "earth_planetary_sci", "environmental_sci"]].sum().to_dict())
print(f"ESIP checklist {ESIP_DOI}: OpenAlex id {esip['id']}, cited_by_count {esip.get('cited_by_count')},"
      f" citing works listed {n_esip}")
for w in esip_cites["results"]:
    print("   cites ESIP:", w.get("publication_year"), w.get("display_name"))


# ---------------- Panel B: dataset hosts ----------------
def hf_count(term):
    path = CACHE / "ai_ready" / TODAY / f"hf_{term}.json"
    if path.exists():
        return json.loads(path.read_text())["n"]
    url, params, n = "https://huggingface.co/api/datasets", dict(search=term, limit=1000), 0
    while url:
        r = requests.get(url, params=params, headers=HDR, timeout=120)
        r.raise_for_status()
        n += len(r.json())
        url, params = r.links.get("next", {}).get("url"), None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"n": n}))
    return n


def zenodo_count(term):
    d = cached_json(f"zenodo_{term}", "https://zenodo.org/api/records",
                    dict(q=term, type="dataset", size=1))
    return d["hits"]["total"]


def github_count(topic):
    path = CACHE / "ai_ready" / TODAY / f"gh_{topic}.json"
    if not path.exists():
        time.sleep(7)  # unauthenticated search allows 10 requests per minute
    d = cached_json(f"gh_{topic}", "https://api.github.com/search/repositories",
                    dict(q=f"topic:{topic} dataset", per_page=1))
    return d["total_count"]


rows = []
for t in TERMS:
    rows.append(dict(host="Hugging Face", term=t, count=hf_count(t)))
    rows.append(dict(host="Zenodo", term=t, count=zenodo_count(t)))
for t in GH_TOPICS:
    rows.append(dict(host="GitHub", term=f"topic:{t} + dataset", count=github_count(t)))
hosts = pd.DataFrame(rows)
print(f"\nDataset hosts, keyword search, retrieved {TODAY}:")
print(hosts.to_string(index=False))

# CSV with both panels
csv = pd.concat([
    lit.melt(id_vars="year", var_name="term", value_name="count").assign(
        source="OpenAlex works, phrase AI-ready in title/abstract"),
    pd.DataFrame([dict(source="OpenAlex works citing ESIP checklist", term=ESIP_DOI, count=n_esip)]),
    hosts.rename(columns={"host": "source"}),
], ignore_index=True)
csv["retrieved"] = TODAY
csv[["source", "term", "year", "count", "retrieved"]].to_csv(OUT / "bibliometrics_counts.csv", index=False)

# ---------------- Figure ----------------
fig = plt.figure(figsize=(27.0, 9.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.22)
ax0, ax1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

x = np.array(YEARS)
ax0.bar(x, lit["all_fields"], color=MUTED, alpha=0.55, label="all fields")
ax0.plot(x, lit["environmental_sci"], "s-", color=GOOD, lw=3, ms=9, label="Environmental Science")
ax0.plot(x, lit["earth_planetary_sci"], "o-", color=ACCENT, lw=3, ms=9, label="Earth & Planetary Sciences")
ax0.set_yscale("symlog", linthresh=10)
ax0.set_yticks([0, 1, 10, 100, 1000])
ax0.set_yticklabels(["0", "1", "10", "100", "1000"])
for xi, v in zip(x, lit["all_fields"]):
    if v:
        ax0.text(xi, v * 1.12, f"{v}", ha="center", va="bottom", fontsize=14, color=INK)
ax0.set_xticks(x[::1])
ax0.set_xticklabels([str(y)[2:] if y != 2026 else "26*" for y in YEARS])
ax0.set_xlabel("publication year (20xx)")
ax0.set_ylabel("works per year")
ax0.set_title('A. Papers with "AI-ready" in title or abstract')
ax0.legend(loc="upper left", fontsize=16, frameon=False)
ax0.text(0.02, 0.72, f"OpenAlex, retrieved {TODAY}; *2026 is a partial year\n"
         f"works citing the ESIP AI-readiness checklist (2022): {n_esip} listed\n"
         f"(cited_by_count {esip.get('cited_by_count')})",
         transform=ax0.transAxes, ha="left", va="top", fontsize=14, color=MUTED)
ax0.set_ylim(0, lit["all_fields"].max() * 3)
for s in ("top", "right"):
    ax0.spines[s].set_visible(False)

hz = hosts[hosts.host != "GitHub"].pivot(index="term", columns="host", values="count").loc[TERMS]
gh = hosts[hosts.host == "GitHub"]
labels = list(hz.index) + [t.replace(" + dataset", "").replace("topic:", "GH topic: ") for t in gh.term]
yy = np.arange(len(labels))[::-1]
h = 0.38
ax1.barh(yy[:len(hz)] + h / 2, hz["Zenodo"], h, color=GOOD, label="Zenodo records, type dataset")
ax1.barh(yy[:len(hz)] - h / 2, hz["Hugging Face"], h, color=THIRD, label="Hugging Face datasets")
ax1.barh(yy[len(hz):], gh["count"], 0.6, color=INK, label="GitHub repos, topic + 'dataset'")
vals = list(zip(hz["Zenodo"], hz["Hugging Face"]))
for i, (z, f) in enumerate(vals):
    ax1.text(z * 1.08, yy[i] + h / 2, f"{z:,}", va="center", fontsize=13, color=GOOD)
    ax1.text(max(f, 1) * 1.08, yy[i] - h / 2, f"{f:,}", va="center", fontsize=13, color=THIRD)
for j, c in enumerate(gh["count"]):
    ax1.text(max(c, 1) * 1.08, yy[len(hz) + j], f"{c:,}", va="center", fontsize=13, color=INK)
ax1.set_xscale("log")
ax1.set_yticks(yy)
ax1.set_yticklabels(labels, fontsize=16)
ax1.set_xlim(1, hosts["count"].max() * 6)
ax1.set_xlabel("number of results for the search term (log scale)")
ax1.set_title("B. Datasets returned by a keyword search")
ax1.legend(loc="lower right", fontsize=15, frameon=False)
fig.text(0.99, -0.04, f"Keyword counts on {TODAY}, not a census. Hugging Face matches the term in the "
         "dataset name; Zenodo searches the full record; GitHub matches topic plus the word dataset.",
         ha="right", fontsize=15, color=ACCENT, style="italic")
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

fig.savefig(OUT / "bibliometrics.png", dpi=110, bbox_inches="tight", facecolor="white")
print("wrote", OUT / "bibliometrics.png", "and", OUT / "bibliometrics_counts.csv")
