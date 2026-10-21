"""Slide figures: how well GPT-4 recovers IPCC expert confidence (Lacombe, Wu & Dilworth, 2023).

Rebuilds the literature example for lecture 3 from open sources instead of
copying the paper's figures:
  statements   ClimateX dataset, 8,094 sentences from the IPCC AR6 WGI, WGII and
               WGIII reports, each with its IPCC confidence label (CC BY 4.0)
               https://huggingface.co/datasets/rlacombe/ClimateX
  model scores Table 5 (average predicted confidence per IPCC class) and Table 6
               (accuracy) of Lacombe et al. (2023), arXiv:2311.17107, CC BY 4.0,
               parsed from the arXiv HTML at run time. The per-statement model
               answers were not released, so the result panel re-plots the
               published class means; it cannot be recomputed from the dataset.

Writes one two-panel PNG to figs/lit_climatex/ and prints the sample records and every
number quoted on the slide.
Regenerate: pixi run python book/slides/2026/figscripts/climatex_ipcc_confidence.py
"""
import html
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pooch  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 18, "ytick.labelsize": 18,
})
CACHE = Path(pooch.os_cache("mlgeo-figscripts"))
INK, MUTED, GRID = "#26241f", "#6e675c", "#d9d4ca"
WG1_C, WG23_C = "#2a78d6", "#eb6834"                       # dataviz categorical slots 1 and 2
LEVELS = ["low", "medium", "high", "very high"]
LEVEL_C = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]     # ordinal blue ramp, steps 250-700
REPORTS = {"AR6_WGI": "WGI physical science", "AR6_WGII": "WGII impacts", "AR6_WGIII": "WGIII mitigation"}
RECORD_IDX = [497, 415, 237, 322]                          # four WGI test statements, one per label
OUT = Path(__file__).resolve().parent.parent / "figs" / "lit_climatex"
OUT.mkdir(parents=True, exist_ok=True)


def fetch(url, name):
    return pooch.retrieve(url, known_hash=None, fname=name, path=CACHE, progressbar=False)


df = pd.read_csv(fetch("https://huggingface.co/datasets/rlacombe/ClimateX/resolve/main/ipcc_statements_dataset.tsv",
                       "climatex/ipcc_statements_dataset.tsv"), sep="\t")

# Published model results, parsed from the paper's HTML tables
raw = open(fetch("https://arxiv.org/html/2311.17107v1", "climatex/arxiv_2311.17107v1.html"), encoding="utf-8").read()
text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", re.sub(r"<(script|style).*?</\1>", "", raw, flags=re.S))))
COLS = ["gpt35_zero", "gpt35_few", "gpt4_zero", "gpt4_few", "cmd_zero", "cmd_few"]


def nums(s):
    return [float(x) for x in s.split()]


table5 = {}
for lvl, score in [("very high", 3), ("high", 2), ("medium", 1), ("low", 0)]:
    m = re.search(rf"‘{lvl}’ \({score}\) WG1 ([\d. ]+?) WG2&3 ([\d. ]+?) All reports ([\d. ]+?) Support", text)
    table5[lvl] = {k: dict(zip(COLS, nums(v))) for k, v in zip(["WGI", "WGII+III", "all"], m.groups())}
acc = dict(zip(COLS, nums(re.search(r"Accuracy ([\d. ]+?) Macro F1", text).group(1))))
low_recall = dict(zip(COLS, nums(re.findall(r"Recall ([\d. ]+?) F1", text)[3])))

# ---- One figure, two panels: the evaluation set (left) and GPT-4's labels against the IPCC labels (right)
counts = pd.crosstab(df["report"], df["confidence"]).reindex(index=list(REPORTS), columns=LEVELS)
fig = plt.figure(figsize=(20, 4.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.42)
ax = fig.add_subplot(gs[0])
left = pd.Series(0, index=counts.index)
for lvl, col in zip(LEVELS, LEVEL_C):
    ax.barh([REPORTS[r] for r in counts.index], counts[lvl], left=left, color=col,
            edgecolor="white", linewidth=2, height=0.68, label=lvl)
    left += counts[lvl]
ax.invert_yaxis()
for i, r in enumerate(counts.index):
    ax.text(left[r] + 60, i, f"{left[r]:,}", va="center", fontsize=17, color=INK)
ax.set_xlim(0, left.max() * 1.16)
ax.set_xlabel("IPCC AR6 statements")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(-0.02, 0.98), frameon=False, fontsize=17,
          handlelength=1.0, columnspacing=1.0)
ax.set_title("8,094 statements in ClimateX, by IPCC label", loc="left", pad=44)

# Average label GPT-4 assigned, against the IPCC label (few-shot setting)
ax = fig.add_subplot(gs[1])
x = range(4)
ax.plot([-0.15, 3.15], [-0.15, 3.15], color=MUTED, lw=2, ls="--", zorder=1, label="perfect agreement")
for key, col, lab in [("WGI", WG1_C, "WGI (before model cutoff)"),
                      ("WGII+III", WG23_C, "WGII + WGIII (after)")]:
    y = [table5[lvl][key]["gpt4_few"] for lvl in LEVELS]
    ax.plot(x, y, color=col, lw=2.5, marker="o", ms=11, mec="white", mew=2, label=lab, zorder=3)
y_all_low = table5["low"]["all"]["gpt4_few"]
ax.annotate(f"low-confidence statements:\nmean label {y_all_low:.2f}, IPCC 0",
            xy=(0.04, y_all_low + 0.06), xytext=(0.2, 2.15), fontsize=16, color=INK,
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))
ax.set_xticks(list(x), LEVELS)
ax.set_yticks([0, 1, 2, 3], LEVELS)
ax.set_xlim(-0.3, 3.3); ax.set_ylim(-0.3, 3.3)
ax.set_xlabel("IPCC confidence label")
ax.set_ylabel("mean GPT-4 label")
ax.grid(color=GRID, lw=0.8); ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower right", frameon=False, fontsize=15, borderaxespad=0.1)
ax.set_title("GPT-4 labels, 300 test statements", loc="left")
fig.savefig(OUT / "climatex_panels.png", dpi=130, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

# ---- Numbers quoted on the slide and in the notes
pd.set_option("display.max_colwidth", 140); pd.set_option("display.width", 220)
print(f"statements: {len(df):,}; by label:", df["confidence"].value_counts().to_dict())
print(f"low share of all statements: {(df['confidence'] == 'low').mean():.1%}")
print(counts)
test = df[df["split"] == "test"]
print(f"test set: {len(test)}", pd.crosstab(test["report"], test["confidence"]).to_dict())
print("\nsample records (WGI test statements):")
print(df.loc[RECORD_IDX, ["report", "page_num", "statement", "confidence", "score", "split"]].to_string())
print("\nGPT-4 few-shot mean label per IPCC class (Table 5):",
      {lvl: {k: v["gpt4_few"] for k, v in table5[lvl].items()} for lvl in LEVELS})
print("accuracy (Table 6):", acc)
print("recall on low-confidence statements (Table 6):", low_recall)
print(f"wrote {OUT}")
