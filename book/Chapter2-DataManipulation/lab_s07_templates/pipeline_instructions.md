# Instructions for an AI tool: <track>, from the raw file to <derived quantity>

<!--
Session 7, submission item (c). Copy this file to s07_my_files/ before you edit it.

What this file is: instructions an AI tool follows to run your track's data chain,
from the provider's raw file to the derived quantity of Part 5, on the real data, not the twin.
Write them so that someone who never saw your notebook (a classmate, or an AI tool)
gets your number. The tool also follows your reproducibility_rules.md; refer to a
rule by its heading instead of repeating it.

Fill in every <...>. Keep each step to one or two lines. The track hints at the end
give the provider, the files and the derived quantity; delete them before you submit.
-->

## 1. Goal

Compute <derived quantity, its unit, the place and the time window> from <provider and dataset>. Follow every rule in `reproducibility_rules.md`. Show all code you write.

## 2. Get the raw file

- Download <file or query> from <URL>. Save it unchanged under the name <file name>.
- Record the retrieval date and time (UTC), the size in bytes and the SHA-256 fingerprint.
- If you cannot download (no network, an error from the server, a tool that cannot fetch files), say so, use the attached copy instead, and say that you did.

## 3. Confirm the file was not modified or corrupted

Compare your file with the course copy, whose values are recorded in `snapshots/<name>.provenance.json`:

| Check | Course copy | Your file | Same? |
|---|---|---|---|
| Size (bytes) | <from the provenance file> | | |
| Number of rows or samples | <from the notebook, Part 1> | | |
| First record | <...> | | |
| Last record | <...> | | |
| SHA-256 fingerprint | <first 12 characters> | | |

If the fingerprints differ, use the other four rows to say which it is: a revision by the provider (rows added, values updated, the file otherwise intact) or a damaged file (cut short, unreadable records). Do not continue with a damaged file.

## 4. Clean and select

Do these steps in this order. For each, report the number of values it touched.

| Step | What to do | Flag, remove or keep | Physical reason |
|---|---|---|---|
| 1 | <convert missing-value codes to missing> | remove | <the provider's code for "no measurement", not a value> |
| 2 | <...> | flag | <...> |
| 3 | <a real extreme you saw in the notebook> | keep | <why it is physically possible, with the date> |
| ... | | | |

Flag when the value may be right and a reader should know (a gap filled by the provider, a sensor change). Remove when the value cannot be a measurement of the quantity (a missing-value code, a duplicated record, a sample at the clipping level). Keep real extremes, and say that you kept them.

## 5. Compute the derived quantity

<The definition in one or two sentences, with the formula, the selection of days, events or cadences that enter it, and the unit. Copy it from Part 5 of your notebook.>

Report it twice: on the file as read, with no step applied, and after all of step 4. Part 8 of your notebook prints reference values on the real data, for comparison.

## 6. Report

Return, in this order:
1. Every command or block of code you ran, with the last line of its output.
2. The table of section 3, filled in.
3. The table of section 4, with the number of values each step touched.
4. The derived quantity on the file as read and after cleaning, with its unit.
5. What you could not do, and why (a download refused, a format you cannot read, a step you skipped). "Nothing" is an answer only if it is true.

---

## Track hints (delete before you submit)

**Volcano**
- Raw files: the catalog from ComCat, `https://earthquake.usgs.gov/fdsnws/event/1/query?minlatitude=46.1&maxlatitude=46.3&minlongitude=-122.32&maxlongitude=-122.08&starttime=2004-01-01&endtime=2009-01-01&format=csv&orderby=time-asc`, and six hours of HSR from EarthScope, `https://service.earthscope.org/fdsnws/dataselect/1/query?net=UW&sta=HSR&loc=--&cha=EHZ&start=2004-11-29T15:00:00&end=2004-11-29T21:00:00&nodata=404`, with its response from the station service (the URL is in `snapshots/msh_2004-2008.provenance.json`).
- Course copies: `comcat_msh_2004-2008.csv` (11,305 events) and `msh_UW.HSR.EHZ_2004-11-29T15-21.mseed`.
- Derived quantity: the drumbeat clock, the coefficient of variation (standard deviation divided by mean) of the intervals between successive drumbeats, dimensionless. Say whether the drumbeats come from detections in the HSR record (Part 3) or from catalog events below magnitude 2.0, and skip intervals that span a data gap.
- Steps to think about: the 20.3-minute gap in the HSR record, clipped samples, the telemetry burst seen at three stations at once, duplicated catalog events, mixed magnitude types.

**Soil moisture**
- Raw files: one file per year from NOAA NCEI, `https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/<YEAR>/CRND0103-<YEAR>-WA_Spokane_17_SSW.txt` for 2015 to 2024, with the column list in `https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/HEADERS.txt` and the codes in `https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/README.txt`, one folder up from the yearly folders.
- Course copy: `uscrn_WA_Spokane_17_SSW_2015-2024.txt` (3,653 days).
- Derived quantity: the soil-moisture response to rain, the least-squares slope of the day-to-day change in soil moisture at 5 cm against daily precipitation, on days with at least 2 mm and soil temperature at 5 cm above 0 °C that day and the day before, in m³/m³ per 10 mm.
- Steps to think about: the −99.000 and −9999.0 codes, frozen-soil gaps, the November to December 2023 outage, snow in the precipitation gauge, changes of the datalogger program version (`CRX_VN`). On the real file, the −99.000 codes left in as numbers inflate the slope about 100 times (Part 8 prints both values); frozen days and the 2023 outage are already blank.

**Land-surface model**
- Raw files: from NOAA PSL, `https://downloads.psl.noaa.gov/Datasets/cpcsoil/soilw.mon.mean.v2.nc` (soil water) and `https://downloads.psl.noaa.gov/Datasets/precl/0.5deg/precip.mon.mean.0.5x0.5.nc` (precipitation), plus `landperc.nc` from the cpcsoil folder. These are whole global files (about 545 MB); the subset is 31 to 49 N, 235 to 251 E (125 to 109 W), 1991 to 2020. PSL's subset service refused every request while the lab was built (HTTP error 429, too many requests), so the tool must download the whole files and cut them, or say that it could not.
- Course copy: `cpc_soilw_precl_westUS_1991-2020.nc` (36 latitudes by 32 longitudes by 360 months).
- Derived quantity: drought frequency, the percent of land cell-months in 2011 to 2020 with soil water below the 20th percentile of the same cell and calendar month over 1991 to 2010. Part 7 of the notebook prints it on the course copy.
- Steps to think about: the longitude convention, latitudes stored north to south, coastal cells with almost no land, the provider recomputing the most recent month, the drop in the number of gauges after 1996.

**Planetary**
- Raw files: the light curve from MAST, `https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:TESS/product/tess2022244194134-s0056-0000000420814525-0243-s_lc.fits`, and the planet table from the NASA Exoplanet Archive through its Table Access Protocol (TAP) service (the full query is in `snapshots/exoplanets_hd209458_tess_s56.provenance.json`).
- Course copies: the `.fits` file above and `nea_pscomppars_2026-10-14.csv` (6,445 planets).
- Derived quantity: the transit depth of HD 209458 b in ppm (flux divided by its median; mean flux in the central half of each transit subtracted from the mean flux 0.75 to 1.5 transit durations on either side), and the planet radius in Earth radii, √depth × stellar radius (solar radii) × 109.08. Name the source of the period, the mid-transit time, the duration and the stellar radius.
- Steps to think about: which `QUALITY` bits to remove, `PDCSAP_FLUX` against `SAP_FLUX`, the scattered-light stretch, the two archive downloads that had the same size and different fingerprints (a revision, not damage), duplicated planet rows with different stellar radii.
