# autoware-ml-reports

Perception-metrics **proposal** and **model release reports** for autoware-ml
(3D detection & segmentation), published with **GitHub Pages**.

The landing page (`index.html`) links to the proposal first, then lists every
report under `reports/`.

## Layout

```
index.html                                   landing page — links to the proposal + all reports
proposal/perception_metrics_proposal.html    the metrics proposal — standalone, open in a browser
reports/                                      published model reports (one self-contained HTML each)
  ptv3_v0.0.1.html                            PTv3 multi-head PoC release v0.0.1 (real evaluation data)
schema/report_data.schema.json               release-data JSON contract (schema 0.1)
templates/report_layout.html                 report template (data injected into #report-data)
scripts/build_report.py                       merge release JSON(s) → one self-contained report
scripts/build_index.py                        regenerate index.html from the files on disk
scripts/mlflow_to_json.py                     ad-hoc: MLflow run → release JSON (demo-data helper)
data/ptv3_v0.0.1.json                         release v0.0.1 input (extracted from MLflow)
.nojekyll                                     disables Jekyll processing on Pages
.github/workflows/deploy-pages.yml           deploys the site to GitHub Pages on push to main
```

## Add or update a report

1. Build (or drop) a self-contained HTML into `reports/`, e.g.:

   ```
   python scripts/build_report.py --layout templates/report_layout.html \
       --out reports/ptv3_v0.0.1.html data/ptv3_v0.0.1.json
   ```

   `build_report.py` reads release JSONs **oldest → newest** (the last is the current
   release) and concatenates their models per test set so the report draws them as
   comparison bars.

2. Regenerate the landing page so it links the new file:

   ```
   python scripts/build_index.py
   ```

3. Commit and push to `main` — the workflow redeploys Pages automatically.

## GitHub Pages

- The workflow publishes only the rendered pages (`index.html`, `proposal/`, `reports/`)
  and verifies `index.html` is regenerated; every page is self-contained and `.nojekyll`
  disables Jekyll processing.
- Enable once, in **Settings → Pages → Build and deployment → Source: GitHub Actions**.
  On a private repo this requires an org plan with private Pages enabled.

## Notes

- The HTML deliverables (proposal, reports) are **fully self-contained**: all data is embedded
  (each report inlines its JSON into a `<script id="report-data">` block). They reference no
  external files or local paths.
- Only the generation **scripts** reference the schema / data paths — never the HTML.
