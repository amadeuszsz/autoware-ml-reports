# autoware-ml-reports

Perception-metrics **proposal** and **model release reports** for autoware-ml
(3D detection & segmentation), published with **GitHub Pages**.

The landing page (`index.html`) links to the proposal first, then lists every
report under `reports/`.

## Layout

```
index.html                                   landing page: links to the proposal + all reports
proposal/perception_metrics_proposal.html    the metrics proposal, standalone, open in a browser
reports/                                      published model reports (one self-contained HTML each)
  ptv3_v0.0.1.html                            PTv3 multi-head PoC release v0.0.1
  ptv3_v0.1.0.html                            PTv3 multi-head release v0.1.0
  ptv3_v0.1.1.html                            PTv3 multi-head release v0.1.1
schema/report_data.schema.json               release-data JSON contract (schema 0.2)
templates/report_layout.html                 report template (data injected into #report-data)
scripts/build_report.py                       merge release JSON(s) into one self-contained report
scripts/build_index.py                        regenerate index.html from the files on disk
scripts/mlflow_to_json.py                     MLflow test runs of one release into a release JSON
data/ptv3_v0.0.1.json                         release v0.0.1 input (extracted from MLflow)
data/ptv3_v0.1.0.json                         release v0.1.0 input (extracted from MLflow)
data/ptv3_v0.1.1.json                         release v0.1.1 input (extracted from MLflow)
.nojekyll                                    disables Jekyll processing on Pages
.github/workflows/deploy-pages.yml           deploys the site to GitHub Pages on push to main
```

## Add or update a report

1. Extract the release JSON from the MLflow database of the test runs. One `--run`
   per test set the release was evaluated on; the run itself supplies the label
   vocabulary, the scene lists, the frame count and the metric parameters, so the
   script needs the run's database record table and scenario lists on disk (run it
   where the evaluation ran, e.g. inside the autoware-ml container):

   ```
   python scripts/mlflow_to_json.py --db /path/to/mlflow.db \
       --release v0.1.0 --model "PTv3 multi-head" \
       --run det3d-seg3d-j6gen2=<test run id> \
       --run det3d-j6gen2=<test run id> \
       --run seg3d-j6gen2=<test run id> \
       --out data/ptv3_v0.1.0.json
   ```

2. Build a self-contained HTML into `reports/`, oldest release first, the current
   release last:

   ```
   python scripts/build_report.py --layout templates/report_layout.html \
       --out reports/ptv3_v0.1.0.html data/ptv3_v0.0.1.json data/ptv3_v0.1.0.json
   ```

   `build_report.py` concatenates the models per test set so the report draws them
   as comparison bars. The **last** file is the current release and defines the test
   set the page describes: its label vocabulary, dataset, metric spec, label and
   version. Older releases show a dash wherever they carry no key for a row. Pass a
   single file for a report without comparison bars.

   The optional top-level `release` block of the last file fills the release notes
   at the top of the page: the release name, one headline sentence, and the major
   changes in the model and in the test as short bullet lists (`changes.model`,
   `changes.test`). Write it by hand, it is the one part of the file MLflow cannot
   supply.

3. Regenerate the landing page so it links the new file:

   ```
   python scripts/build_index.py
   ```

4. Commit and push to `main`; the workflow redeploys Pages automatically.

## GitHub Pages

- The workflow publishes only the rendered pages (`index.html`, `proposal/`, `reports/`)
  and verifies `index.html` is regenerated; every page is self-contained and `.nojekyll`
  disables Jekyll processing.
- Enable once, in **Settings > Pages > Build and deployment > Source: GitHub Actions**.
  On a private repo this requires an org plan with private Pages enabled.

## Notes

- The HTML deliverables (proposal, reports) are **fully self-contained**: all data is embedded
  (each report inlines its JSON into a `<script id="report-data">` block). They reference no
  external files or local paths.
- The label vocabulary is data, not layout: every test set in a release JSON carries the
  classes and behaviour groups its release was scored on, and the charts draw their rows from
  it. A release scored on a different vocabulary therefore renders its own class axis, and
  the previous release shows a dash where a class no longer exists.
- Only the generation **scripts** reference the schema / data paths, never the HTML.
