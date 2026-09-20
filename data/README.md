# Input data

This folder is empty in the repository. The pipeline expects four files here:

| File | What it is | Source |
|---|---|---|
| `PDS_roads_daily_affected.csv` | Road features with a daily fire-perimeter intersection flag, days 7 to 12 | Prepared for this study; available from the authors on request |
| `daily_affected_population_totals.xlsx` | Daily affected population | Prepared for this study; available from the authors on request |
| `pop_PDS.csv` | Block-level population and age structure | [US Census Bureau](https://data.census.gov) |
| `dasymetric_landcover_weights.xlsx` | Land-cover classes and relative dasymetric weights | [National Land Cover Database](https://www.mrlc.gov/data) |

Road geometry is OpenStreetMap-derived and licensed under
[ODbL](https://opendatacommons.org/licenses/odbl/).

Paths are configured in `code/config/default.yaml` under `paths.raw_data` and `inputs`.
