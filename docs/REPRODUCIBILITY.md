# Reproduction guide

## Two different workflows

1. **Restore and verify the frozen run**: use the committed snapshot plus the
   exact saved model bundle. This is the recommended route to existing reports
   and predictions. It does not train or repeat final-holdout inference.
2. **Replay the historical training workflow**: use an isolated checkout of the
   pre-freeze revision. This produces a separate experiment, not a replacement
   for the published freeze. It was not retrained during Phase 11.

The model JSON files are intentionally untracked. A code-only clone is therefore
insufficient for frozen verification; obtain the bundle from the project owner.
The repository does not currently host a public artifact release. If the bundle
is unavailable, reports can still be read, but exact frozen inference is blocked.
Do not clear the frozen configuration or retrain in place to conceal that gap.

## 1. Get the complete Git history

The configured upstream is:

```bash
git clone https://github.com/akulusprimerift/epl-match-outcome-predictor.git
cd epl-match-outcome-predictor
```

No publication/push is performed by Phase 11. An upstream checkout may lag this
local work; obtain the completed Phase 11 revision from the owner if the new
documentation/helpers are absent. Do not use a shallow clone or download only
a source ZIP: the verifier reads earlier committed freeze/protocol records.

Git attributes preserve the exact raw and report bytes and force LF for processed
tables. Do not re-save these files in Excel or an editor that changes encoding,
numeric formatting or line endings. Frozen verification detects such changes.

## 2. Install the locked environment

Use Git on PATH and Python 3.12 (the tested interpreter). The specification's
Python 3.11 minimum does not guarantee that this newer dependency lock installs
on 3.11. No dependency versions need to be changed.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv/runtime
.\.venv\runtime\Scripts\Activate.ps1
python -m pip install -r requirements.lock.txt
python -m pip check
```

macOS/Linux equivalent, not locally tested:

```bash
python3.12 -m venv .venv/runtime
source .venv/runtime/bin/activate
python -m pip install -r requirements.lock.txt
python -m pip check
```

If `py`/`python3.12` is not available, use the executable for your installed
Python 3.12. Do not reuse a moved virtual environment: its interpreter paths can
point to an installation that no longer exists. `.venv/runtime` is a replacement
location that preserves any older environment and cached files. On Windows,
activation can be avoided by using `.\.venv\runtime\Scripts\python.exe` wherever
the following commands say `python`; no system-wide policy change is needed.

## 3. Restore the model bundle

On the owner's already-working checkout, with all nine saved artifacts present:

```bash
python scripts/frozen_artifacts.py export --archive .venv/frozen-models.zip
```

This exports only the nine model/preprocessing/metadata JSON files listed below,
after validating their SHA-256 values against `config/model_config.json`.
It does not include credentials, raw data, virtual environments, or arbitrary
files. Use a new filename if an archive already exists; export never overwrites.
The archive stays local/ignored; the owner must separately supply it to the
recipient. No external upload is performed by the helper.

Place the supplied archive at `.venv/frozen-models.zip` in the fresh clone:

```bash
python scripts/frozen_artifacts.py restore --archive .venv/frozen-models.zip
python -m src.freeze_model --verify
```

Restore checks the entire archive before publishing files. Extra, missing,
duplicate, oversized, or checksum-mismatched entries fail. Identical existing
files are skipped; different existing files are never overwritten. The archive
is not blindly extracted. After an interruption, repeat restore: completed
identical files are retained. A different existing model needs investigation,
not forced replacement.

Required local files (all nine are needed by whole-freeze verification):

```text
models/model_a_xgb.json
models/model_metadata.json
models/preprocessing.json
models/model_a_matched_xgb.json
models/model_a_matched_metadata.json
models/model_a_matched_preprocessing.json
models/model_b_xgb.json
models/model_b_metadata.json
models/model_b_preprocessing.json
```

Model files and archives must remain ignored. An archive matching a different
training run will not match this freeze even if its filenames look correct.

## 4. Reproduce metrics and predictions without retraining

From the project root, with the environment active:

```bash
python -m src.evaluate --model-name model_a --split test
python -m src.evaluate --model-name model_a_matched --split test
python -m src.evaluate --model-name model_b --split test
python -m src.evaluate --model-name selected --split holdout --frozen
python -m src.predict --home "Arsenal" --away "Chelsea" --date 2026-09-12
python -m unittest discover -s tests -v
python -m unittest discover -s scripts/tests -v
python scripts/validate_docs.py
```

The first three commands recalculate test metrics using the existing models,
not new fits. Compare their JSON to `reports/model_results.csv`. Final holdout
returns the already-saved row from `reports/final_holdout_results.csv`; it is
not another evaluation. Its receipt verifies the saved confusion matrix and
probability rows. Tests must not call real holdout prediction again.

The prediction example is hypothetical. Compare its complete JSON with
`docs/sample_prediction.json`; repeated calls on the same snapshot are
deterministic. It warns about a 111-day gap since the last stored team match.

The committed reports are available immediately in a clone. The exact historical
pipeline below describes how reports were originally generated. Do not rerun
report-writing/training commands in the current frozen checkout: even changing
a report's timestamp or formatting can invalidate its original checksum.

## 5. Historical data-to-reports recipe (isolated, pre-freeze only)

This is a separate reconstruction, **not** the frozen-restoration quick start.
From an existing full checkout, create a separate local clone:

```bash
git clone --no-hardlinks . .venv/historical-replay
cd .venv/historical-replay
git switch --detach 0980258f828c63744ab7863eb3a032fdf67b3808
```

That Phase 7 revision precedes model selection and has an unfrozen configuration.
Use the already-active Python 3.12 environment or create one here using step 2.
It contains the tracked historical source CSVs and manifested possession export.

```bash
python -m src.download_data --all
python -m src.clean_data
python -m src.build_history
python -m src.build_features --feature-set baseline
python -m src.train_baselines --feature-set baseline
python -m src.train_xgboost --model-name model_a --feature-set baseline
python -m src.collect_possession --all --max-requests 250
python -m src.train_xgboost --model-name model_a_matched --feature-set baseline_matched
python -m src.train_xgboost --model-name model_b --feature-set possession
python -m src.compare_models --models model_a_matched model_b
```

Expected stage outputs: 6,080 canonical fixtures, 12,160 team-history rows,
6,080 baseline rows, 180 source team-season possession rows, and 3,040 matched
rows. Training creates the model JSONs, validation/test metrics, tuning records,
class-distribution and confusion/importance charts. Compare reports with the
current recorded results, allowing for numerical/platform differences rather
than claiming new artifacts have the old timestamps/checksums.

With valid tracked caches, the downloader/collector should need no network
requests. Without them, upstream availability and schema changes are external
dependencies; the URLs are not guarantees of permanent downloadable snapshots.
The collection commands are not required for the restored frozen run.

Do not copy this reconstruction's models or reports over the current freeze.
Do not select/tune using the already-opened 2025/26 season. Do not run a new final
holdout on that season. A future modeling change needs a genuinely future holdout.

## 6. Local clean-clone rehearsal

After the Phase 11 files have been committed, the following uses only local Git
history and the owner's bundle. Start in the main project root with its locked
environment active and the archive from step 3 already present:

```bash
git clone --no-hardlinks . .venv/repro-checkout
cd .venv/repro-checkout
python scripts/frozen_artifacts.py restore --archive ../frozen-models.zip
python -m src.freeze_model --verify
python scripts/validate_docs.py
python -m unittest discover -s tests -v
python -m unittest discover -s scripts/tests -v
python -m src.predict --home "Arsenal" --away "Chelsea" --date 2026-09-12
git status --short
```

Use an unused clone directory. The clone has independent working files and an
independent Git object store; it shares only the activated dependency environment
for this rehearsal. It does not copy ignored source-tree models automatically.

## Validation scope and troubleshooting

Phase 11 checks the 147 modeling tests, four artifact-transfer tests, exact
sample output, documentation links/metric values, CLI help, and freeze integrity.
The replacement Python environment is installed from the unchanged lock and
passes `pip check`. A local clean clone is checked separately after bundle
restoration. No model is retrained and no holdout inference is repeated.

The full historical training recipe and macOS/Linux setup are documented from
the existing code but are not re-executed as part of Phase 11. Bit-for-bit
reproduction across arbitrary machines is not promised. Phase 12 is a separate
final quality gate, not implicitly completed by these checks.

| Symptom | Safe next step |
|---|---|
| Python executable missing | Create `.venv/runtime` using an installed Python 3.12; preserve old caches/models |
| Missing model JSON | Obtain the matching owner-supplied bundle and restore it; do not clear the freeze |
| Artifact checksum mismatch | Compare with the original checkout/bundle; never edit raw inputs to make hashes pass |
| Git revision unavailable | Use full history including the freeze and Phase 9/10 records |
| Training refuses frozen config | Expected; use the isolated historical recipe only for a separate reconstruction |
| Stale prediction warning | Expected with the May 24 snapshot; live collection is outside this run |
| Unknown team | Use its exact canonical name from the mapping table |
| Future season unsupported | The required preceding complete EPL season is absent; no whole-season imputation fallback |
| Interrupted holdout marker | Preserve all files and investigate; never delete it to obtain another evaluation |
| HTTP 403/429 during optional collection | Preserve caches/export, respect request budgets, and retry later if appropriate |
