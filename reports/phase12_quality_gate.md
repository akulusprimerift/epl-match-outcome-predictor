# Phase 12 — final quality gate

Validation date: 2026-09-05. Result: **PASS within the documented Windows,
Python 3.12, locked-dependency, owner-supplied frozen-bundle environment.**

The user explicitly extended Phase 12 to include a testable match-selection
interface with explanations and actual gathered statistics, selected local
operation, and authorized direct browser testing. There is no hosted release,
live collection, new model-selection decision, or changed statistical contract.

## Acceptance evidence

| Criterion | Evidence |
|---|---|
| Automated tests | 174 pass: 147 frozen modeling tests, 4 artifact-transfer tests, 23 interface/evidence/HTTP tests |
| Raw reproducibility | All 16 Football-Data seasons were checksum-validated from immutable caches in an independent pre-freeze clone; all 180 SofaScore rows were reused with zero requests |
| Leakage and EPL-only scope | Existing temporal, split, possession-lag, source-integrity, and prediction tests pass; no other-league inputs added |
| Processed-data reproduction | Canonical fixtures, team history, baseline and matched datasets, possession table, and both split manifests reproduce identically as parsed tables |
| Models and baselines | Full declared historical recipe rerun in isolation; all validation/test metric rows match, including both baselines; all three XGBoost model JSON files match original SHA-256 hashes |
| Selection and holdout order | Existing freeze verifier and historical protocols pass; Model B remains selected; no final-holdout inference repeated |
| Normalized prediction | Interface prediction equals the committed Arsenal–Chelsea sample exactly; probabilities sum to one within numerical tolerance |
| CLI and dependencies | All six required Phase 12 module help commands and interface help succeed; unchanged lock passes `pip check` |
| Documentation | README contains launch, restore, test and limitation guidance; documentation validator checks links, exact sample, metrics, feature order, paths and credential patterns |
| Secrets | Tracked filename/content scan and review found no credentials/private keys; the only tracked environment template is comments-only `.env.example`; local environments and model bundle remain ignored |
| Testable user interface | Browser team/date selection, submission, swap, duplicate-team error, result recovery, displayed evidence, and browser-agent prediction tested; screenshot reviewed and browser error log empty |
| Honest limitations | Snapshot date, stale history, missing possession/median replacement, unverified fixture/membership, probabilistic uncertainty, and weak draw recall are disclosed |

## Interface behavior and explanation checks

Launch with `python -m interface.server`, then open the printed loopback address
(default `http://127.0.0.1:8765/`). Stop with Ctrl+C. The existing Python stack
serves only three whitelisted static assets and two local JSON routes; it does
not expose repository files, write predictions, or require credentials.
Host/origin checks reject foreign requests, JSON size is bounded, concurrent
predictions receive a retry response, and security headers restrict page assets.
This is a local demonstration server, not an internet-facing deployment.

The separate `interface/` package calls the unchanged Phase 10 prediction path.
Native exact XGBoost TreeSHAP uses the same saved tree range `[0, 268)`. Its
three class-score sums must reconstruct the predicted probabilities within
absolute tolerance `1e-6`. All 25 frozen features appear exactly once in seven
groups. The grouped score contributions plus learned baseline reconstruct the
log-probability ratio for the displayed outcome comparison. No retraining,
new calibration, generic gain chart, or invented causal story is used.

For Arsenal at home against Chelsea on the illustrative date 2026-09-12:

| Value | Arsenal | Chelsea |
|---|---:|---:|
| Win probability | 71.0719% | 11.8456% |
| Points in last five stored EPL matches | 15 | 4 |
| Goals scored per match in that window | 1.6 | 1.0 |
| Goals conceded per match in that window | 0.2 | 2.0 |
| Shots per match in that window | 14.8 | 10.4 |
| Last-five home/away-role points per match | 2.4 | 0.8 |
| Previous-season possession (2025/26, 38 matches) | 56.1% | 57.7% |

Draw probability is 17.0825%. The page shows the actual recent and venue
matches and links to their manifested Football-Data source and SofaScore
season summary. The source links are provenance, not a guarantee that an
upstream endpoint will remain reachable. Recorded matches end on 2026-05-24.

Additional real-data cases cover a sub-50% leader with opposing influences
(Everton–Manchester United) and missing prior-season possession
(Luton Town–Arsenal). Luton possession is shown as unavailable, separately
from the frozen model's 51% training-median input. Synthetic tests cover a
leading draw, invalid attribution, and empty history. Invalid teams, identical
teams, malformed/historical dates and unsupported seasons fail deliberately.

The optional `predict_epl_match` WebMCP tool was verified in a supported browser
context: expected registration/schema/annotations, valid Luton–Arsenal execution
with matching visible result, and invalid duplicate-team error. Ordinary form
controls were tested separately and require no WebMCP support.

The Sites-building workflow informed the local-first form/evidence layout,
early preview and task-completing browser-agent surface. The user's local-only
choice overrides publishing; no hosting configuration or new dependencies
were introduced. The rendered desktop/in-app layout was inspected; a separate
mobile device, every browser engine and every possible fixture were not tested.

## Isolated historical replay

An independent local Git clone at pre-freeze revision
`0980258f828c63744ab7863eb3a032fdf67b3808` executed every command in the
[historical recipe](../docs/REPRODUCIBILITY.md), including the bounded training
search for Model A, Model A-Matched and Model B. It reused the already-tested
locked Python environment; this was not a second package installation.

The replay produced 6,080 fixtures, 12,160 history rows, 6,080 baseline rows,
180 source team-season rows and 3,040 matched rows. Selected iterations were
121, 240 and 267, respectively. All validation/test report metrics reproduced
exactly and all three model JSONs were byte-identical to their frozen originals.
Historical training reports explicitly returned `holdout_evaluated=False`.
Nothing from the replay was copied over the current frozen artifacts.

The Phase 11 independent-clone bundle restoration and fresh locked-environment
installation remain applicable because this interface adds no dependencies.
Its [reproduction guide](../docs/REPRODUCIBILITY.md) records that separate
restoration exercise. A code-only clone still requires the owner's nine-file
model bundle to reproduce the existing freeze without retraining.

Original freeze record:
`f3b9ac19656dc1a8218176250f182406895b4fa0bcff0cb3a8432a18cf79a530`.
The current source, raw/processed data, configuration, model files, Phase 9/10
protocols, and original evaluation reports were not modified by Phase 12.

## Boundaries

This validates reproducibility on the tested platform, not permanent provider
availability, universal bit-for-bit portability, present-day team form,
calibrated probabilities, profitability or causal inference. The final holdout
accuracy remains 46.32%, with only 2 of 104 draws correctly classified. No
modeling change was chosen using those holdout results. Any future model change
needs separate approval and a genuinely future evaluation season.
