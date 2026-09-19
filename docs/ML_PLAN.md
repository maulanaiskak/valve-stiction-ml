# Valve Stiction ML Training — Plan

Status: v1.1 (added 6 verified additional SACAC files from the official portal, see §4.1; milestones 1-10 complete — only the optional gradient-boosting comparison and the PRD's future simulator integration remain, see §12) · Scope: the V3 "bounded ML comparison" component of the [Valve Stiction Fault Detection — Distributed IoT Pipeline PRD](../../../Kuliah/Tugas%20Akhir/File%20TA/Github%20Tugas%20Akhir). This repo owns two things now, not one: a reference implementation of the unsupervised classic detector (needed as a label source, and reusable for the PRD's V1), and training/evaluation of the ML classifier that's meant to approximate it cheaply.

**Note on "unsupervised":** we considered making clustering (KMeans/GMM) the primary modeling method instead of a classic-detector-taught RF. Rejected after brainstorming — see rationale in §10.1. Short version: clustering has no way to know your target concept (stiction specifically, vs. whatever axis of variation happens to dominate feature space), and grounding cluster identity credibly runs straight back into the same file-vs-window granularity problem that started this whole revision (§2). The classic detector is *already* unsupervised (zero training data, zero human labels) — it's just not clustering. Clustering is kept, but demoted to a validation/sanity-check role.

## 1. Decisions already made (with rationale)

| Decision | Choice | Why |
|---|---|---|
| Repo | New, separate from the thesis repo | Clean history, own CI, matches the PRD's framing as a standalone portfolio project |
| Generalization | One general model, unit-agnostic features | The ISDB dataset already mixes chemical/paper/power/mineral loops with different physical PV types (flow, pressure, level, temperature). Splitting by sensor type would fragment an already-small dataset for no proven accuracy gain |
| Training data | Thesis data (ISDB + SACAC) now; pipeline designed to accept PRD's synthetic simulator data later (FR-11) | Simulator doesn't exist yet (gated on V1 Rust/MQTT work); no reason to block ML work on it |
| Model family | Classical ML only (Random Forest primary) — no deep learning | Matches PRD FR-12 and the explicit non-goal "not claiming state-of-the-art ML performance" |
| **Label source** | **Derived per-window from an unsupervised classic detector, not the thesis's yes/no folders** | See §2 — the folder labels are file-level and don't hold at window granularity |

## 2. The label problem, and why it changes the plan

The thesis's `yes`/`no` folder labels were assigned **per file**, then every sliding window cut from that file inherited the file's label. That's an assumption, not a measurement: a file labeled "yes" (stiction) doesn't necessarily stick for its entire duration — a 100-sample window from partway through that file may show nothing stiction-like at all, yet gets trained as a positive example. This isn't occasional mislabeling that averages out — it's a systematic mismatch between the label's granularity (file) and the model's training unit (window), and it affects both ISDB and SACAC equally since it comes from how the data was partitioned, not from either dataset specifically.

Training a supervised classifier directly against these labels means the model partly learns to reproduce *your original file-level guess*, not the underlying physical phenomenon. Any accuracy/F1 number computed against them is not trustworthy on its own.

**The fix isn't to go unsupervised in the clustering sense** — clustering features and then eyeballing which cluster "looks like" stiction just moves the same judgment call to after training, with less rigor than before, not more. **The fix is to replace the label source with something that evaluates each window on its own signal shape, using theory that doesn't need training data at all**: the classic, physics-grounded stiction detection methods from the process control literature. These compute a stiction indicator directly from the PV/OP relationship in a given window — no labeled examples required, so there's no file-to-window generalization to get wrong.

## 3. Research summary — classic detection methods

Three well-established, training-free methods, compared for fit with this project:

| Method | How it works | Scope limitation | Verdict |
|---|---|---|---|
| **Horch cross-correlation** | Cross-correlation function (CCF) between OP and PV; a 90° phase lag (odd CCF) indicates stiction, 180° (even CCF) indicates an aggressive controller or external oscillation | **Not applicable to integrating processes (e.g. level control) or compressible-medium loops (steam/air)** — mainly valid for flow control loops | Disqualified as the *primary* / universal label source — conflicts directly with the "one general model across loop types" decision in §1. Worth keeping as an optional secondary signal, flow loops only |
| **Ellipse fitting (He et al.)** | Fits an ellipse to the PV-OP phase plot; the ellipse's width along the OP axis is the "apparent stiction" magnitude. An improved variant handles parallelogram-shaped plots for a better fit | Estimated magnitude is confounded by controller tuning ("apparent" stiction, not absolute) — needs a considered threshold, not a hard physical constant | **Primary candidate.** Operates directly on PV/OP shape with no assumption about process type (flow/level/pressure/temperature) — matches the general-model goal and the exact two columns this data already has |
| **Kano's method** | Qualitative pattern-matching on the OP-vs-PV plot (characteristic stick-then-jump shape); foundational method, later work fixed known defects in the original model | Rule-based pattern matcher — sensitive to noise, and known defects existed in the original formulation before later corrections | Good as an **independent second opinion**, not primary — a different failure mode than ellipse-fitting, useful for cross-checking |

**Honest ceiling to expect**: even recent non-deep-learning signal-processing methods only reach ~85–90% accuracy on the ISDB benchmark (e.g. a 2024 successive-ridge-detection method reports 88.46% on ISDB, 85.00% on an independent industrial set), and the literature explicitly notes traditional data-driven methods "lack robustness... with high noise and abnormal behavior." So the classic detector is not perfect ground truth either — it's a *principled, per-window, training-free* signal, which is a meaningfully higher bar than file-level label propagation, but not infallible. This gets documented honestly rather than treated as a new gold standard.

## 4. Revised pipeline

```
V1-equivalent (this repo's classic.py — reusable by the PRD's real-time V1 service):
  Raw PV/OP window
    -> ellipse-fit stiction index      (primary)
    -> Kano pattern check              (secondary, independent)
    -> label = "confident yes/no" where both agree,
       "uncertain" where they disagree (dropped from training)

V3 (this repo's actual training target):
  Raw PV/OP window
    -> per-window z-score normalize
    -> tsfel feature extraction
    -> RandomForest trained against the classic detector's confident-agreement labels
    -> evaluated on: (a) how well RF approximates the classic detector on held-out
       windows, (b) precision/recall/F1 in the classic sense, (c) a reported —
       not trained-on — comparison against the old thesis folder labels, framed
       explicitly as "does this roughly agree with the original file-level guess"
```

The old yes/no folder labels are demoted to a **sanity cross-check only**: after training, report how often the RF's (or the classic detector's) per-window verdict agrees with the file-level folder label, framed honestly as a rough consistency check, never as an accuracy claim.

This also strengthens the portfolio story: *"I didn't trust my own thesis-era file-level labels, so I implemented an established unsupervised detector as ground truth and used ML only to approximate it cheaply, with metrics reported against both."* That's a stronger, more defensible narrative than presenting old supervised numbers at face value.

## 4.1 Additional data — checked the official sources, imported what was usable

Went looking for more/other public stiction datasets. Checked directly against the official portals rather than trusting search-result summaries at face value (one summary claimed a "sacac.org.za" URL for our SACAC data, which looked at first like a mismatched result — turned out to be correct: SACAC is literally the *South African Council for Automation and Control*, and that's its real official host).

- **ISDB** (Jelali & Huang, University of Alberta): no public download link found — likely distributed via their book or on request. No evidence our 77-file copy is missing anything readily obtainable.
- **SACAC** (sacac.org.za/resources): diffed the portal's file listing against what's in this repo — **10 files were missing**. Downloaded and inspected each before importing anything:
  - 3 `plantwide-*` files: **not usable** — multi-tag, semicolon+comma-decimal format (temperature/level indicators across a whole plant, e.g. `TI1;TI2;TI3;...`), no clear single PV/OP pair to extract without domain documentation we don't have.
  - `quantisation-T-chemicals-Thornhill-2003.csv`: **not usable** — PV only, no OP column.
  - 6 `unknown-*` files: **usable** — real PV/OP columns (semicolon-delimited on the portal, normalized to this repo's comma convention on import), just a different delimiter. `folder_label="unknown"` for these — the original researchers couldn't determine a root cause either, so these are never forced into yes/no, same treatment as everything else (`scripts/download_additional_sacac.py`).
- **DAMADICS** (real actuator-fault benchmark, sugar factory evaporation station) and **GIMSCOP/UFRGS** (Brazilian oil & gas SISO loops + a synthetic oscillation dataset): both real, verified leads, **not integrated** — different domain/format, would need a real adapter, and the marginal value for this project's scope didn't seem to justify it. Documented here in case that changes later.
- **Result**: 6 new files → 153 new SACAC windows (14 confident "yes", 1 confident "no", 138 "uncertain" — a notably higher uncertain rate than SACAC's overall 49%, which lines up with these being exactly the files the original researchers themselves couldn't root-cause; not over-claiming a strong conclusion from 6 files, but a nice bit of unplanned corroboration for the detector's conservatism). Retrained: SACAC ROC-AUC 0.860→0.865, PR-AUC 0.772→0.788, everything else essentially stable — a real but modest improvement, not a step change.
- **Bug caught while wiring this in**: `sanity_check_agreement` (`evaluate.py`) compared predictions against `folder_label` with no exclusion — `"unknown"` rows would never match a "yes"/"no" prediction and were silently counted as disagreements, understating the agreement rate for no real reason. Fixed to exclude `folder_label == "unknown"` from the comparison entirely (both there and in `label_windows.py`'s sanity-check print), with a regression test.

## 5. Data strategy

- **Source**: `Data/Data Latih ISDB/{yes,no}/*.csv` (train) and `Data/Data Validasi SACAC/{yes,no}/*.csv` (held-out test) from the thesis repo, copied in via `scripts/import_thesis_data.py`. Old folder labels are still imported — as the sanity-check signal from §4, not as training targets.
- **Manifest**: every imported CSV gets a row in `data/processed/manifest.csv` recording `source_file`, `origin_dataset` (ISDB/SACAC), `loop_id` (group key for CV), `folder_label` (renamed from `label` to make clear it's not the training target).
- **SACAC stays untouched by any tuning** — final honest test, both for the classic detector's threshold choice and the RF.
- **Class balance check** happens on the classic detector's *derived* labels, not the folder labels — the actual distribution the model will learn from.
- **Simulator-readiness**: unchanged — PRD FR-11 synthetic output can be dropped into the same loader later.

## 6. Preprocessing

1. Load raw `PV`, `OP` columns per file.
2. Slice into fixed-size windows (inherit `n=100` from the thesis as a starting point — flagged as an open item in §13, and now also relevant to whether a window reliably contains a full oscillation cycle for the classic detector to work on, see §13).
3. Per-window z-score normalize each of PV and OP independently — needed both for the ML features (unit-agnostic across sensor types) and arguably for the ellipse-fit/Kano detectors, which are typically described against normalized/scaled OP-PV plots in the literature.
4. Run the classic detector (§3–4) on each normalized window to derive its label.
5. Extract tsfel features on the same normalized windows for the ML side.

## 7. Classic detector implementation (`classic.py`) — status: implemented, validated against real data

- `ellipse_stiction_index(pv_window, op_window) -> float`: **not** a literal algebraic ellipse fit. Measures the point cloud's actual OP-axis width in a horizontal band around PV's median instead of fitting a parametric ellipse — chosen deliberately, since the literature itself notes ellipse fitting struggles when the true shape is closer to a parallelogram (which is what stick-slip limit cycles often look like), and a direct point-cloud measurement is more numerically stable on small, noisy windows than algebraic conic fitting. Documented as a methodological simplification, not a reproduction of He et al.'s exact algorithm.
- `kano_pattern_check(pv_window, op_window) -> bool`: simplified proxy for Kano's qualitative pattern check — segments the window into OP-direction "strokes" and flags a stroke as stick-slip if most of its PV movement is near-zero with at least one dominant catch-up jump. Not a reimplementation of the original pattern taxonomy.
- `has_sufficient_activity(op_raw, reference_op_std) -> bool`: **found during sanity-testing, not planned upfront.** Per-window z-score normalization divides by that window's own std — a window where OP barely moved at all gets its tiny noise rescaled to unit variance, which can look like structure to the shape detectors. This function is a caller-side pre-filter (needs file-level context a pure per-window function shouldn't own) comparing a window's raw OP std against a wider reference (e.g. the full source file's OP std).
- `is_likely_quantized(pv) -> bool`: **found while validating against SACAC's own `quantisation-*` file.** A quantized signal produces the exact same "flat, then jump" step shape as real stick-slip — OP moves continuously while PV sits at a level then jumps to the next one. What actually tells them apart: quantization revisits a small fixed set of levels repeatedly, while a genuinely stuck valve settles wherever friction happened to catch it — effectively continuous, rarely exactly repeated. Measured as the fraction of unique PV values in the window. On real SACAC data: stick-slip windows had 46–77% unique values; the quantization file had 6%. `kano_pattern_check` now excludes quantized-looking windows. Without this guard, every window of the `quantisation-Q-paper-horch-2003.csv` file was a false positive.
- `derive_label(pv_window, op_window, ellipse_threshold) -> Literal["yes", "no", "uncertain"]`: combines both; "uncertain" windows are excluded from RF training data (not imputed, not guessed).
- `label_window(pv_raw, op_raw, ellipse_threshold, reference_op_std) -> Literal[...]`: the actual production entry point — raw (non-normalized) window in, label out. Runs the activity guard on raw data first, then normalizes and calls `derive_label`. Exists so callers can't accidentally get the activity-guard/normalization order wrong.
- **`ellipse_threshold` tuning (milestone 3, `scripts/label_windows.py`)**: the first approach tried — search for the threshold that maximizes agreement (or F1) between `ellipse_verdict` and `kano_verdict` on ISDB — failed. kano's positive rate is only ~15%, so "always predict False" already scores ~85% raw agreement; the search degenerately converged near the 99th percentile of the ellipse-index distribution, collapsing the result to 1 "yes" out of 3029 ISDB windows. Root cause: the two detectors are *supposed* to disagree often (§3 — they catch different failure modes), so optimizing one against the other as if they should align doesn't make sense. **Resolved**: `ellipse_threshold` is set as a low, round percentile (10th) of the ellipse-index distribution on ISDB's active windows — its real job is a fine-grained backstop beyond `has_sufficient_activity`'s per-file check (drop windows with negligible width even when their file passed the file-level guard), not a second independent stiction classifier. Deliberately *not* chosen by searching for whatever value maximizes agreement with the old folder labels either — that would quietly reintroduce the exact thing this redesign exists to avoid trusting. Final value: `ellipse_threshold=0.3762`.
- Validated against SACAC's literature-tagged files (`tests/test_classic_real_fixtures.py`, real data, skipped if not imported): `stiction-F-paper-horch-2003.csv` — stick-slip detected in 11/11 windows. `stiction-P-oilgas-DB-1-baccidicapaci-2018.csv` — detected in only its last 2 of 7 windows; PV sits essentially flat for the rest of the file despite being folder-labeled "yes" for its entire duration. **This is a real, observed instance of exactly the file-vs-window label mismatch §2 predicted**, not just a theoretical concern. `saturation-T-oilgas-thornhill-2002.csv` and `quantisation-Q-paper-horch-2003.csv` (both folder-labeled "no", different fault types) — correctly 0/14 and 0/11.
- This module is written so it could be lifted directly into the PRD's V1 real-time service later — same interface, no ML dependency, intentionally kept dependency-light (numpy/scipy only).

## 8. Feature engineering (for the RF side) — status: implemented, run on full dataset

- Use tsfel's statistical + temporal feature domains, as before.
- Correlation-based feature pruning (threshold ~0.9) to control overfitting risk on a small, now-further-filtered (uncertain windows dropped) dataset.
- Feature list saved in the model artifact's metadata — never hardcoded separately in an inference script.
- **Performance**: calling tsfel once per window measured at ~1.4s/window (~55 minutes for this corpus). Fixed by normalizing each window independently, concatenating normalized windows back-to-back, and letting tsfel do its own internal windowing in one batched call per run — since every concatenated segment is exactly `window_size` samples, tsfel's internal split at the same size recovers identical boundaries, just ~125x faster (~11ms/window batched vs. ~1.4s per-window). Full corpus (2328 windows) extracts in under a minute.
- **Results** (`scripts/extract_features.py` → `data/processed/features.csv`): 90 raw tsfel features (statistical + temporal, both PV and OP) pruned to 58 after correlation filtering (threshold 0.9). 96 of 2328 confident windows (4.1%) dropped for NaN features — **all 96 were "no"-labeled windows**, not a random sample: `derive_label`'s "no" case requires both detectors to agree there's minimal width/activity, which selects for the flattest, most near-constant windows — exactly where tsfel's skew/kurtosis calculations hit numerical instability (catastrophic cancellation on near-zero-variance signals). Expected given the definition, not a bug, but shifts the positive rate slightly (18.3% → 19.1%). Final dataset: 2232 windows (ISDB: 1203 no / 255 yes: SACAC: 602 no / 172 yes).

## 9. Model selection — status: implemented, trained

- **Primary**: `sklearn.ensemble.RandomForestClassifier`, trained against the classic detector's confident-agreement labels.
- **Secondary comparison (optional)**: `HistGradientBoostingClassifier` (see §14 for why over LightGBM), same setup. Not yet run.
- Hyperparameter search: `RandomizedSearchCV` (n_iter=30) over `n_estimators`/`max_depth`/`min_samples_leaf`/`max_features`, scored with `average_precision` (PR-AUC — threshold-independent, appropriate for ~18% positive rate) via `StratifiedGroupKFold` — never on SACAC.
- Imbalance handling: `class_weight='balanced'`. Not escalated further — CV/test metrics below didn't show a need.
- **Bug found and fixed**: plain `GroupKFold` (as originally planned) doesn't consider class labels at all. Checked on the real ISDB training data and found one fold with **zero** positive windows and two others with only 3–5, while two folds had 113+ each — positives are concentrated in a handful of the 77 ISDB loops. That makes per-fold precision/recall/PR-AUC meaningless for most folds and silently biases model selection toward whichever positive-heavy folds happen to dominate. Switched to `StratifiedGroupKFold` (same grouping guarantee, plus class balance across folds): 45–65 positives per fold on the same data. Regression-tested (`tests/test_models.py`) against a synthetic reproduction — including confirming the fix has a real mathematical limit: if positives live in fewer distinct groups than there are folds, *no* splitter can avoid empty folds (pigeonhole principle), so this only works because ISDB's positives, while concentrated, still span enough distinct loops.

## 10. Evaluation & reproducibility — status: baseline trained and evaluated

- **Metrics reported**, computed against the classic-detector labels (the actual training target): precision, recall, F1, PR-AUC, ROC-AUC, confusion matrix, on StratifiedGroupKFold-CV and on the untouched SACAC set.
- **Reported separately, not as a training metric**: agreement rate between (a) classic detector's verdict and the old folder labels (§7), (b) RF's verdict and the old folder labels — both framed as sanity checks, per §4.
- **Model artifact**: joblib bundle with `{model, feature_names, window_size, normalization, label_source, classic_detector_threshold, predict_threshold, sklearn_version, git_commit_hash, best_params, cv_metrics, cv_sanity_check_agreement, test_metrics, test_sanity_check_agreement, trained_at}` — self-describing, so a deployed model can't silently drift out of sync with what it was trained on (unlike the thesis's `subscribe.py`, where the feature list lived separately from the model file). `inference.py`'s `load_artifact`/`predict_window` is the one sanctioned way to consume it — raw window in, `{"label", "probability"}` out, re-deriving normalization and feature selection internally rather than trusting a caller to get that order right.
- **Versioning**: directory-based registry, `models/<date>_<git-short-hash>/`, mirrored to `reports/<date>_<git-short-hash>.json` for the metrics alone — no MLflow.
- **Tests**: feature pipeline determinism, `classic.py` correctness against real fixtures (§7), `StratifiedGroupKFold` non-degeneracy regression test (§9).
- **Baseline results** (`python -m valve_stiction_ml.train`, best params `n_estimators=300, min_samples_leaf=4, max_features='sqrt', max_depth=20`; after §4.1's SACAC additions):

  | | ISDB (StratifiedGroupKFold CV) | SACAC (held-out test) |
  |---|---|---|
  | Precision | 0.733 | 0.866 |
  | Recall | 0.839 | 0.522 |
  | F1 | 0.782 | 0.651 |
  | ROC-AUC | 0.944 | 0.865 |
  | PR-AUC | 0.809 | 0.788 |
  | Confusion (TN/FP/FN/TP) | 1125/78/41/214 | 588/15/89/97 |
  | Sanity-check agreement with folder_label | 85.5% | 89.0% |

  Precision and recall trade off differently between ISDB-CV and SACAC — SACAC has far fewer false positives (15) but misses more true positives (89 FN). That's a real, honestly-reported generalization gap between two independently-sourced benchmark corpora, consistent with what §3 already set as the expectation (this isn't a SOTA claim, and even the classic detector it's approximating isn't perfect ground truth). ROC-AUC/PR-AUC (threshold-independent) stayed reasonably strong on both (0.865/0.788 on SACAC), suggesting the 0.5 decision threshold — not the model's underlying discrimination — is what's driving the precision/recall trade-off shift; not re-tuned for this baseline. (Before the §4.1 SACAC additions: 0.860/0.772 — the extra data moved things slightly, not dramatically, which is the honest expectation for +15 net confident windows on a ~774-window test set.)

### 10.1 Clustering as a sanity check (not a modeling method) — status: run, result is a genuine caveat, not a clean pass

Ran KMeans and GMM (k=2) on the same standardized tsfel features used to train the RF (`scripts/cluster_sanity_check.py`), independent of any label, then checked whether the classic detector's yes/no windows fall into separate clusters.

**They don't** — this came back as the "don't separate" case, not the hoped-for clean one:

| | vs. `derived_label` (classic detector) | vs. `folder_label` (old thesis) | vs. `origin_dataset` |
|---|---|---|---|
| KMeans, Adjusted Rand Index | 0.005 | 0.039 | -0.001 |
| GMM, Adjusted Rand Index | -0.079 | — | — |

All close to 0 (chance-level); GMM is even slightly *below* chance. The PCA scatter (`reports/cluster_sanity_check.png`) confirms it visually — "yes" windows are scattered diffusely through the "no" cloud, no visible separation, while KMeans/GMM instead split the data along some other axis entirely.

**Investigated why, rather than stopping at the number**: checked what that dominant axis actually correlates with. Not `origin_dataset` (ARI ≈ 0 — not an ISDB-vs-SACAC artifact). The top PC1 loadings are dominated by generic magnitude/spread features — `PV_Area under the curve`, `PV/OP_Peak to peak distance`, `PV/OP_Mean absolute deviation`, `PV/OP_Interquartile range` — i.e. "how much did this window move overall," not shape-specific stick-slip structure. That's consistent with everything else this investigation already found: `ellipse_stiction_index` (a general width/activity measure) is lenient and true ~91% of the time, while `kano_pattern_check` (the actual shape-specific test) fires on only ~17% — the stiction-relevant signal is a comparatively small, specific slice of the feature space's total variance, not its dominant axis. A k=2 unsupervised split naturally finds the *dominant* axis first, which here is generic activity level, not stiction shape.

**What this means, honestly**: this is weaker evidence for the feature pipeline than hoped, but not a reason to distrust the RF result on its own. RF is supervised and nonlinear (many trees, entropy-based splits) — it can exploit a specific, lower-variance combination of features that a simple k=2 unsupervised split won't surface as the top axis, and its ROC-AUC/PR-AUC on held-out SACAC (0.86 / 0.77) show it found *something* real. But this check doesn't independently corroborate that finding the way a clean cluster separation would have. Documented as a genuine, moderate limitation for the writeup — not spun as either "fine" or "broken."

## 11. Project structure

```
valve-stiction-ml/
  pyproject.toml
  README.md
  docs/
    ML_PLAN.md              (this file)
  configs/
    default.yaml             (window size, classic-detector threshold, model hyperparams, paths)
  data/
    raw/                     (gitignored — imported CSVs)
    processed/
      manifest.csv
  src/valve_stiction_ml/
    __init__.py
    dataset.py                (loading, windowing, manifest)
    classic.py                 (ellipse-fit + Kano detector — label source, reusable for PRD V1)
    features.py                 (per-window normalization + tsfel extraction)
    models.py                    (RF training, optional GBM baseline)
    evaluate.py                   (metrics, plots, cross-dataset test, sanity-check agreement reporting)
    train.py                      (CLI entrypoint, config-driven, writes model artifact)
    inference.py                   (load a trained artifact, score one raw window)
  scripts/
    import_thesis_data.py          (one-off: copy + manifest thesis CSVs)
    download_additional_sacac.py    (one-off: pull in official-portal SACAC files missing from the thesis copy, see §4.1)
    label_windows.py                  (milestone 3: classic detector -> window_labels.csv)
    extract_features.py                (milestone 4: tsfel -> features.csv)
  models/                           (gitignored large artifacts, or git-lfs)
  reports/                           (generated per training run: metrics.json, plots)
  notebooks/                         (exploration only — never the source of truth)
  tests/
    test_dataset.py
    test_classic.py
    test_classic_real_fixtures.py    (real SACAC-tagged files, skipped if data not imported)
    test_features.py
    test_evaluate.py
    test_models.py
    test_inference.py
```

## 12. Milestones (reordered — classic detector now comes before any ML)

1. ✅ Scaffold repo, import thesis data + build manifest (folder labels imported as sanity-check field only). 115 files (77 ISDB + 38 SACAC) → 4453 windows.
2. ✅ Implement `classic.py` (ellipse-fit primary, Kano secondary) + real-data test fixture from SACAC's tagged files. Two real bugs found and fixed during validation, not anticipated in the original plan — see §7: quantization/stick-slip confusion, and per-window normalization amplifying noise in inactive windows. Also empirically confirmed the §2 file-vs-window label mismatch on a real file.
3. ✅ Ran the classic detector across all 4453 windows (`scripts/label_windows.py` → `data/processed/window_labels.csv`). Results:

   | Dataset | no | uncertain | yes | uncertain rate |
   |---|---|---|---|---|
   | ISDB | 1286 | 1488 | 255 | 49.1% |
   | SACAC | 615 | 637 | 172 | 44.7% |

   2328 confident windows total (1901 no / 427 yes, ~18.3% positive rate) survive for RF training. Sanity-check agreement with the old folder labels, on confident windows only, **not used to pick any parameter**: 86.8% overall (87.0% ISDB, 86.3% SACAC) — reassuring: where the new method is confident, it mostly agrees with the original file-level labels, and the difference is in correctly abstaining on ambiguous windows rather than blindly propagating a file's label to all of them. Checkpoint verdict: uncertain rate is substantial but not disqualifying — see §13.
4. ✅ Preprocessing + tsfel feature pipeline, with tests. 90 raw features → 58 after correlation pruning; 2232 confident windows survive (see §8).
5. ✅ Baseline RF trained against classic-detector labels: StratifiedGroupKFold CV on ISDB (fixed a real fold-degeneracy bug along the way, see §9), final untouched evaluation on SACAC. Results in §10.
6. ✅ Sanity-check agreement (RF vs. old folder labels) reported in §10 — not the headline metric.
7. ✅ Clustering sanity check (§10.1): KMeans/GMM on the same standardized features, check separation against classic-detector labels, report as a figure. Result: no separation (ARI ≈ 0) — a genuine, investigated caveat, not a clean pass. See §10.1 for what the dominant clustering axis turned out to be instead.
8. Optional: gradient-boosting comparison.
9. ✅ Correlation-based feature pruning — done as part of milestone 4 (§8), folded in rather than a separate later pass.
10. ✅ Finalized model artifact format (`predict_threshold` added) + `inference.py` (`load_artifact`/`predict_window`, tested including a real end-to-end run against the trained model on a real SACAC window) + top-level README.md (setup, full pipeline reproduction commands, results, known limitations, inference usage).
11. Later, once V1's simulator exists: same pipeline, new data source; `classic.py` can also be lifted directly into the PRD's real V1 service at that point.

## 13. Open items / risks

- **Window size — investigated, kept at 100.** File lengths in the corpus range from 200 to 42,512 samples (median 1441) — much more heterogeneous than assumed. Autocorrelation-based period estimation found 40% of files (with detectable periodicity) have an oscillation period longer than 100 samples, i.e. a single window can't even span one full cycle for a meaningful chunk of the corpus — real motivating evidence, not just a theoretical concern. Tried window sizes 100/150/200/300 through the actual labeling pipeline expecting larger windows to reduce the uncertain rate (more room for kano to see a complete cycle). **Result went the other way**: uncertain rate rose monotonically with window size (49%→52%→56%→59% on ISDB), and usable confident windows dropped sharply (2328→1406→962→560) as fewer non-overlapping windows fit per file. Likely mechanism: `kano_pattern_check` requires a *majority* of a window's OP-direction strokes to show the stick-slip signature — a larger window contains more strokes, making a clean majority harder to sustain across natural signal variability, independent of whether more of any single cycle is captured. (Sanity-check agreement with old folder labels did rise with window size, 87%→93%, but that's a secondary metric, not the tuning target.) **Decision: kept window=100** — the evidence doesn't support a larger window given kano's current majority-vote design, and 100 yields far more usable data. A more correct fix (per-loop-adaptive window sized to each loop's own estimated period, or changing kano's stroke-aggregation rule) is a real option but adds real scope; not pursued now. Revisit only if milestone 5 surfaces evidence current windowing specifically hurts model quality.
- **Discovered while investigating window size**: per-loop window counts are wildly imbalanced under fixed non-overlapping windowing — file lengths span 200 to 42,512 samples, so the longest file (`ISDB/no/BAS8`, 42,512 samples) contributes ~425 windows on its own, vs. 2 windows from the shortest. GroupKFold prevents this from leaking across train/test, but a handful of very long recordings could still dominate what the model actually sees during training. Worth checking at milestone 5 (e.g. cap windows sampled per loop, or inspect per-loop contribution to the trained model) — not blocking, but flagged so it isn't a surprise later.
- **Ellipse-fit threshold** is tuned on ISDB only (10th percentile of active-window ellipse-index, = 0.3762); document it as a chosen convention, not a physical law — same "apparent stiction" caveat the literature itself flags.
- **Classic detector ceiling**: even modern non-DL literature tops out around 85-90% on ISDB. Don't oversell derived labels as ground truth in the writeup — they're a principled proxy, documented as such.
- **"Uncertain" windows get dropped — measured, not hypothetical**: 49.1% (ISDB) / 44.7% (SACAC). 2328 confident windows remain (427 positive, ~18.3% rate) — workable but not large. Deliberately not loosened by weakening the agreement criterion (e.g. confidence-weighted instead of hard agree/disagree) purely to keep more data — that would trade away the conservatism this whole approach exists for. If milestone 5's RF training turns out data-starved, revisit then with evidence in hand, not preemptively.
- **Investigated and confirmed the uncertain rate's exact mechanism** (not left as an unexplained number): on active windows, `ellipse_verdict` is True 91% of the time (by construction — the threshold is the 10th percentile) while `kano_verdict` is True only 17% of the time. The crosstab: `{ellipse=True, kano=False}` = 2093 windows, accounting for 98% of all uncertain windows on its own. At this lenient threshold, ellipse mostly answers "is there any real oscillation at all" (true for nearly everything in this already-oscillating benchmark corpus), so kano is doing almost all the actual shape discrimination — meaning most of "uncertain" is really "kano said no, and ellipse's lenient yes didn't confirm or deny that." Considered making kano the decisive signal (ellipse as a coarse floor only), which would recover most of those 2093 windows as confident "no" — **decided against**: those windows are honestly ambiguous (kano is a simplified heuristic, not verified specific enough to trust alone for a confident negative), and the point of pairing two detectors was exactly to avoid over-trusting either one individually. Revisit only if milestone 5 shows the dataset is too small to train on, with that evidence in hand.
- **Horch's method excluded** from the primary label source due to its flow-loop-only, no-integrator constraint, which conflicts with the general-model goal — but it's cheap to add later as an extra signal specifically for flow loops if the ISDB manifest shows enough of them.
- **License** for the new repo not yet decided (MIT typical default) — flagging, not blocking.

## 14. Resolved (was: open questions)

- **Test fixture for `classic.py`**: no manual eyeballing needed from you. At implementation time, pull a handful of windows from SACAC's literature-tagged files — `stiction-*.csv` for easy positives, `saturation-*.csv`/`quantisation-*.csv` for *hard negatives* (still oscillating/abnormal, but not stiction — a meaningfully harder test than a calm normal file), plus one calm file for an easy negative. Plotted and visually confirmed during implementation, shown to you at that point, not decided now.
- **Window size**: keep 100 samples for both the classic detector and ML features for v1 — don't split into two tunable sizes preemptively. Validate empirically at milestone 3 whether 100 samples reliably contains a full oscillation cycle; split only if that check fails. Flagged: ISDB spans many industrial loops that likely differ in sampling rate, so "100 samples" may represent different real-world durations per loop — inherited from the thesis, not new, but worth checking in the data audit.
- **Secondary model**: `HistGradientBoostingClassifier` (sklearn, no new dependency) over LightGBM — easier to justify in an interview for what's explicitly an optional comparison, and avoids adding a dependency for a nice-to-have.
