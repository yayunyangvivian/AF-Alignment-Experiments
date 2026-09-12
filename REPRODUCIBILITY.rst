Workflow and reproduction levels
================================

There are distinct reproduction levels: arithmetic from seed scores; downstream
evaluation conditional on representations; checkpoint forward agreement; new
training under retained settings; and authentication of historical execution.
Passing a check at one level does not establish the next. No encoder training was
rerun when preparing this export.

Environment
-----------

``requirements.txt`` records observed Python 3.11 training/scoring packages.
``requirements-preprocessing.txt`` records the separate waveform/video
environment. Use separate virtual environments. In particular, the recorded
NumPy/scikit-learn/Torch versions differ between the two environments. Small
floating-point differences are possible across versions/platforms. Hardware,
decoder versions and randomness can matter. A current installation is not an
authenticated historical environment for every experiment.

Inputs and access
-----------------

Acquire PTB-XL, EchoNet-Dynamic and the Georgia subset of the PhysioNet/CinC
Challenge 2021 through their dataset providers, accepting applicable terms.
The retained Georgia source route is
https://physionet.org/files/challenge-2021/1.0.3/training/georgia/ .
No download, access approval or redistribution permission is implied here.

Supply inputs in excluded ``private_inputs/`` directories:

* ``ptb/``: ``emb_ecg.npy``, ``tab.npy``, ``y_af.npy``, ``ids_train.txt``,
  ``ids_val.txt``, ``ids_test.txt``, ``test_patient_ids.npy``. The patient array
  maps test-index order, not all dataset rows. ``norm_ecg.npz`` is additionally
  required for frozen Georgia preprocessing. Retain the existing PTB split and
  transform; do not fit a replacement transform on Georgia.
* ``echo/``: ``emb_echo.npy``, ``y_ef.npy``, ``ids_echo_train.txt``,
  ``ids_echo_val.txt``, ``ids_echo_test.txt``. The documented extraction uses
  R3D-18 KINETICS400_V1, 32 frames, evaluation/no-grad mode and 512 pooled
  coordinates after replacing the classification layer with identity.
* ``georgia/``: the complete WFDB release, 10,344 records before exclusions.
  The frozen adapter retains 10,338 records, with 741 AF/AFL-positive records.
  Six nonfinite records were excluded. Record IDs are not patient identities.
* Trusted model folders: ``selected_model.pt`` and ``probe_parameters.npz``
  for each routing arm and seed. Checkpoints are not included in this export.

``configs/retained_input_hashes.json`` identifies the PTB/Echo inputs actually
fingerprinted in the rebuilt three-arm producing configuration. Runners reject
different inputs rather than silently calling another configuration a
reproduction. The normalizer, Echo validation/test identifiers and Georgia raw
records are not all covered by that manifest. Their exact historical link must
be checked against permitted supporting artifacts before claiming exact replay.

Feature reconstruction (separate preprocessing environment)::

    PYTHONPATH=src python -m af_alignment.preprocess_ptb --ptb-root private_inputs/ptb_raw --out-dir outputs/ptb_new
    python experiments/extract_echo.py --data private_inputs/echo_raw --weights private_inputs/r3d_weights.pth --output outputs/echo_new
    python experiments/prepare_georgia.py --data private_inputs/georgia --normalizer private_inputs/ptb/norm_ecg.npz --output outputs/georgia_features --stresses

PTB preprocessing is a new reconstruction, not authentication of the historical
export. It does not supply the historical test-patient mapping automatically;
that mapping must match the split and row order. Echo extraction checks the
retained full weight hash; a new execution still requires output comparison with
the input manifest before it is substituted in a locked training experiment.
The portable Echo runner processes videos serially instead of the original five
workers, keeping the per-video feature calculation and one Torch thread fixed.

Training families
-----------------

The following commands without ``--run`` only check required input hashes::

    python experiments/train_three_arm.py --ptb-dir private_inputs/ptb --echo-dir private_inputs/echo --output outputs/three_arm
    python experiments/train_routing_initial.py --ptb-dir private_inputs/ptb --echo-dir private_inputs/echo --output outputs/routing_initial
    python experiments/train_routing_added.py --ptb-dir private_inputs/ptb --echo-dir private_inputs/echo --output outputs/routing_added
    python experiments/train_routing_dose03.py --ptb-dir private_inputs/ptb --echo-dir private_inputs/echo --output outputs/routing_dose03

Add ``--run`` only when intentionally launching a new training execution, with
sufficient resources. These commands are not downloads of retained checkpoints.
All output directories must be new/empty. Budget exhaustion is a failed partial
execution, never grounds for replacing seeds or reporting a complete result.

* Literal three-arm: seeds 52--61, 30 fits; PTB-only / PTB+EF /
  PTB+EF+symmetric alignment. Validation AF trained-head AUC chooses the earliest
  maximum. Both contrasts are retained with all four endpoints.
* Initial routing: seeds 77--81, 15 fits, EF supervision in all arms.
* Added routing: seeds 82--86, 20 fits including the separate anchored-no-MMD
  ablation. The principal three routing arms thus have ten seeds each.
* Dose challenge: seeds 77--86, symmetric/anchored coefficient 0.3, 20 fits.
  This is not a search for an optimal coefficient.

Routing selection uses validation frozen-probe AUC, earliest maximum. All
packages use dimension128, 30 epochs, 17 PTB updates/epoch, batches1024/256,
AdamW learning rate .001/weight decay .0001, cosine schedule, PTB-only common
variance/covariance penalties and training-standardized EF. Exact settings and
cohort distinctions are in ``configs/``. Training-loop and downstream numerical
bodies are retained; bootstrap/path/source-hash setup was ported. The three-arm
runner and routing runner are deliberately not collapsed into one configurable
training loop. The initial five-seed literal experiment is retained separately
as scalar evidence; the seeds52--61 runner is not its producing execution.

Frozen evaluation and diagnostics
--------------------------------

``configs/models_main.json`` maps the 30 principal routing models to new-run
output locations. ``models_with_dose.json`` additionally maps the 20 dose models.
``models_with_ablation.json`` adds the five no-MMD models. Adjust directory
entries to supplied permitted retained model folders when reproducing frozen
outputs; do not change seed/arm identities. Paths are relative to the registry.

Example frozen scoring, with no fitting on Georgia::

    python experiments/evaluate_frozen.py --features outputs/georgia_features/external_inputs.npz --models configs/models_main.json --output outputs/georgia_clean
    python experiments/evaluate_frozen.py --features outputs/georgia_features/stress_features.npz --models configs/models_main.json --output outputs/georgia_stress

Frozen probes use row normalization, training StandardScaler, balanced logistic
regression, then a separately validation-fitted probability calibrator. Exported
scoring preserves float32 in-place scaling. Raw and calibrated Brier are
different procedures. There is no Georgia-specific recalibration.

To compute subgroup/calibration diagnostics from newly scored arrays, place
``predictions.npz`` from clean scoring under the runtime input name
``clean_predictions.npz``, alongside ``record_metadata.csv`` from preprocessing.
This is a runtime copy, not a public artifact or historical recovery. Then::

    AF_ANALYSIS_OUTPUT=outputs/prediction_diagnostics python analysis/prediction_diagnostics.py --input private_inputs/georgia_evaluation
    AF_EVALUATION_INPUT=outputs/prediction_diagnostics AF_ANALYSIS_OUTPUT=outputs/calibration_audit python analysis/calibration_audit.py

Calibration intercept/slope on Georgia are diagnostic only; probabilities and
Brier scores are not refitted. Conduction groups use the retained diagnosis
code list; age/sex groups and display counts remain fixed. Seed, not record,
is the independent unit of model-comparison summaries.

Frozen geometry requires both base and dose model mappings plus PTB/Echo inputs::

    AF_ANALYSIS_OUTPUT=outputs/geometry python analysis/representation_diagnostics.py --dose 0.1 --models configs/models_with_dose.json --ptb-dir private_inputs/ptb --echo-dir private_inputs/echo --features outputs/georgia_features/stress_features.npz --reference-predictions outputs/georgia_stress/predictions.npz
    AF_ANALYSIS_OUTPUT=outputs/geometry python analysis/representation_diagnostics.py --dose 0.3 --models configs/models_with_dose.json --ptb-dir private_inputs/ptb --echo-dir private_inputs/echo --features outputs/georgia_features/stress_features.npz --reference-predictions outputs/georgia_stress/predictions.npz

The first pass asserts forward agreement with the supplied base stress
predictions. The diagnostics retain common subset draws, native MMD, effective
rank, class separation and paired representation movement. They do not establish
causal mediation. Stress definitions are fixed before feature calculation and
shared across arms. Smaller degradation does not establish better absolute
stressed performance.

Arithmetic and figures from included materials
----------------------------------------------

No data/checkpoints are needed for these examples::

    python analysis/paired_results.py --scores results/three_arm/per_seed_arm_metrics.csv --candidate ptb_ef --reference ptb_only --output outputs/ef_added
    python analysis/paired_results.py --scores results/three_arm/per_seed_arm_metrics.csv --candidate ptb_ef_alignment --reference ptb_ef --output outputs/alignment_added
    python analysis/subgroup_and_stress.py
    python analysis/mechanism_summary.py --include-dose
    python analysis/historical_calibration_ablation.py --brier results/historical_calibration/config2_brier_per_arm.csv --ablation results/p5_historical/ablation_results.csv --output outputs/historical_sensitivity

``analysis/common.py`` copies immutable seed summaries to the ignored working
directory before recomputation. Set ``AF_ANALYSIS_OUTPUT`` to choose a new
directory. ``AF_EVALUATION_INPUT`` can point to a supplied compatible evaluation
directory. The retained CSVs are never overwritten. Source code generates PNG/PDF
figures in ignored output directories; no PDF is part of the public inventory.

Historical component diagnostics have explicit archived-input routes::

    python analysis/component_probe_diagnostic.py --archive private_inputs/historical_archive --output outputs/components_new
    python analysis/rank_performance_check.py --archive private_inputs/historical_archive --output outputs/ranks_new

The required historical directory/configuration mapping is retained in those
scripts. Do not substitute current generic checkpoints for original/r32 tagged
representations. Covariance-only head outputs use a different evaluation
functional from the frozen logistic probes. The supplied component and rank
tables omit only local path columns; scientific cell values are retained.

The recorded P1 new-refit route is available, but was NOT run in this export
pass and is not recovery of the original historical P1 model::

    python analysis/p1_refit.py --ptb-dir private_inputs/ptb --representations private_inputs/clean_exports --output outputs/p1_new
    python analysis/p1_score_refit.py --predictions outputs/p1_new/p1_refit_predictions.npz --output outputs/p1_scores

This requires matching clean-export configurations, metadata and split arrays.
It fits downstream predictors (not encoders), preserves the recorded seed42
fitting order, and writes a new probability artifact. P1's MLP is distinct from
the common frozen logistic probe in the routing analysis.

Simulations are separate, explicit computations (not run during export)::

    python analysis/fixed_benchmark_simulation.py --output outputs/fixed_simulation
    python analysis/two_component_simulation.py --output outputs/two_component_simulation

The first conditions on a fixed benchmark and tests specified total-error laws.
The second preserves a prior illustrative two-component stream; its variance
scale is a heuristic and is not a validated decomposition of joint uncertainty.
Neither procedure establishes universal small-K mean coverage. Order-statistic
sign results target a median under the stated assumptions, separately from means.

Limits of this export
---------------------

Retained seed tables permit verification of arithmetic. Full scoring requires
excluded input arrays/model binaries, and full training requires dataset access
and computational resources. Preprocessing and checkpoint-forward commands are
provided but were not executed on full datasets in the export validation pass.
No end-to-end clean-room replay or public data release is claimed.

Historical original/r32 procedures remain incompletely authenticated. The older
r128 AUC/AP scalar record is separate from the later routing/three-arm packages.
Historical rank/component and calibration tables are retained as conditional
evaluation evidence. P1 recomputation is explicitly a new refit record, not a
recovered original downstream prediction vector. P5 historical scalar results
do not constitute the literal matched three-arm experiment. Their original
downstream models/prediction artifacts are not supplied or claimed recovered.
The export does not provide a validated joint patient-population interval.

Publication packaging
--------------------

No remote is configured, no commit or push has been made, and no public license
has been inferred. Create an empty appropriately owned GitHub repository, choose
a code license and review the explicit allowlist before uploading. Upload only
this export, never the original project. Data/model access must be arranged
separately in accordance with provider terms. The safety audit is a local scan,
not proof of redistribution rights or a guarantee against every secret format.
