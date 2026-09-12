AF alignment: experimental code and reproducibility materials
============================================================

This export contains computational materials only. It contains no manuscript,
review, notes, raw records, per-record predictions, videos, or model binaries.
Retained CSV results preserve the numerical evidence, including unfavorable and
uncertain contrasts. They must not be overwritten by a new execution.

Start with ``REPRODUCIBILITY.rst``. ``provenance/source_mapping.csv`` maps ported
code and retained outputs to producing-source checksums. This is a portable
export, not a claim that its edited scripts are the original historical files.

Quick verification without data or training::

    python -m venv .venv
    . .venv/bin/activate
    python -m pip install -r requirements.txt
    python scripts/safety_audit.py
    python scripts/verify_export.py
    python analysis/subgroup_and_stress.py
    python analysis/mechanism_summary.py --include-dose

Figures and recomputed tables go to ignored ``outputs/`` directories. The
commands above verify arithmetic and code paths; they do not reproduce training
or establish patient-population inference. Installing dependencies on a different
platform may require platform-compatible wheels; no clean-install claim is made.

Structure
---------

* ``src/af_alignment/``: v2b, losses, frozen probe, features, stress and geometry.
* ``experiments/``: explicit-input preprocessing, locked training and evaluation.
* ``configs/``: retained settings, seeds, input fingerprints and model routes.
* ``analysis/``: paired uncertainty, simulations, subgroup/calibration/mechanism
  calculations and figure generation.
* ``results/``: small retained aggregate and seed-level records, not patient data.
* ``scripts/``: verification and public-file safety audit.
* ``provenance/``: source mapping, hashes, test record and inclusion inventory.

Scientific boundaries
---------------------

The literal three-arm experiment uses PTB-only, PTB+EF, PTB+EF+symmetric
alignment. In the later routing experiments, **all arms receive EF supervision**;
``none`` means no alignment penalty. It does not mean PTB-only. The two
families also use different checkpoint selection functionals. Do not merge them
as one effect or substitute routing results for historical r32/r128 results.

Seed is the comparison replicate. Mean intervals are nominal paired-t references
on fixed evaluation records, not selection-adjusted, simultaneous, clinical, or
joint patient-population intervals. Sign symmetry is an assumption, not a
randomization design. Mechanism associations do not identify mediation.

Release status
--------------

This is a local, unpushed export. Reviewer/public access has not been arranged.
No redistribution permission for input data, derived patient representations,
or model binaries is assumed. Authors must choose a code license and arrange
permitted artifact access before public release. No original project files are
included by symlink, moved, or changed.
