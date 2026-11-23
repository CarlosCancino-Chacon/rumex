# Release Notes

## Version 0.1.0 (prerelease)

This is a first prerelease of the package. In this release configuration functionality is missing and the documentation is incomplete.

## Unreleased

* RuMEx no longer depends on the Basis Mixer. The note features (formerly basis functions) are computed with
  `partitura.musicanalysis.note_features` (`rumex.features`), the performance is decoded with the `partitura`
  performance codec (`rumex.render`), and the helpers for onsetwise aggregation come from `partitura`.
* Rules: `compute_increase_decrease_profile` (dynamics and tempo) drops the value of a ramp at the onset where it ends if a new constant marking starts there (e.g. a crescendo up to an f). Without this, the end value 1 of the `partitura` ramps made the profile restart at that onset and hold a constant offset for the whole next marking.
* License: GNU GPL version 3 or later (`GPL-3.0-or-later`), with copyright notices in the source files and the README.
* Requires Python >= 3.10, `partitura` >= 1.9, `numpy` >= 2 and `scipy` >= 1.13.
* Packaging with `pyproject.toml` (replaces `setup.py`); the `RuMExRender` command is a console entry point (`rumex.cli:main`), `bin/RuMExRender` is a thin wrapper for running from the source tree.
* `render_piece` and `load_score` moved from `bin/RuMExRender` into the package (`rumex.render`).
* Config key `basis_functions` is now `note_features` (the old key is still read).
* Compatibility with `partitura` 1.9, `numpy` 2 and recent `scipy`.
* Differences to the previous version due to the `partitura` features: the end note of a crescendo/diminuendo or
  accelerando/ritardando now has the value 1 of the ramp (it was 0); markings of impulsive dynamics such as `fz`
  appear as `unknown_impulsive`; `beat_phase` is available (the measure pattern rules need it).
