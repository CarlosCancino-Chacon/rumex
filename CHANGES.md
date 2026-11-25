# Release Notes

## Version 0.0.1 (prerelease)

This is a first prerelease of the package. In this release configuration functionality is missing and the documentation is incomplete.

## Version 0.1.0

* RuMEx no longer depends on the Basis Mixer. The note features (formerly basis functions) are computed with
  `partitura.musicanalysis.note_features` (`rumex.features`), the performance is decoded with the `partitura`
  performance codec (`rumex.render`), and the helpers for onsetwise aggregation come from `partitura`.
* Rules: `compute_increase_decrease_profile` (dynamics and tempo) drops the value of a ramp at the onset where it ends if a new constant marking starts there (e.g. a crescendo up to an f). Without this, the end value 1 of the `partitura` ramps made the profile restart at that onset and hold a constant offset for the whole next marking.
* Tempo markings: `tempo_markings_rules` now samples the tempo of each kind of marking only once per piece (before, every instance of a marking, e.g. each allegro, was sampled separately). New alternative rule `tempo_markings_variation_rules`: the tempo of each instance of a marking varies by a small proportion around the tempo of the marking (`variation`, default 0.03).
* Grace notes are no longer removed by `load_score` (new argument `grace_notes="keep"` (default) or `"remove"`). `render_piece` expands them in score time according to their type with `partitura.utils.music.expand_grace_notes_from_local_grace_order` (acciaccatura: before the beat, takes time from the previous note; appoggiatura: on the beat, takes time from the main note; the grace notes of the same type at an onset share `grace_offset_quarter`, default 1/4 quarter note), see `rumex.grace`. The rules see the expanded note array (`render_piece(..., expand_grace_notes=False)` disables it). The notes of the returned performance are ordered by score onset and pitch (`decode_performance` sorts them), and the alignment pairs the right ids.
* Double notes (same pitch and onset in different voices, which cannot be played twice and are a problem for mechanical pianos) are removed by `render_piece` (`remove_double_notes=True`, `keep_longer_double_note=True`; `rumex.double_notes_keep_mask`, built on `partitura.utils.music.remove_double_notes_from_score_note_array`). The returned `W`, `predictions` and performance only have the kept notes.
* The entries of a rule pipeline can now be the name of a rule, a function, or a tuple `(rule, {"parameter": value})`.
* New dynamics rule `melody_loudness_rules` (`melody_relative_loudness`: the melody, found with the skyline algorithm, is played louder or softer) and the skyline algorithms `skyline` and `monophonic_skyline` (strictly monophonic melody).
* Requires Python >= 3.10, `partitura` >= 1.9, `numpy` >= 2 and `scipy` >= 1.13.
* Packaging with `pyproject.toml` (replaces `setup.py`); the `RuMExRender` command is a console entry point (`rumex.cli:main`), `bin/RuMExRender` is a thin wrapper for running from the source tree.
* `render_piece` and `load_score` moved from `bin/RuMExRender` into the package (`rumex.render`).
* Config key `basis_functions` is now `note_features` (the old key is still read).
* Compatibility with `partitura` 1.9, `numpy` 2 and recent `scipy`.
* Differences to the previous version due to the `partitura` features: the end note of a crescendo/diminuendo or
  accelerando/ritardando now has the value 1 of the ramp (it was 0); markings of impulsive dynamics such as `fz`
  appear as `unknown_impulsive`; `beat_phase` is available (the measure pattern rules need it).
