# Copyright (C) 2020-2026 Carlos Cancino-Chacón
#
# This file is part of RuMEx.
#
# RuMEx is free software: you can redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either version 3 of the License, or (at
# your option) any later version.
#
# RuMEx is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License along with RuMEx. If not, see
# <https://www.gnu.org/licenses/>.

"""
Rendering of a performance from a score using the rule pipelines.

Only `partitura` is needed: the note features come from `partitura.musicanalysis.note_features` and the
expressive parameters are turned into a performance with the `partitura` performance codec
(`partitura.musicanalysis.decode_performance`).
"""
import numpy as np
import partitura
import partitura.score as score
from partitura.musicanalysis import decode_performance

from rumex.features import compute_note_features
from rumex.grace import NoteArrayScore, expanded_note_array, has_grace_notes
from rumex.doublenotes import double_notes_keep_mask
from rumex.dynamics import DynamicsRules, DEFAULT_PIPELINE as DEFAULT_DYNAMICS_PIPELINE
from rumex.tempo import TempoRules, DEFAULT_PIPELINE as DEFAULT_TEMPO_PIPELINE
from rumex.articulation import (
    ArticulationRules,
    DEFAULT_PIPELINE as DEFAULT_ARTICULATION_PIPELINE,
)
from rumex.timing import TimingRules, DEFAULT_PIPELINE as DEFAULT_TIMING_PIPELINE

RNG = np.random.RandomState(1984)

# `note_features`: "all" or a list of the names of partitura note feature functions
DEFAULT_CONFIG = dict(
    dynamics=dict(pipeline=DEFAULT_DYNAMICS_PIPELINE, note_features="all"),
    tempo=dict(pipeline=DEFAULT_TEMPO_PIPELINE, note_features="all"),
    timing=dict(pipeline=DEFAULT_TIMING_PIPELINE, note_features="all"),
    articulation=dict(pipeline=DEFAULT_ARTICULATION_PIPELINE, note_features="all"),
)

# The codec uses `beat_period`, `velocity`, `timing` and `articulation_log`; `velocity_trend` and `velocity_dev`
# (velocity = trend - dev) are kept as well, since the rules compute them
PARAMETER_NAMES = [
    "beat_period",
    "velocity_trend",
    "velocity_dev",
    "velocity",
    "timing",
    "articulation_log",
]


def load_score(filename, grace_notes="keep", **kwargs):
    """
    Load a score (MusicXML, MIDI, MEI, ...) as a single part.

    The part is the merge of all the parts of the score. Notes without an id get one (the rules need
    unique note ids).

    Parameters
    ----------
    filename : str
    grace_notes : "keep" or "remove"
        "keep" keeps the grace notes (`render_piece` expands them according to their type, see
        `rumex.grace`), "remove" removes them.
    """
    if str(filename).lower().endswith((".mid", ".midi")):
        scr = partitura.load_score_midi(filename, **kwargs)
    else:
        scr = partitura.load_score(filename, force_note_ids="keep", **kwargs)
    parts = scr.parts
    part = parts[0] if len(parts) == 1 else score.merge_parts(parts)
    if grace_notes == "remove":
        score.remove_grace_notes(part)
    elif grace_notes != "keep":
        raise ValueError("`grace_notes` must be 'keep' or 'remove'")
    return part


def _feature_subset(config):
    names = []
    for par in config.values():
        subset = par.get("note_features", par.get("basis_functions", "all"))
        if subset == "all":
            return "all"
        names += list(subset)
    return list(dict.fromkeys(names))


def render_piece(
    part,
    config=DEFAULT_CONFIG,
    rng=RNG,
    return_features=False,
    expand_grace_notes=True,
    grace_offset_quarter=0.25,
    remove_double_notes=True,
    keep_longer_double_note=True,
):
    """
    Render a performance of a score part.

    Parameters
    ----------
    part : partitura.score.Part
        Score (see `load_score`).
    config : dict
        For each of "dynamics", "tempo", "timing" and "articulation" a dict with the rule `pipeline`
        and the `note_features` ("all" or a list of partitura note feature functions).
    rng : np.random.RandomState
    return_features : bool
        Also return the note features, the expressive parameters and the alignment.
    expand_grace_notes : bool
        Give the grace notes a position and a duration in the score according to their type
        (appoggiatura, acciaccatura), see `rumex.grace`. Otherwise they have no duration.
    grace_offset_quarter : float
        Total duration (in quarter notes) of the grace notes of the same type at an onset.
    remove_double_notes : bool
        Remove the double notes (same pitch and onset in different voices): they cannot be played twice,
        which is a problem, e.g., on a mechanical piano. Only one note of each group is performed.
    keep_longer_double_note : bool
        Keep the longest note of the group of double notes (the shortest otherwise).

    Returns
    -------
    ppart : partitura.performance.PerformedPart
        The notes are ordered by score onset and pitch (the grace notes at their expanded position).
    W, names, predictions, alignment : only if `return_features`
        `W` and `predictions` have one row per note of `part.note_array()` (without the removed double
        notes), `alignment` pairs the ids of the score notes with the ones of the performed notes.
    """
    W, names = compute_note_features(part, _feature_subset(config))

    # The rules and the codec use the note array of `score_`: with the grace notes expanded and without
    # the double notes (same row order as `part`; `W` has the same rows)
    note_array = part.note_array()
    if expand_grace_notes and has_grace_notes(part):
        note_array = expanded_note_array(part, grace_offset_quarter)
    keep = np.ones(len(note_array), dtype=bool)
    if remove_double_notes:
        keep = double_notes_keep_mask(part, keep_longer_double_note)
    W = W[keep]
    score_ = NoteArrayScore(note_array[keep], part)

    # Initialize rule pipelines
    dr = DynamicsRules(pipeline=config["dynamics"]["pipeline"], rng=rng)
    ter = TempoRules(pipeline=config["tempo"]["pipeline"], rng=rng)
    ar = ArticulationRules(pipeline=config["articulation"]["pipeline"], rng=rng)
    tir = TimingRules(pipeline=config["timing"]["pipeline"], rng=rng)

    # Compute expressive parameters
    velocity_trend, velocity_dev = dr.compute_parameters(part=score_, W=W, names=names)
    beat_period = ter.compute_parameters(part=score_, W=W, names=names)
    articulation_log = ar.compute_parameters(part=score_, W=W, names=names)
    timing = tir.compute_parameters(
        part=score_,
        W=W,
        names=names,
        velocity_trend=velocity_trend,
        velocity_dev=velocity_dev,
    )

    # Parameters of the `partitura` codec (velocity in [0, 1])
    predictions = np.zeros(len(W), dtype=[(pn, "f4") for pn in PARAMETER_NAMES])
    predictions["beat_period"] = beat_period
    predictions["velocity_trend"] = velocity_trend
    predictions["velocity_dev"] = velocity_dev
    predictions["velocity"] = velocity_trend - velocity_dev
    predictions["timing"] = timing
    predictions["articulation_log"] = articulation_log

    # `decode_performance` sorts the notes by onset and pitch: give it the notes (and the parameters) in
    # that order, so that the ids of the performed notes are the right ones
    note_array = score_.note_array()
    sort_idx = np.lexsort((note_array["pitch"], note_array["onset_div"]))
    ppart, alignment = decode_performance(
        NoteArrayScore(note_array[sort_idx]), predictions[sort_idx], return_alignment=True
    )

    if return_features:
        return ppart, W, names, predictions, alignment
    return ppart
