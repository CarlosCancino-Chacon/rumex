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


def load_score(filename, **kwargs):
    """
    Load a score (MusicXML, MIDI, MEI, ...) as a single part without grace notes.

    The part is the merge of all the parts of the score. Notes without an id get one (the rules need
    unique note ids).
    """
    if str(filename).lower().endswith((".mid", ".midi")):
        scr = partitura.load_score_midi(filename, **kwargs)
    else:
        scr = partitura.load_score(filename, force_note_ids="keep", **kwargs)
    parts = scr.parts
    part = parts[0] if len(parts) == 1 else score.merge_parts(parts)
    # TODO: expand grace notes (`score.expand_grace_notes`) instead of removing them
    score.remove_grace_notes(part)
    return part


def _feature_subset(config):
    names = []
    for par in config.values():
        subset = par.get("note_features", par.get("basis_functions", "all"))
        if subset == "all":
            return "all"
        names += list(subset)
    return list(dict.fromkeys(names))


def render_piece(part, config=DEFAULT_CONFIG, rng=RNG, return_features=False):
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

    Returns
    -------
    ppart : partitura.performance.PerformedPart
    W, names, predictions, alignment : only if `return_features`
    """
    W, names = compute_note_features(part, _feature_subset(config))

    # Initialize rule pipelines
    dr = DynamicsRules(pipeline=config["dynamics"]["pipeline"], rng=rng)
    ter = TempoRules(pipeline=config["tempo"]["pipeline"], rng=rng)
    ar = ArticulationRules(pipeline=config["articulation"]["pipeline"], rng=rng)
    tir = TimingRules(pipeline=config["timing"]["pipeline"], rng=rng)

    # Compute expressive parameters
    velocity_trend, velocity_dev = dr.compute_parameters(part=part, W=W, names=names)
    beat_period = ter.compute_parameters(part=part, W=W, names=names)
    articulation_log = ar.compute_parameters(part=part, W=W, names=names)
    timing = tir.compute_parameters(
        part=part,
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

    ppart, alignment = decode_performance(part, predictions, return_alignment=True)

    if return_features:
        return ppart, W, names, predictions, alignment
    return ppart
