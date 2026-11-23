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
Note features for the rules, computed with `partitura.musicanalysis.note_features`.

The rules look up columns by name (e.g. `"loudness_incr"`, `"p"`, `"staccato"`). Partitura names its
columns `<feature function>.<name>`; here the prefix is dropped when the name is unique among the
selected features.
"""
import numpy as np
from partitura.musicanalysis import make_note_features

# note features used by the rules (names of `partitura.musicanalysis.note_features` functions)
DEFAULT_NOTE_FEATURES = [
    "loudness_direction_feature",  # p, mf, f, ..., loudness_incr, loudness_decr
    "tempo_direction_feature",  # adagio, ..., tempo_incr, tempo_decr
    "articulation_feature",  # staccato, tenuto, accent, ...
    "slur_feature",  # slur_incr, slur_decr
    "fermata_feature",  # fermata
    "metrical_strength_feature",  # beat_phase, metrical_strength_*
]


def short_names(names):
    """Drop the `<function>.` prefix of partitura feature names where the result is unique."""
    short = [n.split(".", 1)[-1] for n in names]
    return [s if short.count(s) == 1 else n for s, n in zip(short, names)]


def compute_note_features(part, note_features="all"):
    """
    Note features of a part, one row per note of `part.note_array()`.

    Parameters
    ----------
    part : partitura.score.Part
    note_features : "all" or list of str
        "all" (all the features of `DEFAULT_NOTE_FEATURES`), or the names of partitura feature
        functions (or any list accepted by `partitura.musicanalysis.make_note_features`).

    Returns
    -------
    W : np.ndarray
        Array (n_notes, n_features).
    names : list
        Names of the columns.
    """
    if note_features == "all":
        note_features = DEFAULT_NOTE_FEATURES
    W, names = make_note_features(part, note_features)
    return np.asarray(W, dtype=float), short_names(names)
