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

import os

import numpy as np
import partitura

from rumex import compute_note_features, load_score, render_piece

EXAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data_examples",
    "mozart_k331.musicxml",
)


def test_note_features():
    part = load_score(EXAMPLE)
    W, names = compute_note_features(part)
    assert W.shape == (len(part.note_array()), len(names))
    for name in ("loudness_incr", "tempo_incr", "beat_phase", "fermata", "slur_incr"):
        assert name in names


def test_render_piece(tmp_path):
    part = load_score(EXAMPLE)
    ppart, W, names, predictions, alignment = render_piece(
        part, rng=np.random.RandomState(1), return_features=True
    )
    assert len(ppart.notes) == len(part.note_array()) == len(alignment)
    assert np.all(np.isfinite([n["note_on"] for n in ppart.notes]))
    assert all(1 <= n["velocity"] <= 127 for n in ppart.notes)
    # same seed, same performance
    ppart2 = render_piece(part, rng=np.random.RandomState(1))
    assert [n["note_on"] for n in ppart.notes] == [n["note_on"] for n in ppart2.notes]
    # the performance can be saved
    partitura.save_performance_midi(ppart, str(tmp_path / "out.mid"))
