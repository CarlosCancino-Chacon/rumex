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

"""
Grace notes.

A grace note has no duration in the score. To perform it, it needs a position in time, which depends on
the type of ornament. `partitura.utils.music.expand_grace_notes_from_local_grace_order` places them in
score time:

* **appoggiatura**: expands *forwards*, it starts on the beat and takes time from the main note;
* **acciaccatura** (and other grace notes): expands *backwards*, it starts before the beat and takes time
  from the previous note of the same staff;
* the grace notes of the same type at an onset share a total duration `grace_offset_quarter`
  (in quarter notes), in the order of the score.

Here the expanded note array is put in the row order of `part.note_array()`, so that it can be used
together with the note features and the expressive parameters of the rules, which have one row per note
of the part.
"""
import contextlib
import io
import warnings

import numpy as np

try:
    from partitura.utils.music import expand_grace_notes_from_local_grace_order
except ImportError:  # pragma: no cover
    expand_grace_notes_from_local_grace_order = None


def has_grace_notes(part):
    """True if the part has grace notes."""
    return bool(len(list(part.iter_all(_grace_note_class()))))


def _grace_note_class():
    from partitura.score import GraceNote

    return GraceNote


def expanded_note_array(part, grace_offset_quarter=0.25):
    """
    Note array of a part with the grace notes expanded in score time.

    Parameters
    ----------
    part : partitura.score.Part
    grace_offset_quarter : float
        Total duration (in quarter notes) of the grace notes of the same type at an onset. The previous
        note (acciaccatura) or the main note (appoggiatura) is shortened by this amount.

    Returns
    -------
    note_array : np.ndarray
        Structured note array in the same order as `part.note_array()` (the notes of the part are
        identified by their `id`).
    """
    if expand_grace_notes_from_local_grace_order is None:
        raise ImportError(
            "Expanding grace notes needs a version of partitura with "
            "`partitura.utils.music.expand_grace_notes_from_local_grace_order` (partitura >= 1.9.0)"
        )
    # partitura prints a message when it has to shorten the grace notes (the main note is too short):
    # turn it into a warning (shown once)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        expanded = expand_grace_notes_from_local_grace_order(
            part, grace_offset_quarter=grace_offset_quarter
        )
    for line in out.getvalue().splitlines():
        if line.strip():
            warnings.warn(line.strip().replace("Warning: ", ""), stacklevel=2)
    ids = part.note_array()["id"]
    row_of_id = {nid: i for i, nid in enumerate(expanded["id"])}
    return expanded[[row_of_id[nid] for nid in ids]]


class NoteArrayScore(object):
    """
    Score-like object that gives a fixed note array (what the rules and `decode_performance` need).

    Everything else (e.g., `beat_map`, `iter_all`, ...) is the one of `part` (if given), so the rules can
    use it in place of the part.
    """

    def __init__(self, note_array, part=None):
        self._note_array = note_array
        self.part = part

    def note_array(self, *args, **kwargs):
        return self._note_array

    def __getattr__(self, name):
        if name == "part" or self.part is None:
            raise AttributeError(name)
        return getattr(self.part, name)


class ExpandedScore(NoteArrayScore):
    """A part whose `note_array()` has the grace notes expanded."""

    def __init__(self, part, grace_offset_quarter=0.25):
        super().__init__(expanded_note_array(part, grace_offset_quarter), part)
