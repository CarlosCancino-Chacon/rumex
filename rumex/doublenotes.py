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
Double notes.

Composers sometimes write the same pitch at the same onset in two voices (e.g., to show that a note
belongs to the melody and to an inner voice). A performer cannot play it twice, and a performance with
both notes can be a problem for a mechanical piano (e.g., a Disklavier). This module finds the notes
to keep, using `partitura.utils.music.remove_double_notes_from_score_note_array`.
"""
import numpy as np

from partitura.utils.music import remove_double_notes_from_score_note_array


def double_notes_keep_mask(part, keep_longer=True):
    """
    Boolean mask of the notes of a part that are kept when the double notes are removed.

    A double note is a non-grace note with the same onset and pitch as another one. Of each group, only
    one note is kept.

    Parameters
    ----------
    part : partitura.score.Part
    keep_longer : bool
        Keep the longest note of the group (the shortest otherwise), so that the sustain of the longer
        note is not lost.

    Returns
    -------
    keep : np.ndarray
        Boolean array with one entry per note of `part.note_array()` (same order).
    """
    note_array = part.note_array(include_grace_notes=True)
    kept = remove_double_notes_from_score_note_array(
        note_array, choose_longer_note=keep_longer
    )
    return np.isin(note_array["id"], kept["id"])
