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

import numpy as np
from scipy.interpolate import interp1d

from partitura.musicanalysis.performance_codec import (
    get_unique_onset_idxs,
    notewise_to_onsetwise,
    onsetwise_to_notewise,
)

from rumex.base import (
    PerformanceRules,
    _get_score_info,
    _get_constant_markings_array_and_maps,
    _initialize_onsetwise_parameter,
    compute_increase_decrease_profile,
)

# Range values for each dynamics marking
DYNAMICS_MARKINGS_VELOCITY = {
    "ppppp": (5, 12),
    "pppp": (7, 17),
    "ppp": (10, 25),
    "pp": (15, 30),
    "p": (30, 40),
    "mp": (35, 45),
    "mf": (40, 55),
    "f": (65, 75),
    "ff": (85, 100),
    "fff": (90, 105),
    "ffff": (95, 110),
    # "ffff": (100, 110),
    "fffff": (102, 112),
    "ffffff": (110, 120),
}

ACCENT_INCREASE = {
    "accent": (0.1, 0.2),
    "fz": (0.15, 0.2),
    "sf": (0.15, 0.2),
    "sfz": (0.15, 0.2),
    # impulsive markings not recognized by partitura (e.g. "fz") are `unknown_impulsive`
    "unknown_impulsive": (0.15, 0.2),
    "staccato": (0.02, 0.05),
    "tenuto": (-0.02, -0.01),
}

# Constant dynamics markings
CONSTANT_MARKINGS = np.array(
    [
        "ppppp",
        "pppp",
        "ppp",
        "pp",
        "p",
        "mp",
        "mf",
        "f",
        "ff",
        "fff",
        "ffff",
        "fffff"
    ]
)

RNG = np.random.RandomState(1984)

DEFAULT_PIPELINE = [
    "dynamics_markings_rules",
    "simple_accent_rules",
    "polynomial_pitch_height_rules",
    "motor_noise_rules",
]


class DynamicsRules(PerformanceRules):
    """
    Pipeline for computing dynamics-related expressive parameters
    (velocity_trend and velocity_dev) using rules
    """

    def __init__(self, pipeline=DEFAULT_PIPELINE, rng=RNG):
        super().__init__(pipeline=pipeline, rng=rng, module_globals=globals())

    def compute_parameters(self, part, W, names, *args, **kwargs):

        (pitch, onsets, duration, unique_onsets, unique_onset_idxs) = _get_score_info(
            part
        )

        velocity_trend = np.ones(len(W)) * 64
        velocity_dev = np.zeros(len(W))

        for func in self.pipeline:

            velocity_trend, velocity_dev = func(
                W=W,
                names=names,
                onsets=onsets,
                pitch=pitch,
                duration=duration,
                rng=self.rng,
                unique_onset_idxs=unique_onset_idxs,
                unique_onsets=unique_onsets,
                velocity_trend=velocity_trend,
                velocity_dev=velocity_dev,
                *args,
                **kwargs
            )

        return velocity_trend / 127, velocity_dev / 127


def dynamics_markings_rules(
    W,
    names,
    onsets,
    rng=RNG,
    unique_onset_idxs=None,
    velocity_trend=None,
    velocity_dev=None,
    *args,
    **kwargs
):
    """
    Compute
    """

    if unique_onset_idxs is None:
        unique_onset_idxs, unique_onsets = get_unique_onset_idxs(
            onsets, return_unique_onsets=True
        )
    else:
        unique_onsets = kwargs.get("unique_onsets", None)

        if unique_onsets is None:
            unique_onsets = np.array([onsets[uix].mean() for uix in unique_onset_idxs])
        else:
            assert len(unique_onsets) == len(unique_onset_idxs)

    # Get onsetwise features
    W_onsetwise = notewise_to_onsetwise(W, unique_onset_idxs)

    # Compute dynamics markings map
    (
        dm_array,
        dynamics_markings_map,
        next_dynamics_markings_map,
    ) = _get_constant_markings_array_and_maps(
        W_onsetwise=W_onsetwise,
        names=names,
        unique_onsets=unique_onsets,
        constant_markings=CONSTANT_MARKINGS,
        default_marking="mf",
    )
    # sample velocity per dynamics marking
    velocity_per_dm = _sample_velocity_per_dm(dm_array=dm_array, rng=rng)

    # Initialize velocity trend
    velocity_trend = _initialize_onsetwise_parameter(
        unique_onsets=unique_onsets,
        cm_array=dm_array,
        parameter_per_marking=velocity_per_dm,
    )

    # Compute crescendo profile
    if "loudness_incr" in names:
        crescendo = W_onsetwise[:, names.index("loudness_incr")]
    else:
        crescendo = np.zeros(len(unique_onsets))

    if "loudness_decr" in names:
        diminuendo = W_onsetwise[:, names.index("loudness_decr")]
    else:
        diminuendo = np.zeros(len(unique_onsets))

    cresc_dim_profile = compute_crescendo_profile(
        crescendo=crescendo,
        diminuendo=diminuendo,
        unique_onsets=unique_onsets,
        dynamics_markings_map=dynamics_markings_map,
        dm_array=dm_array,
        velocity_per_dm=velocity_per_dm,
    )

    velocity_trend += cresc_dim_profile
    velocity_trend = onsetwise_to_notewise(velocity_trend, unique_onset_idxs)

    if velocity_dev is None:
        velocity_dev = np.zeros_like(velocity_trend)

    return velocity_trend, velocity_dev


def simple_accent_rules(
    W, names, velocity_trend=None, velocity_dev=None, rng=RNG, *args, **kwargs
):

    if velocity_trend is None:
        velocity_trend = np.ones(len(W)) * 50
    if velocity_dev is None:
        velocity_dev = np.zeros_like(velocity_trend)

    velocity_dev += compute_accents(
        W=W, names=names, velocity_trend=velocity_trend, rng=rng
    )

    return velocity_trend, velocity_dev


def polynomial_pitch_height_rules(
    pitch, velocity_trend=None, velocity_dev=None, *args, **kwargs
):

    if velocity_trend is None:
        velocity_trend = np.ones(len(pitch)) * 50
    if velocity_dev is None:
        velocity_dev = np.zeros_like(velocity_trend)

    velocity_dev += (
        1
        - centered_quadratic_pitch(pitch, max_pitch=pitch.max(), min_pitch=pitch.min())
    ) * velocity_trend

    return velocity_trend, velocity_dev


def skyline(pitch, unique_onset_idxs):
    """
    Skyline algorithm (Uitdenbogerd & Zobel, 1999): at each onset, the highest note is the melody.

    Parameters
    ----------
    pitch : np.ndarray
        MIDI pitch of each note.
    unique_onset_idxs : list of np.ndarray
        Indices of the notes that start at each onset (see `get_unique_onset_idxs`).

    Returns
    -------
    is_melody : np.ndarray
        Boolean array, True for the notes of the main melody line.
    """
    is_melody = np.zeros(len(pitch), dtype=bool)
    for idx in unique_onset_idxs:
        is_melody[idx[np.argmax(pitch[idx])]] = True
    return is_melody


def monophonic_skyline(pitch, onsets, duration, eps=1e-3):
    """
    Skyline algorithm that returns a strictly monophonic melody.

    The basic skyline (`skyline`) takes the highest note of each onset, but the notes that it
    selects can overlap (e.g., a held melody note and the highest note of the next onset). Here the
    notes are processed in order of onset: take the highest note that starts at the first onset;
    while it sounds, no other note can be selected; at the first onset after it ended, take the
    highest note that starts there, and so on. The selected notes never overlap.

    Parameters
    ----------
    pitch, onsets, duration : np.ndarray
        MIDI pitch, onset and duration (in the same time unit, e.g., beats) of each note.
    eps : float
        Tolerance for the end of a note (a note that starts less than `eps` before the end of the
        melody note is allowed to be selected; avoids problems with rounding).

    Returns
    -------
    is_melody : np.ndarray
        Boolean array, True for the notes of the main melody line.
    """
    pitch = np.asarray(pitch)
    onsets = np.asarray(onsets, dtype=float)
    ends = onsets + np.asarray(duration, dtype=float)

    is_melody = np.zeros(len(pitch), dtype=bool)
    unique_onset_idxs, unique_onsets = get_unique_onset_idxs(onsets, return_unique_onsets=True)
    free_at = -np.inf  # time when the last melody note ends
    for idx, onset in zip(unique_onset_idxs, unique_onsets):
        if onset >= free_at - eps:
            top = idx[np.argmax(pitch[idx])]
            is_melody[top] = True
            free_at = ends[top]

    return is_melody


def melody_loudness_rules(
    pitch,
    onsets,
    duration,
    unique_onset_idxs,
    velocity_trend,
    velocity_dev,
    melody_relative_loudness=0.0,
    monophonic=True,
    *args,
    **kwargs
):
    """
    Play the main melody louder (or softer) than the accompaniment.

    The melody is found with the skyline algorithm. Since `velocity = velocity_trend - velocity_dev`,
    the rule subtracts `melody_relative_loudness * velocity_trend` from `velocity_dev` for the
    melody notes.

    Parameters
    ----------
    melody_relative_loudness : float
        0 keeps the loudness, a positive value makes the melody louder by that proportion (0.5 is
        50% louder, 1 is twice as loud) and a negative value makes it softer (-0.5 is half
        way to silence). The final MIDI velocity is clipped to [1, 127] by the codec.
    monophonic : bool
        Use `monophonic_skyline` (takes the overlap of the notes into account) instead of the
        basic `skyline`.
    """
    if monophonic:
        is_melody = monophonic_skyline(pitch, onsets, duration)
    else:
        is_melody = skyline(pitch, unique_onset_idxs)
    velocity_dev[is_melody] -= melody_relative_loudness * velocity_trend[is_melody]

    return velocity_trend, velocity_dev


def motor_noise_rules(velocity_trend, velocity_dev, rng=RNG, std=0.05, *args, **kwargs):

    velocity_dev += (
        rng.normal(loc=0, scale=std, size=len(velocity_dev)) * velocity_trend
    )

    return velocity_trend, velocity_dev


def _sample_velocity_per_dm(dm_array, rng=RNG):
    """Sample a MIDI velocity for each dynamics marking"""
    velocity_per_dm = [
        rng.randint(*DYNAMICS_MARKINGS_VELOCITY[CONSTANT_MARKINGS[int(dm)]])
        for _, dm in dm_array
    ]

    return velocity_per_dm


def compute_crescendo_profile(
    crescendo,
    diminuendo,
    unique_onsets,
    dynamics_markings_map,
    dm_array,
    velocity_per_dm,
    default_dyn_diff=10,
):
    """
    Profile of crescendo and diminuendo

    Parameters
    ----------
    crescendo : np.ndarray
        Value from the crescendo note feature for each score onset (i.e., `unique_onsets`)
        The note feature evaluates to 0 at the start of the crescendo mark and 1 at the end and
        zero otherwise. See `partitura.musicanalysis.note_features.loudness_direction_feature`.
    diminuendo : np.ndarray
        Value from the diminuendo note feature for each score onset (i.e., `unique_onsets`)
        The note feature evaluates to 0 at the start of the diminuendo mark and 1 at the end and
        zero otherwise. See `partitura.musicanalysis.note_features.loudness_direction_feature`.
    unique_onsets : np.ndarray
        Unique score positions in beats in the score
    dynamics_markings_map : callable
        A method which takes as inputs the score position in beats and returns the index of the dynamics
        marking
    dm_array : np.ndarray
        An array indicating the starting times and dynamics markings (as indices of CONSTANT_MARKINGS)
        in the score.
    velocity_per_dm : iterable
        A list with the value of the constant MIDI velocity for each instance of a dynamics marking
        in the score.
    default_dyn_diff : int or float
       Default MIDI velocity for crescendo or diminuendo (if there is only a single dynamics marking)

    Returns
    -------
    cresc_dim_profile : np.array
        An array with the change in MIDI velocity for each score onset resulting from
        crescendo and diminuendo markings.
    """
    return compute_increase_decrease_profile(
        increase=crescendo,
        decrease=diminuendo,
        unique_onsets=unique_onsets,
        constant_markings_map=dynamics_markings_map,
        cm_array=dm_array,
        parameter_per_marking=velocity_per_dm,
        default_diff=default_dyn_diff,
    )


def compute_accents(W, names, velocity_trend, rng=RNG):

    velocity_dev = np.zeros(len(W))
    for i, (w, vt) in enumerate(zip(W, velocity_trend)):

        for accent_type in ACCENT_INCREASE:
            if accent_type in names:
                if w[names.index(accent_type)] != 0:
                    # import pdb
                    # pdb.set_trace()
                    velocity_dev[i] -= vt * rng.uniform(*ACCENT_INCREASE[accent_type])

    return velocity_dev


def centered_quadratic_pitch(
    pitch, min_proportion=0.8, max_pitch=96, min_pitch=24, max_proportion=1.2
):

    b = (
        np.sqrt(max_proportion - min_proportion)
        * (1 + min_pitch / max_pitch)
        / (1 - min_pitch / max_pitch)
    )
    a = (np.sqrt(max_proportion - min_proportion) + b) / max_pitch

    return np.clip((a * pitch - b) ** 2 + min_proportion, min_proportion, 1.0)
