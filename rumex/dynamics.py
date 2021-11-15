import numpy as np
from scipy.interpolate import interp1d

from basismixer.utils import (
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
        Value from the crescendo basis function for each score onset (i.e., `unique_onsets`)
        The basis function evaluates to 0 at the start of the crescendo mark and 1 at the end and
        zero otherwise. See `basismixer.basisfunction.loudness_direction_basis`.
    diminuendo : np.ndarray
        Value from the diminuendo basis function for each score onset (i.e., `unique_onsets`)
        The basis function evaluates to 0 at the start of the diminuendo mark and 1 at the end and
        zero otherwise. See `basismixer.basisfunction.loudness_direction_basis`.
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
