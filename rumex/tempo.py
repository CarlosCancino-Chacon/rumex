import numpy as np
from scipy.interpolate import interp1d
from scipy.misc import derivative

import partitura
from basismixer.utils import (
    get_unique_onset_idxs,
    notewise_to_onsetwise,
    onsetwise_to_notewise,
)
import basismixer.utils.rendering as rendering

from rumex.base import (
    PerformanceRules,
    _get_score_info,
    _get_constant_markings_array_and_maps,
    _initialize_onsetwise_parameter,
    compute_increase_decrease_profile,
)


TEMPO_MARKINGS_BPM = {
    "larghissimo": (18, 20),
    "grave": (20, 40),
    "largo": (40, 60),
    "larghetto": (60, 66),
    "pesante": (60, 66),
    "adagio": (66, 76),
    "adagietto": (70, 80),
    "andante": (76, 108),
    "andantino": (86, 108),
    "moderato": (108, 120),
    "allegretto": (112, 124),
    "scherzando": (115, 130),
    "allegro": (120, 168),
    "animato": (120, 150),
    "mosso": (120, 150),
    "leggiero": (120, 145),
    "agitato": (120, 145),
    "vivace": (140, 168),
    "vivo": (140, 168),
    "vivacissimo": (168, 200),
    "presto": (168, 200),
    "prestissimo": (180, 210),
}

CONSTANT_MARKINGS = np.array(
    [
        "larghissimo",
        "grave",
        "largo",
        "larghetto",
        "pesante",
        "adagio",
        "adagietto",
        "andante",
        "andantino",
        "moderato",
        "allegretto",
        "scherzando",
        "animato",
        "leggiero",
        "allegro",
        "animato",
        "mosso",
        "agitato",
        "vivace",
        "vivo",
        "vivacissimo",
        "presto",
        "prestissimo",
    ]
)


RNG = np.random.RandomState(1984)

DEFAULT_PIPELINE = [
    "tempo_markings_rules",
    "ioi_section_rules",
    "measure_pattern_rules",
    "phrase_arch_rules",
    "fermata_rules",
    "final_ritardando_rules",
    "motor_noise_rules"
]


class TempoRules(PerformanceRules):
    def __init__(self, pipeline=DEFAULT_PIPELINE, rng=RNG):
        super().__init__(pipeline=pipeline, rng=rng, module_globals=globals())

    def compute_parameters(self, part, W, names, *args, **kwargs):

        (pitch, onsets, duration, unique_onsets, unique_onset_idxs) = _get_score_info(
            part
        )

        timing = np.zeros(len(W))

        velocity_trend = kwargs.pop("velocity_trend", None)
        velocity_dev = kwargs.pop("velocity_dev", None)
        timing = kwargs.pop("timing", None)
        articulation_log = kwargs.pop("articulation_log", None)
        beat_period = kwargs.pop("beat_period", None)

        for func in self.pipeline:

            beat_period = func(
                W=W,
                names=names,
                onsets=onsets,
                pitch=pitch,
                duration=duration,
                rng=self.rng,
                unique_onset_idxs=unique_onset_idxs,
                unique_onsets=unique_onsets,
                beat_period=beat_period,
                velocity_trend=velocity_trend,
                velocity_dev=velocity_dev,
                articulation_log=articulation_log,
                timing=timing,
                part=part,
                *args,
                **kwargs
            )

            # import matplotlib.pyplot as plt
            # plt.plot(onsets, beat_period)
            # plt.show()

        return beat_period


def tempo_markings_rules(
    W,
    names,
    onsets,
    rng=RNG,
    unique_onset_idxs=None,
    unique_onsets=None,
    beat_period=None,
    *args,
    **kwargs
):
    """
    Rules for constant and gradual tempo markings.

    Parameters
    ----------
    W : np.ndarray
    names : list
    onsets : np.array
    rng : np.random.RandomState
    unique_onset_idxs: list or None
    unique_onsets: np.array or None
    beat_period : np.array

    Returns
    -------
    beat_period : np.array
        Beat period for each note (all notes in the same score position
        have the same beat period)
    Notes
    -----
    This method overwrites beat_period if given
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

    W_onsetwise = notewise_to_onsetwise(W, unique_onset_idxs)
    (
        tm_array,
        tempo_markings_map,
        next_tempo_markings_map,
    ) = _get_constant_markings_array_and_maps(
        W_onsetwise=W_onsetwise,
        names=names,
        unique_onsets=unique_onsets,
        constant_markings=CONSTANT_MARKINGS,
        default_marking="moderato",
    )
    beat_period_per_tm = _sample_beat_period_per_dm(tm_array=tm_array, rng=rng)

    # Initialize beat period (i.e., overwrites any given input beat_period)
    beat_period = _initialize_onsetwise_parameter(
        unique_onsets=unique_onsets,
        cm_array=tm_array,
        parameter_per_marking=beat_period_per_tm,
    )

    if "tempo_incr" in names:
        accelerando = W_onsetwise[:, names.index("tempo_incr")]
    else:
        accelerando = np.zeros(len(unique_onsets))

    if "tempo_decr" in names:
        ritardando = W_onsetwise[:, names.index("tempo_decr")]
    else:
        ritardando = np.zeros(len(unique_onsets))

    acc_rit_profile = compute_increase_decrease_profile(
        increase=accelerando,
        decrease=ritardando,
        unique_onsets=unique_onsets,
        constant_markings_map=tempo_markings_map,
        cm_array=tm_array,
        parameter_per_marking=beat_period_per_tm,
        default_diff=0.05,
        max_multiplier=0.5,
    )  # think of a reasonable default

    beat_period -= acc_rit_profile

    return onsetwise_to_notewise(beat_period, unique_onset_idxs)


def _sample_beat_period_per_dm(tm_array, rng):
    beat_period_per_tm = [
        60.0 / rng.uniform(*TEMPO_MARKINGS_BPM[CONSTANT_MARKINGS[int(tm)]])
        for _, tm in tm_array
    ]
    return beat_period_per_tm


def final_ritardando_rules(
    W,
    names,
    onsets,
    part,
    num_measures=1,
    rng=RNG,
    unique_onset_idxs=None,
    unique_onsets=None,
    beat_period=None,
    *args,
    **kwargs
):

    if beat_period is None:
        beat_period = np.ones(len(W)) * 60 / 108

    if unique_onset_idxs is None or unique_onsets is None:
        unique_onset_idxs, unique_onsets = get_unique_onset_idxs(
            onsets, return_unique_onsets=True
        )

    beats, _ = part.time_signature_map(part.inv_beat_map(unique_onsets.max()))

    final_rit_idxs = np.where(unique_onsets >= onsets.max() - (num_measures * beats))[0]

    init_onset = unique_onsets[final_rit_idxs].min()

    final_rit = friberg_sundberg_rit(len(final_rit_idxs))

    for ix in final_rit_idxs:
        
        fx = (unique_onsets[ix] - init_onset) / (onsets.max() - init_onset)
        beat_period[unique_onset_idxs[ix]] *= final_rit(fx)
        
    return beat_period


def fermata_rules(
    W,
    names,
    onsets,
    part,
    unique_onset_idxs=None,
    unique_onsets=None,
    beat_period=None,
    rng=RNG,
    *args,
    **kwargs
):

    if beat_period is None:
        beat_period = np.ones(len(W)) * 60 / 108

    fermata_tempo_deviations = np.ones(len(W))
    if "fermata" in names:
        if unique_onset_idxs is None or unique_onsets is None:
            unique_onset_idxs, unique_onsets = get_unique_onset_idxs(
                onsets, return_unique_onsets=True
            )

        fermata = notewise_to_onsetwise(W, unique_onset_idxs)[:, names.index("fermata")]
        fermata_idxs = np.where(fermata != 0)[0]

        if len(fermata_idxs) > 0:
            _ftd = []
            for fix in fermata_idxs:
                beats, _ = part.time_signature_map(
                    part.inv_beat_map(unique_onsets[fix])
                )
                f_onset = unique_onsets[fix]
                f_start = f_onset - beats
                f_end = f_onset + 1e-2
                max_ritardando = 0.25 + 0.75 * rng.rand()
                _ftd += [
                    (f_start, 0, max_ritardando),
                    (f_onset, 1, max_ritardando),
                    (f_end, 0, max_ritardando),
                ]

            if _ftd[0][0] > unique_onsets.min():

                _ftd.insert(0, (unique_onsets.min(), 0, 1))

            if _ftd[-1][0] < unique_onsets.max():
                _ftd.append((unique_onsets.max(), 0, 1))

            _ftd = np.array(_ftd)

            ftd_func = interp1d(
                _ftd[:, 0],
                _ftd[:, 1],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            max_rit_func = interp1d(
                _ftd[:, 0],
                _ftd[:, 2],
                kind="previous",
                bounds_error=False,
                fill_value=(1, 1),
            )

            fermata_tempo_deviations = (ftd_func(onsets) ** 5) * max_rit_func(
                onsets
            ) + 1

    beat_period *= fermata_tempo_deviations

    return beat_period


def motor_noise_rules(
    beat_period,
    onsets,
    unique_onset_idxs=None,
    unique_onsets=None,
    rng=RNG,
    std=0.05,
    *args,
    **kwargs
):

    if unique_onset_idxs is None or unique_onsets is None:
        unique_onset_idxs, unique_onsets = get_unique_onset_idxs(
            onsets, return_unique_onsets=True
        )

    beat_period_onsetwise = notewise_to_onsetwise(
        beat_period.reshape(-1, 1), unique_onset_idxs
    ).reshape(
        -1,
    )

    beat_period_onsetwise += rng.uniform(
        low=-std * beat_period_onsetwise.mean(),
        high=std * beat_period_onsetwise.mean(),
        size=len(beat_period_onsetwise),
    )
    # beat_period += rng.normal(loc=0, scale=std * beat_period.mean(), size=len(beat_period))

    return onsetwise_to_notewise(beat_period_onsetwise, unique_onset_idxs)


def friberg_sundberg_rit(len_c, r_w=0.5, r_q=2.0):
    """
    Compute ritard_curve
    """
    if len_c > 0:
        ritard_curve = (1 + (r_w ** r_q - 1) * np.linspace(0, 1, 50)) ** (1.0 / r_q)

        ritard_curve_func = interp1d(
            x=np.linspace(0, 1, 50),
            y=1.0 / ritard_curve,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        return ritard_curve_func

    else:
        return lambda x: 1.0


class RhythmicPattern(object):
    def __init__(self, pattern, exponent=1):
        self.pattern = pattern
        self.exponent = exponent
        self.func = interp1d(
            self.pattern[:, 0],
            self.pattern[:, 1],
            kind="quadratic",
            bounds_error=False,
            fill_value=(0, 0),
        )

    def __call__(self, onsets):
        return self.func(onsets) ** self.exponent


_TS_TRIPLE_PATTERN = np.array([[0, 1], [0.5, 0], [1, 1]])
TRIPLE_PATTERN = RhythmicPattern(_TS_TRIPLE_PATTERN)

_TS_DUPLE_PATTERN = np.array([[0, 1], [0.25, 0], [0.5, 0.5], [0.75, 0], [1, 1]])
DUPLE_PATTERN = RhythmicPattern(_TS_DUPLE_PATTERN)


def measure_pattern_rules(
    W,
    names,
    part,
    onsets,
    beat_period=None,
    unique_onset_idxs=None,
    unique_onsets=None,
    max_dev=1.15,
    min_dev=1,
    *args,
    **kwargs
):

    if beat_period is None:
        beat_period = np.ones(len(W)) * 60 / 108

    if unique_onset_idxs is None or unique_onsets is None:
        unique_onset_idxs, unique_onsets = get_unique_onset_idxs(
            onsets, return_unique_onsets=True
        )

    beat_phase = notewise_to_onsetwise(W, unique_onset_idxs)[
        :, names.index("beat_phase")
    ]

    time_signatures = np.array(
        [part.time_signature_map(o) for o in part.inv_beat_map(unique_onsets)]
    )

    unique_time_signatures = np.unique(time_signatures, axis=0)
    unique_ts_idxs = [
        np.where(
            np.logical_and(
                time_signatures[:, 0] == uts[0], time_signatures[:, 1] == uts[1]
            )
        )[0]
        for uts in unique_time_signatures
    ]

    pattern_deviation = np.ones(len(unique_onsets))
    for uts, uix in zip(unique_time_signatures, unique_ts_idxs):

        if uts[0] in (3, 9):
            pf = TRIPLE_PATTERN
        elif uts[0] in (2, 4, 6, 8, 12):
            pf = DUPLE_PATTERN
        else:
            # Default for now...
            pf = TRIPLE_PATTERN

        pattern_deviation[uix] = (max_dev - min_dev) * pf(beat_phase[uix]) + min_dev

    pattern_deviation = onsetwise_to_notewise(pattern_deviation, unique_onset_idxs)

    return beat_period * pattern_deviation


def phrase_arch_rules(
    W,
    names,
    part,
    onsets,
    phrases=None,
    beat_period=None,
    unique_onset_idxs=None,
    unique_onsets=None,
    scale=0.15,
    *args,
    **kwargs
):

    if beat_period is None:
        beat_period = np.ones(len(W)) * 60 / 108

    if unique_onset_idxs is None or unique_onsets is None:
        unique_onset_idxs, unique_onsets = get_unique_onset_idxs(
            onsets, return_unique_onsets=True
        )

    x = np.linspace(-1, 1, 50)

    y = rendering.minmax(x ** 2, lowest=1 - scale, highest=1)

    paf = interp1d(np.linspace(0, 1, 50), y,
                   kind="linear",
                   bounds_error=False, fill_value=(1, 1))

    if phrases is None:
        # a "phrase" per bar
        phrases = []
        for m in part.iter_all(partitura.score.Measure):
            phrases.append(part.beat_map([m.start.t, m.end.t]))

        phrases = np.array(phrases)

    for pstart, pend in phrases:
        valid_onset_idxs = np.where(
            np.logical_and(unique_onsets >= pstart, unique_onsets <= pend)
        )[0]
        valid_onsets = unique_onsets[valid_onset_idxs]

        if len(valid_onsets) > 0:
            px = (valid_onsets - pstart) / (pend - pstart)
            beat_period_dev = paf(px)
            for vix, bpd in zip(valid_onset_idxs, beat_period_dev):
                # print('before', beat_period[unique_onset_idxs[vix]])
                beat_period[unique_onset_idxs[vix]] *= bpd
                # print('after', beat_period[unique_onset_idxs[vix]], bpd)
    return beat_period


def ioi_section_rules(
    W,
    names,
    onsets,
    beat_period=None,
    unique_onset_idxs=None,
    unique_onsets=None,
    scale=5,
    *args,
    **kwargs
):

    if beat_period is None:
        beat_period = np.ones(len(W)) * 60 / 108

    if unique_onset_idxs is None or unique_onsets is None:
        unique_onset_idxs, unique_onsets = get_unique_onset_idxs(
            onsets, return_unique_onsets=True
        )

    beat_period_onsetwise = notewise_to_onsetwise(
        beat_period.reshape(-1, 1), unique_onset_idxs
    ).reshape(
        -1,
    )

    s_ioi = np.diff(unique_onsets)

    small_ioi_idxs = np.where(s_ioi < 0.25)[0]

    small_iois = s_ioi[small_ioi_idxs] / 0.25
    # p_ioi = beat_period_onsetwise[:-1] * s_ioi

    tempo_scale = np.ones_like(beat_period_onsetwise)

    tempo_scale[small_ioi_idxs] = 1 + np.exp(-scale * small_iois)

    tempo_scale = onsetwise_to_notewise(
        tempo_scale.reshape(-1, 1), unique_onset_idxs
    ).reshape(
        -1,
    )

    beat_period *= tempo_scale

    return beat_period
