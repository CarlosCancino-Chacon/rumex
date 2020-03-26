import numpy as np
from scipy.interpolate import interp1d

from basismixer.utils import get_unique_onset_idxs, notewise_to_onsetwise, onsetwise_to_notewise

from expression_rules.base import (PerformanceRules,
                                   _get_score_info,
                                   _get_constant_markings_array_and_maps,
                                   _initialize_onsetwise_parameter,
                                   compute_increase_decrease_profile)


TEMPO_MARKINGS_BPM = {
    "larghissimo" : (18, 20),
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
    "leggiero": (120, 145),
    "agitato": (120, 145),
    "vivace": (140, 168),
    "vivo": (140, 168),
    "vivacissimo": (168, 200),
    "presto": (168, 200),
    "prestissimo": (180, 210)
}

CONSTANT_MARKINGS = np.array(["larghissimo",
                              "grave",
                              "largo",
                              "larghetto",
                              "adagio",
                              "adagietto",
                              "andante",
                              "andantino",
                              "moderato",
                              "allegretto",
                              "scherzando",
                              "animato",
                              "leggiero"
                              "allegro"
                              "agitato",
                              "vivace",
                              "vivo",
                              "vivacissimo",
                              "presto",
                              "prestissimo",
])


RNG = np.random.RandomState(1984)

DEFAULT_PIPELINE = ['tempo_markings_rules',
                    'fermata_rules',
                    'motor_noise_rules'
                    # 'final_ritardando_rules',
]

class TempoRules(PerformanceRules):
    def __init__(self, pipeline=DEFAULT_PIPELINE, rng=RNG):
        super().__init__(pipeline=pipeline,
                         rng=rng,
                         module_globals=globals())

    def compute_parameters(self, part, W, names, *args, **kwargs):

        (pitch, onsets, duration, unique_onsets, unique_onset_idxs) = _get_score_info(part)

        timing = np.zeros(len(W))

        velocity_trend = kwargs.pop('velocity_trend', None)
        velocity_dev = kwargs.pop('velocity_dev', None)
        timing = kwargs.pop('timing', None)
        articulation_log = kwargs.pop('articulation_log', None)
        beat_period = kwargs.pop('beat_period', None)

        for func in self.pipeline:

            beat_period = func(W=W,
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
                               *args, **kwargs)
        return beat_period


def tempo_markings_rules(W, names, onsets, rng=RNG,
                         unique_onset_idxs=None,
                         unique_onsets=None,
                         beat_period=None,
                         *args, **kwargs):
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
        unique_onset_idxs, unique_onsets = get_unique_onset_idxs(onsets, return_unique_onsets=True)
    else:
        unique_onsets = kwargs.get('unique_onsets', None)

        if unique_onsets is None:
            unique_onsets = np.array([onsets[uix].mean() for uix in unique_onset_idxs])
        else:
            assert len(unique_onsets) == len(unique_onset_idxs)


    W_onsetwise = notewise_to_onsetwise(W, unique_onset_idxs)

    (tm_array, tempo_markings_map, next_tempo_markings_map) = _get_constant_markings_array_and_maps(
        W_onsetwise=W_onsetwise,
        names=names,
        unique_onsets=unique_onsets,
        constant_markings=CONSTANT_MARKINGS,
        default_marking='moderato')
    beat_period_per_tm = _sample_beat_period_per_dm(tm_array=tm_array,
                                                    rng=rng)
    
    # Initialize beat period (i.e., overwrites any given input beat_period)
    beat_period = _initialize_onsetwise_parameter(unique_onsets=unique_onsets,
                                                  cm_array=tm_array,
                                                  parameter_per_marking=beat_period_per_tm)

    if 'tempo_incr' in names:
        accelerando = W_onsetwise[:, names.index('tempo_incr')]
    else:
        accelerando = np.zeros(len(unique_onsets))

    if 'tempo_decr' in names:
        ritardando = W_onsetwise[:, names.index('tempo_decr')]
    else:
        ritardando = np.zeros(len(unique_onsets))

    
    acc_rit_profile = compute_increase_decrease_profile(increase=accelerando,
                                                        decrease=ritardando,
                                                        unique_onsets=unique_onsets,
                                                        constant_markings_map=tempo_markings_map,
                                                        cm_array=tm_array,
                                                        parameter_per_marking=beat_period_per_tm,
                                                        default_diff=0.05,
                                                        max_multiplier=0.5) # think of a reasonable default
                                                        
    beat_period -= acc_rit_profile

    return onsetwise_to_notewise(beat_period, unique_onset_idxs)

def _sample_beat_period_per_dm(tm_array, rng):
    beat_period_per_tm = [60. / rng.uniform(*TEMPO_MARKINGS_BPM[CONSTANT_MARKINGS[int(tm)]])
                          for _, tm in tm_array]
    return beat_period_per_tm

def final_ritardando_rules(W, names, onsets, rng=RNG,
                           unique_onset_idxs=None,
                           unique_onsets=None,
                           beat_period=None,
                           *args, **kwargs):

    
    return

def fermata_rules(W, names, onsets, part,
                  unique_onset_idxs=None,
                  unique_onsets=None,
                  beat_period=None,
                  rng=RNG,
                  *args, **kwargs):

    if beat_period is None:
        beat_period = np.ones(len(W)) * 60 / 108
    
    fermata_tempo_deviations = np.ones(len(W))
    if 'fermata' in names:
        if unique_onset_idxs is None or unique_onsets is None:
            unique_onset_idxs, unique_onsets = get_unique_onset_idxs(onsets, return_unique_onsets=True)            
    
        fermata = notewise_to_onsetwise(W, unique_onset_idxs)[:, names.index('fermata')]

        fermata_idxs = np.where(fermata != 0)[0]

        _ftd = []
        for fix in fermata_idxs:
            beats, _ = part.time_signature_map(unique_onsets[fix])
            f_onset = unique_onsets[fix]
            f_start = f_onset - beats
            f_end = f_onset + 1e-2
            max_ritardando = 0.25 + 0.75 * rng.rand()
            _ftd += [(f_start, 0, max_ritardando),
                     (f_onset, 1, max_ritardando),
                     (f_end, 0, max_ritardando)]

        if _ftd[0][0] > unique_onsets.min():

            _ftd.insert(0, (unique_onsets.min(), 0, 1))

        if _ftd[-1][0] < unique_onsets.max():
            _ftd.append((unique_onsets.max(), 0, 1))

        _ftd = np.array(_ftd)

        ftd_func = interp1d(_ftd[:, 0], _ftd[:, 1], kind='linear',
                            bounds_error=False, fill_value='extrapolate')
        max_rit_func = interp1d(_ftd[:, 0], _ftd[:, 2], kind='previous',
                                bounds_error=False, fill_value=(1, 1))

        fermata_tempo_deviations = (ftd_func(onsets) ** 3) * max_rit_func(onsets) + 1

    beat_period *= fermata_tempo_deviations

    return beat_period


def motor_noise_rules(beat_period, rng=RNG, std=0.05, *args, **kwargs):

    beat_period += rng.normal(loc=0, scale=std * beat_period.mean(), size=len(beat_period))

    return beat_period


def friberg_sundberg_rit(len_c, r_w=0.5, r_q=2.0):
    """
    Compute ritard_curve
    """
    if len_c > 0:
        ritard_curve = (1 + (r_w ** r_q - 1) *
                        np.linspace(0, 1, len_c)) ** (1. / r_q)
        return 1.0 / ritard_curve

    else:
        return 1.0
