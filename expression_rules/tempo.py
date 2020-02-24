import numpy as np

from basismixer.utils import get_unique_onset_idxs, notewise_to_onsetwise, onsetwise_to_notewise

from base import (PerformanceRules,
                  _get_score_info,
                  _get_constant_markings_array_and_maps,
                  _initialize_onsetwise_parameter,
                  compute_increase_decrease_profile)


TEMPO_MARKINGS_BPM = {
    "larghissimo" : (18, 20),
    "grave": (20, 40),
    "largo": (40, 60),
    "larghetto": (60, 66),
    "adagio": (66, 76),
    "adagietto": (70, 80),
    "andante": (76, 108),
    "andantino": (86, 108),
    "moderato": (108, 120),
    "allegretto": (112, 124),
    "allegro": (120, 168),
    "vivace": (140, 168),
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
                              "allegro"
                              "vivace",
                              "vivacissimo",
                              "presto",
                              "prestissimo",
])


RNG = np.random.RandomState(1984)

DEFAULT_PIPELINE = ['tempo_markings_rules',
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
                               *args, **kwargs)
        return beat_period


def tempo_markings_rules(W, names, onsets, rng=RNG,
                         unique_onset_idxs=None,
                         unique_onsets=None,
                         beat_period=None,
                         *args, **kwargs):
    """
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







