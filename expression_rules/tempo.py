import numpy as np

from basismixer.utils import get_unique_onset_idxs, notewise_to_onsetwise, onsetwise_to_notewise

from base import PerformanceRules, _get_score_info, _get_constant_markings_array

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

DEFAULT_PIPELINE = ['tempo_markings_rules', 'final_ritardando_rules']

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

    (tempo_markings_map, next_tempo_markings_map,
     tm_array) = _get_constant_tempo_markings_map(W_onsetwise=W_onsetwise,
                                                  names=names,
                                                  unique_onsets=unique_onsets,
                                                  rng=rng)
    beat_period_per_tm = _sample_beat_period_per_dm(tm_array=tm_array,
                                                    rng=rng)

    beat_period = _initialize_beat_period(unique_onsets, tm_array, beat_period_per_tm)


    return beat_period

def _get_constant_tempo_markings_map(W_onsetwise, names, unique_onsets, rng=RNG):

    # tm_array = []
    # for on, w in zip(unique_onsets, W_onsetwise):

    #     for ti, tm in enumerate(CONSTANT_MARKINGS):
    #         if tm in names:
    #             if w[names.index(tm)] != 0:
    #                 try:
    #                     if tm_array[-1][1] != ti:
    #                         tm_array.append((on, ti))
    #                 except:
    #                     tm_array.append((on, ti))

    # if len(tm_array) == 0:
    #     tm_array.append((unique_onsets.min(), int(np.where(CONSTANT_MARKINGS == 'moderato')[0])))
    # tm_array = np.array(tm_array)
    tm_array = _get_constant_markings_array(W_onsetwise=W_onsetwise,
                                            names=names,
                                            unique_onsets=unique_onsets,
                                            constant_markings=CONSTANT_MARKINGS,
                                            default_marking='moderato')

    return tm_array


def _initialize_beat_period(W_onsetwise, names, unique_onsets):
    """
    Helper method to initialize beat_period with a constant value per
    constant tempo markings.
    """
    # beat_period = np.ones(len(unique_onsets)) * 60 / 108.0

    # for tm in TEMPO_MARKINGS_BPM:
    #     if tm in names:
    #         ti = names.index(tm)
    #         t_idx = np.where(W_onsetwise[:, ti] == 1)[0]
    #         # bp_low, bp_high = TEMPO_MARKINGS_BPM[tm]
    #         beat_period[t_idx] = 60. / rng.randint(*TEMPO_MARKINGS_BPM[tm])


    return beat_period




    # measure_array = []
    # for m in part.iter_all(score.Measure):
    #     measure_array.append((m.number,) + tuple(part.beat_map([m.start.t, m.end.t])))

    # measure_array = np.array(measure_array)
    # measure_map = interp1d(measure_array[:, 0], measure_array[:, 1],
    #                        kind='previous',
    #                        bounds_error=False,
    #                        fill_value=(measure_array[:, 1].min(),
    #                                    measure_array[:, 1].max()))
    # inv_measure_map = interp1d(measure_array[:, 1], measure_array[:, 0],
    #                        kind='previous',
    #                        bounds_error=False,
    #                        fill_value=(measure_array[:, 0].min(),
    #                                    measure_array[:, 0].max()))

    # n_bars_rit = 2
    # final_ritard_idxs = np.where(unique_onsets >= measure_array[-n_bars_rit, 1])[0]
    # rit_curve = friberg_sundberg_rit(len(final_ritard_idxs))
    # beat_period[final_ritard_idxs] *= rit_curve
    # return beat_period


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







