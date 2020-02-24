import numpy as np

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
    "prestissimo": (180, 210)}

from basismixer.utils import get_unique_onset_idxs, notewise_to_onsetwise, onsetwise_to_notewise

RNG = np.random.RandomState(1984)

DEFAULT_PIPELINE = ['tempo_markings_rules', 'final_ritardando_rules']

class TempoRules(object):
    def __init__(self, pipeline=DEFAULT_PIPELINE, rng=RNG):
        self.pipeline = []

        for proc in pipeline:
            try:
                func = globals()[proc]
                self.pipeline.append(func)
            except:
                raise ValueError('{0} is not an available method'.format(proc))
            
        self.rng = rng

    def compute_tempo(self, part, W, names, *args, **kwargs):

        # get unique onsets of the piece
        onsets = part.note_array['onset']
        duration = part.note_array['duration']
        pitch = part.note_array['pitch']

        unique_onset_idxs, unique_onsets = get_unique_onset_idxs(onsets, return_unique_onsets=True)

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


def tempo_markings_rules(W, names, onsets, *args, **kwargs):

    beat_period = np.ones(N_o) * 60 / 108.

    for tm in TEMPO_MARKINGS_BPM:
        if tm in names:
            ti = names.index(tm)
            t_idx = np.where(W_onsetwise[:, ti] == 1)[0]
            bp_low, bp_high = TEMPO_MARKINGS_BPM[tm]
            beat_period[t_idx] = 60. / rng.randint(bp_low, bp_high)


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





    
    
