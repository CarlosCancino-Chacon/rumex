import numpy as np

from basismixer.utils import get_unique_onset_idxs, notewise_to_onsetwise, onsetwise_to_notewise

from rumex.base import PerformanceRules, _get_score_info

ARTICULATION_MARKINGS_RATIO = {"staccato": (0.4, 0.6),
                               "tenuto": (0.7, 0.9),
                               "legato": (1.1, 1.5),
                               "detached-legato": (0.8, 1.1)
}

RNG = np.random.RandomState(1984)

DEFAULT_PIPELINE = ['slur_markings_rules',
                    'notewise_articulation_markings_rules',
                    'motor_noise_rules']

class ArticulationRules(PerformanceRules):
    def __init__(self, pipeline=DEFAULT_PIPELINE, rng=RNG):
        super().__init__(pipeline=pipeline,
                         rng=rng,
                         module_globals=globals())

    def compute_parameters(self, part, W, names, *args, **kwargs):

        (pitch, onsets, duration, unique_onsets, unique_onset_idxs) = _get_score_info(part)

        articulation = np.ones(len(W))

        for func in self.pipeline:

            articulation = func(W=W,
                                names=names,
                                onsets=onsets,
                                pitch=pitch,
                                duration=duration,
                                rng=self.rng,
                                unique_onset_idxs=unique_onset_idxs,
                                unique_onsets=unique_onsets,
                                articulation=articulation,
                                *args, **kwargs)
            # import matplotlib.pyplot as plt
            # plt.plot(onsets, articulation)
            # plt.show()

        return np.log2(articulation)



def notewise_articulation_markings_rules(W, names, articulation=None, rng=RNG, *args, **kwargs):

    
    if articulation is None:
        articulation = np.ones(len(W))

    
    for am in ARTICULATION_MARKINGS_RATIO:
        if am in names:
            art = W[:, names.index(am)]
            art_idxs = np.where(art != 0)[0]
            articulation[art_idxs] *= rng.uniform(*ARTICULATION_MARKINGS_RATIO[am], len(art_idxs))
            

    return articulation


def motor_noise_rules(W, articulation=None, rng=RNG, std=0.05, *args, **kwargs):

    if articulation is None:
        articulation = np.ones(len(W))

    articulation += rng.uniform(low=-std, high=std, size=len(articulation))

    return articulation


def slur_markings_rules(W, names, onsets,
                        articulation=None,
                        unique_onsets=None, unique_onset_idxs=None,
                        rng=RNG,
                        *args, **kwargs):

    if unique_onset_idxs is None or unique_onsets is None:
        unique_onset_idxs, unique_onsets = get_unique_onset_idxs(onsets,
                                                                 return_unique_onsets=True)

    if articulation is None:
        articulation = np.ones(len(W))

    if 'slur_incr' in names:
        slur = W[:, names.index('slur_incr')]
        # notewise_to_onsetwise(W, unique_onset_idxs)[:, names.index('slur_incr')]
        slur_idxs = np.where(slur != 0)[0]
        articulation[slur_idxs] *= rng.uniform(*ARTICULATION_MARKINGS_RATIO['legato'], len(slur_idxs))
        # articulation_onsetwise = notewise_to_onsetwise(articulation.reshape(-1, 1), unique_onset_idxs).reshape(-1, )
        # articulation_onsetwise[slur_idxs] *= rng.uniform(*ARTICULATION_MARKINGS_RATIO['legato'], len(slur_idxs))
        # import pdb
        # pdb.set_trace()
        # articulation = notewise_to_onsetwise(articulation_onsetwise.reshape(-1, 1), unique_onset_idxs)

    return articulation
