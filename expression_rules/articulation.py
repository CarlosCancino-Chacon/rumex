import numpy as np

from basismixer.utils import get_unique_onset_idxs, notewise_to_onsetwise, onsetwise_to_notewise

from base import PerformanceRules, _get_score_info

ARTICULATION_MARKINGS_RATIO = {"staccato": (0.4, 0.6),
                               "tenuto": (0.7, 0.9),
                               "legato": (1.0, 1.3),
                               "detached-legato": (0.8, 1.1)
}

RNG = np.random.RandomState(1984)

DEFAULT_PIPELINE = ['notewise_articulation_markings_rules']

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

        return np.log2(articulation)



def notewise_articulation_markings_rules(W, names, articulation=None, rng=RNG, *args, **kwargs):

    
    if articulation is None:
        articulation = np.ones(len(W))

    
    for am in ARTICULATION_MARKINGS_RATIO:
        if am in names:
            art = W[:, names.index(am)]
            art_idxs = np.where(art != 0)[0]
            articulation[art_idxs] = rng.uniform(*ARTICULATION_MARKINGS_RATIO[am], len(art_idxs))


    return articulation
