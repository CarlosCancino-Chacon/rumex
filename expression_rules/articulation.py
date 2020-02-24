import numpy as np

from basismixer.utils import get_unique_onset_idxs, notewise_to_onsetwise, onsetwise_to_notewise

ARTICULATION_MARKINGS_RATIO = {"staccato": (0.4, 0.6),
                               "tenuto": (0.7, 0.9),
                               "legato": (1.0, 1.3),
                               "detached-legato": (0.8, 1.1)
}

RNG = np.random.RandomState(1984)

DEFAULT_PIPELINE = ['notewise_articulation_markings_rules']

class ArticulationRules(object):
    def __init__(self, pipeline=DEFAULT_PIPELINE, rng=RNG):
        self.pipeline = []

        for proc in pipeline:
            try:
                func = globals()[proc]
                self.pipeline.append(func)
            except:
                raise ValueError('{0} is not an available method'.format(proc))
            
        self.rng = rng

    def compute_articulation(self, part, W, names, *args, **kwargs):

        # get unique onsets of the piece
        onsets = part.note_array['onset']
        duration = part.note_array['duration']
        pitch = part.note_array['pitch']

        unique_onset_idxs, unique_onsets = get_unique_onset_idxs(onsets, return_unique_onsets=True)

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
