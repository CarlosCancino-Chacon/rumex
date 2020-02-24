from abc import ABC, abstractmethod
import numpy as np

from basismixer.utils import get_unique_onset_idxs

RNG = np.random.RandomState(1984)

class PerformanceRules(ABC):
    """
    Base class for implementing performance rules

    Parameters
    ----------
    pipeline : list
       A list with the names of the methods defining the rules for computing
       expressive parameters.
    module_globals : dict
       Globals of the module where the subclass is instantitated. (This is an ugly hack!)
    rng : np.random.RandomState (optional)
       A random state for sampling random numbers
    """
    def __init__(self, pipeline, module_globals, rng=RNG):
        self.pipeline = []

        for proc in pipeline:
            try:
                func = module_globals[proc]
                self.pipeline.append(func)
            except:
                raise ValueError('{0} is not an available method'.format(proc))
            
        self.rng = rng

    @abstractmethod
    def compute_parameters(self, *args, **kwargs):
        """
        Method for computing expressive parameters according
        to the rules specified in the pipeline. This method needs
        to be implemented in each subclass
        """
        pass

def _get_score_info(part):
    """Helper method for getting score information
    """
    onsets = part.note_array['onset']
    duration = part.note_array['duration']
    pitch = part.note_array['pitch']

    unique_onset_idxs, unique_onsets = get_unique_onset_idxs(onsets, return_unique_onsets=True)

    return pitch, onsets, duration, unique_onsets, unique_onset_idxs


def _get_constant_markings_array(W_onsetwise, names, unique_onsets, constant_markings, default_marking):
    """
    Get an array indicating the starting points of constant markings
    """
    # get starting point of all constant markings
    cm_array = []
    for on, w in zip(unique_onsets, W_onsetwise):

        for ci, dm in enumerate(constant_markings):
            if dm in names:
                di = names.index(dm)
                if w[di] != 0:
                    try:
                        if cm_array[-1][1] != ci:
                            cm_array.append((on, ci))
                    except:
                        cm_array.append((on, ci))

    # if there are no constant markings, assume `default_marking`
    if len(cm_array) == 0:
        cm_array.append((unique_onsets.min(), int(np.where(constant_markings == default_marking)[0])))
    cm_array = np.array(cm_array)

    # assume `default_marking` if there is no marking at the begining of the piece
    if cm_array[0, 0] != unique_onsets.min():
        cm_array = np.insert(cm_array, 0, (unique_onsets.min(),
                                           int(np.where(constant_markings == default_marking)[0])))

    return cm_array

