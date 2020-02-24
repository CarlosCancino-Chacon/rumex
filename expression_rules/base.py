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

