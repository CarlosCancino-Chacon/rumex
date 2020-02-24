import numpy as np

from basismixer.utils import get_unique_onset_idxs, notewise_to_onsetwise, onsetwise_to_notewise

RNG = np.random.RandomState(1984)

DEFAULT_PIPELINE = ['mechanical_lead_rules']

class TimingRules(object):
    def __init__(self, pipeline=DEFAULT_PIPELINE, rng=RNG):
        self.pipeline = []

        for proc in pipeline:
            try:
                func = globals()[proc]
                self.pipeline.append(func)
            except:
                raise ValueError('{0} is not an available method'.format(proc))
            
        self.rng = rng

    def compute_timing(self, part, W, names, *args, **kwargs):

        # get unique onsets of the piece
        onsets = part.note_array['onset']
        duration = part.note_array['duration']
        pitch = part.note_array['pitch']

        unique_onset_idxs, unique_onsets = get_unique_onset_idxs(onsets, return_unique_onsets=True)

        timing = np.zeros(len(W))

        velocity_trend = kwargs.pop('velocity_trend', None)
        velocity_dev = kwargs.pop('velocity_dev', None)
        beat_period = kwargs.pop('beat_period', None)
        articulation_log = kwargs.pop('articulation_log', None)

        for func in self.pipeline:

            timing = func(W=W,
                          names=names,
                          onsets=onsets,
                          pitch=pitch,
                          duration=duration,
                          rng=self.rng,
                          unique_onset_idxs=unique_onset_idxs,
                          unique_onsets=unique_onsets,
                          timing=timing,
                          velocity_trend=velocity_trend,
                          velocity_dev=velocity_dev,
                          beat_period=beat_period,
                          articulation_log=articulation_log,
                          *args, **kwargs)

        return timing


def mechanical_lead_rules(W, names, pitch, timing=None, velocity_trend=None, velocity_dev=None,
                          rng=RNG, *args, **kwargs):

    if velocity_trend is None:
        velocity_trend = np.ones(len(W)) * 50
    if velocity_dev is None:
        velocity_dev = np.zeros_like(velocity_trend)

    velocity = (velocity_trend - velocity_dev) * 127
    timing += mechanical_lead(pitch, velocity)

    return timing

    
def mechanical_lead(pitch, velocity, lead=0.05):
    """
    Compute mechanical lead
    """
    return (np.exp((pitch - 127) / 127.) *
            np.exp(-(velocity - 127) / 127) * lead)
