import numpy as np
from scipy.interpolate import interp1d
import partitura

import partitura.score as score
import basismixer.basisfunctions as bf
import basismixer

from basismixer.utils import get_unique_onset_idxs, notewise_to_onsetwise, onsetwise_to_notewise

from basismixer.performance_codec import get_performance_codec

import matplotlib.pyplot as plt

from tempo import TEMPO_MARKINGS_BPM, friberg_sundberg_rit

from dynamics import DynamicsRules
from articulation import ArticulationRules
from timing import TimingRules
from tempo import TempoRules


if __name__ == '__main__':

    fn = '../data_examples/mozart_k331.musicxml'
    part = partitura.load_musicxml(fn, force_note_ids=True)
    score.expand_grace_notes(part)
    # part = score.unfold_part_maximal(part)

    # extract metrical features
    metrical, metrical_names = bf.metrical_strength_basis(part)
    # extract tempo/timing related features
    tempo, tempo_names = bf.tempo_direction_basis(part)
    # extract articulation related features
    articulation, articulation_names = bf.articulation_basis(part)
    # extract dynamics related features
    loudness, loudness_names = bf.loudness_direction_basis(part)

    # Concatenate features
    W_notewise = np.column_stack((metrical, tempo, articulation, loudness))
    names = metrical_names + tempo_names + articulation_names + loudness_names

    # get unique onsets of the piece
    onsets = part.note_array['onset']
    duration = part.note_array['duration']
    pitch = part.note_array['pitch']
    N_n = len(onsets)
    unique_onset_idxs, unique_onsets = get_unique_onset_idxs(onsets, return_unique_onsets=True)
    N_o = len(unique_onset_idxs)
    W_onsetwise = notewise_to_onsetwise(W_notewise, unique_onset_idxs)

    rng = np.random.RandomState(1984)

    dr = DynamicsRules(rng=rng)
    ter = TempoRules(rng=rng)
    ar = ArticulationRules(rng=rng)
    tir = TimingRules(rng=rng)

    velocity_trend, velocity_dev = dr.compute_parameters(part, W_notewise, names)
    beat_period = ter.compute_parameters(part, W_notewise, names)
    articulation_log = ar.compute_parameters(part, W_notewise, names)
    timing = tir.compute_parameters(part, W_notewise, names, velocity_trend=velocity_trend,
                                    velocity_dev=velocity_dev)
    
    parameter_names = ['beat_period', 'velocity_trend', 'velocity_dev', 'timing', 'articulation_log']

    predictions = np.zeros(N_n,
                           dtype=[(pn, 'f4') for pn in parameter_names])
    predictions['beat_period'] = beat_period
    predictions['velocity_trend'] = velocity_trend
    predictions['velocity_dev'] = velocity_dev
    predictions['timing'] = timing
    predictions['articulation_log'] = articulation_log
    

    pc = get_performance_codec(parameter_names)

    predicted_ppart = pc.decode(part, predictions)
    partitura.save_performance_midi(predicted_ppart, 'test2x.mid')
    
