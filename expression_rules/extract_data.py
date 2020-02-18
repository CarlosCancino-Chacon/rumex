import numpy as np

import partitura

import partitura.score as score
import basismixer.basisfunctions as bf
import basismixer

from basismixer.utils import get_unique_onset_idxs, notewise_to_onsetwise, onsetwise_to_notewise

import matplotlib.pyplot as plt

from dynamics import DYNAMICS_MARKINGS_VELOCITY
from tempo import TEMPO_MARKINGS_BPM


if __name__ == '__main__':

    fn = '/Users/aae/Downloads/mozart_k331.musicxml'
    part = partitura.load_musicxml(fn, force_note_ids=True)
    metrical, metrical_names = bf.metrical_strength_basis(part)
    tempo, tempo_names = bf.tempo_direction_basis(part)
    articulation, articulation_names = bf.articulation_basis(part)
    loudness, loudness_names = bf.loudness_direction_basis(part)

    W_notewise = np.column_stack((metrical, tempo, articulation, loudness))
    names = metrical_names + tempo_names + articulation_names + loudness_names
    # plt.plot(part.note_array['onset'], loudness[:, loudness_names.index('loudness_incr')])
    # plt.plot(part.note_array['onset'], loudness[:, loudness_names.index('loudness_decr')])
    # plt.show()

    onsets = part.note_array['onset']
    N_n = len(onsets)
    unique_onset_idxs, unique_onsets = get_unique_onset_idxs(onsets, return_unique_onsets=True)
    N_o = len(unique_onset_idxs)
    W_onsetwise = notewise_to_onsetwise(W_notewise, unique_onset_idxs)

    rng = np.random.RandomState(1984)

    trend_dynamics = np.ones(N_o) * 50
    for dm in DYNAMICS_MARKINGS_VELOCITY:

        if dm in names:
            di = names.index(dm)
            d_idx = np.where(W_onsetwise[:, di] == 1)[0]
            trend_dynamics[d_idx] = DYNAMICS_MARKINGS_VELOCITY[dm]

    
    # for i, (on, w) in zip(onsets, W_onsetwise):

    #     for dm in DYNAMICS_MARKINGS_VELOCITY:

    #         if dm in names:
    #             di = names.index(dm)
    #             current_constant = dm
                

    trend_tempo = np.ones(N_o) * 60 / 108.

    for tm in TEMPO_MARKINGS_BPM:
        if tm in names:
            ti = names.index(tm)
            t_idx = np.where(W_onsetwise[:, ti] == 1)[0]
            bp_low, bp_high = TEMPO_MARKINGS_BPM[tm]
            trend_tempo[t_idx] = 60. / rng.randint(bp_low, bp_high)

            
            
            
            

            
    
    

    
    # tss = np.array([(ts.start.t, ts.beats, ts.beat_type)
    #                 for ts in score.iter_all(TimeSignature)])
    # if len(tss) == 0:
    #     # default time sig
    #     beats, beat_type = 4, 4
    #     LOGGER.warning('No time signatures found, assuming {}/{}'
    #                    .format(beats, beat_type))
    #     if score.first_point is None:
    #         t0, tN = 0, 0
    #     else:
    #         t0 = score.first_point.t
    #         tN = score.last_point.t
    #     tss = np.array([(t0, beats, beat_type),
    #                     (tN, beats, beat_type)])
    # elif tss[0, 0] > score.first_point.t:
    #     tss = np.vstack(((score.first_point.t, tss[0, 1], tss[0, 2]),
    #                      tss))

    
