import numpy as np

import partitura

import partitura.score as score
import basismixer.basisfunctions as bf
import basismixer

from basismixer.utils import get_unique_onset_idxs, notewise_to_onsetwise, onsetwise_to_notewise

import matplotlib.pyplot as plt



if __name__ == '__main__':

    fn = '/Users/aae/Downloads/mozart_k331.musicxml'
    part = partitura.load_musicxml(fn, force_note_ids=True)
    metrical, metrical_names = bf.metrical_strength_basis(part)
    tempo, tempo_names = bf.tempo_direction_basis(part)
    articulation, articulation_names = bf.articulation_basis(part)
    loudness, loudness_names = bf.loudness_direction_basis(part)

    # plt.plot(part.note_array['onset'], loudness[:, loudness_names.index('loudness_incr')])
    # plt.plot(part.note_array['onset'], loudness[:, loudness_names.index('loudness_decr')])
    # plt.show()

    onsets = part.note_array['onset']
    unique_onset_idxs, unique_onsets = get_unique_onset_idxs(onsets, return_unique_onsets=True)

    
    
    

    
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

    
