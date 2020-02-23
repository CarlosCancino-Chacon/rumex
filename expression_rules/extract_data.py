import numpy as np
from scipy.interpolate import interp1d
import partitura

import partitura.score as score
import basismixer.basisfunctions as bf
import basismixer

from basismixer.utils import get_unique_onset_idxs, notewise_to_onsetwise, onsetwise_to_notewise

from basismixer.performance_codec import get_performance_codec

import matplotlib.pyplot as plt

from dynamics import DYNAMICS_MARKINGS_VELOCITY
from tempo import TEMPO_MARKINGS_BPM

CONSTANT_MARKINGS = np.array(['pp', 'p', 'mf', 'f', 'ff'])

def melody_lead(pitch, velocity, lead=0.02):
    """
    Compute melody lead
    """
    return (np.exp((pitch - 127) / 127.) *
            np.exp(-(velocity - 127) / 127) * lead)


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


if __name__ == '__main__':

    fn = '/Users/aae/Downloads/mozart_k331.musicxml'
    part = partitura.load_musicxml(fn, force_note_ids=True)

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

    velocity_trend = np.ones(N_o) * 50
    for dm in DYNAMICS_MARKINGS_VELOCITY:

        if dm in names:
            di = names.index(dm)
            d_idx = np.where(W_onsetwise[:, di] == 1)[0]
            velocity_trend[d_idx] = DYNAMICS_MARKINGS_VELOCITY[dm]

    dm_array = []
    cresc_idx = names.index('loudness_incr')
    dim_idx = names.index('loudness_decr')

    for on, w in zip(unique_onsets, W_onsetwise):
        for ci, dm in enumerate(CONSTANT_MARKINGS):
            if dm in names:
                di = names.index(dm)
                if w[di] != 0:
                    try:
                        if dm_array[-1][1] != ci:
                            dm_array.append((on, ci))
                    except:
                        dm_array.append((on, ci))

    if len(dm_array) == 0:
        dm_array.append((onsets.min(), int(np.where(CONSTANT_MARKINGS == 'mf')[0])))
    dm_array = np.array(dm_array)

    # assume mf if there is no dynamics marking at the begining of the piece
    if dm_array[0, 0] != onsets.min():
        dm_array = np.insert(dm_array, 0, (onsets.min(), int(np.where(CONSTANT_MARKINGS == 'mf')[0])))

    if len(dm_array) > 1:
        dynamics_markings_map = interp1d(dm_array[:, 0], dm_array[:, 1], kind='previous',
                                         bounds_error=False,
                                         fill_value=(dm_array[0, 1], dm_array[-1, 1]))
        next_dynamics_markings_map = interp1d(dm_array[:-1, 0], dm_array[1:, 1], kind='previous',
                                              bounds_error=False,
                                              fill_value=(dm_array[0, 1], dm_array[-1, 1]))

    else:
        dynamics_markings_map = next_dynamics_markings_map = lambda onset: dm_array[0, 1]


    crescendo_profile = []
    diminuendo_profile = []
    prev_crescendo = False
    prev_diminuendo = False
    prev_dyn = dynamics_markings_map(unique_onsets.min())
    reset_cres = True
    default_dyn_diff = 10
    
    for on, w in zip(unique_onsets, W_onsetwise):
        c_dyn = int(dynamics_markings_map(on))
        n_dyn = int(next_dynamics_markings_map(on))
        if c_dyn != prev_dyn:
            reset_cres = True
            prev_dyn = c_dyn
        else:
            reset_cres = False

        cresc = w[cresc_idx]
        dim = w[dim_idx]
        if cresc != 0:
            if reset_cres:
                crescendo_profile.append(cresc)
            else:
                crescendo_profile.append(crescendo_profile[-1] + cresc)
            prev_crescendo = True

        elif cresc == 0 and prev_crescendo:
            if reset_cres:
                crescendo_profile.append(0)
            else:
                crescendo_profile.append(crescendo_profile[-1])        
        else:
            crescendo_profile.append(0)
            prev_crescendo = False

        if dim != 0:
            if reset_cres:
                diminuendo_profile.append(dim)
            else:
                diminuendo_profile.append(diminuendo_profile[-1] - dim)
            prev_diminuendo = True
        elif dim == 0 and prev_diminuendo:
            if reset_cres:
                diminuendo_profile.append(0)
            else:
                diminuendo_profile.append(diminuendo_profile[-1])
        else:
            diminuendo_profile.append(0)
            prev_diminuendo = False

    crescendo_profile = np.array(crescendo_profile)
    diminuendo_profile = np.array(diminuendo_profile)

    cresc_dim_profile = crescendo_profile + diminuendo_profile

    for i, (d_start, dm) in enumerate(dm_array):

        try:
            d_end, n_dm = dm_array[i + 1]            
        except:
            d_end = unique_onsets.max() + 1
            n_dm = dm

        dm_idxs = np.where(np.logical_and(unique_onsets >= d_start, unique_onsets < d_end))[0]

        dyn_diff = (DYNAMICS_MARKINGS_VELOCITY[CONSTANT_MARKINGS[int(n_dm)]] -
                    DYNAMICS_MARKINGS_VELOCITY[CONSTANT_MARKINGS[int(dm)]])

        if dyn_diff > 0:
            dyn_multiplier = 0.9 * dyn_diff
        else:
            dyn_multiplier = 0.9 * default_dyn_diff

        
        scale = max(abs(cresc_dim_profile[dm_idxs]))
        if scale == 0:
            scale = 1.0
        cresc_dim_profile[dm_idxs] *= dyn_multiplier / scale

    # plt.plot(unique_onsets, crescendo_profile)
    # plt.plot(unique_onsets, diminuendo_profile)
    # plt.plot(unique_onsets, crescendo_profile + diminuendo_profile)
    plt.plot(unique_onsets, cresc_dim_profile)
    plt.plot(unique_onsets, cresc_dim_profile + velocity_trend)

    velocity_trend += cresc_dim_profile

    max_dyn = max(abs(crescendo_profile).max(), abs(diminuendo_profile).max())

    for on, dm in dm_array:
        x = np.ones(10) * on
        y = np.linspace(-max_dyn, max_dyn, 10)
        plt.plot(x, y)    
    plt.show()
        
        

        
            
        
    beat_period = np.ones(N_o) * 60 / 108.

    for tm in TEMPO_MARKINGS_BPM:
        if tm in names:
            ti = names.index(tm)
            t_idx = np.where(W_onsetwise[:, ti] == 1)[0]
            bp_low, bp_high = TEMPO_MARKINGS_BPM[tm]
            beat_period[t_idx] = 60. / rng.randint(bp_low, bp_high)


    measure_array = []
    for m in part.iter_all(score.Measure):
        measure_array.append((m.number,) + tuple(part.beat_map([m.start.t, m.end.t])))

    measure_array = np.array(measure_array)
    measure_map = interp1d(measure_array[:, 0], measure_array[:, 1],
                           kind='previous',
                           bounds_error=False,
                           fill_value=(measure_array[:, 1].min(),
                                       measure_array[:, 1].max()))
    inv_measure_map = interp1d(measure_array[:, 1], measure_array[:, 0],
                           kind='previous',
                           bounds_error=False,
                           fill_value=(measure_array[:, 0].min(),
                                       measure_array[:, 0].max()))

    n_bars_rit = 2
    final_ritard_idxs = np.where(unique_onsets >= measure_array[-n_bars_rit, 1])[0]
    rit_curve = friberg_sundberg_rit(len(final_ritard_idxs))
    beat_period[final_ritard_idxs] *= rit_curve

    plt.plot(unique_onsets, beat_period)
    plt.show()


    velocity_dev = np.zeros(N_n)
    timing = np.zeros(N_n)
    articulation = np.ones(N_n)

    
    parameter_names = ['beat_period', 'velocity_trend', 'velocity_dev', 'timing', 'articulation_log']

    predictions = np.zeros(N_n,
                           dtype=[(pn, 'f4') for pn in parameter_names])
    predictions['beat_period'] = onsetwise_to_notewise(beat_period, unique_onset_idxs)
    predictions['velocity_trend'] = onsetwise_to_notewise(velocity_trend / 127., unique_onset_idxs)
    predictions['velocity_dev'] = velocity_dev / 127.
    predictions['timing'] = timing
    predictions['articulation_log'] = 0 * articulation
    

    pc = get_performance_codec(parameter_names)

    predicted_ppart = pc.decode(part, predictions)
    partitura.save_performance_midi(predicted_ppart, 'test.mid')
    
