import numpy as np
from scipy.interpolate import interp1d

from basismixer.utils import get_unique_onset_idxs, notewise_to_onsetwise, onsetwise_to_notewise

from base import PerformanceRules, _get_score_info

DYNAMICS_MARKINGS_VELOCITY = {"ppppp": (5, 12),
                              "pppp": (7, 17),
                              "ppp": (10, 25),
                              "pp": (15, 30),
                              "p": (30, 40),
                              "mp": (35, 45),
                              "mf": (40, 55),
                              "f": (65, 75),
                              "ff": (85, 100),
                              "fff": (90, 105),
                              "ffff": (95, 110),
                              "ffff": (100, 110),
                              "fffff": (102, 112),
                              "ffffff": (110, 120)
}

ACCENT_INCREASE = {"accent": (0.1, 0.2),
                   "fz": (0.15, 0.2),
                   "staccato": (0.02, 0.05),
                   "tenuto": (-0.02, -0.01),
}

CONSTANT_MARKINGS = np.array(['ppppp', 'pppp', 'ppp', 'pp', 'p',
                              'mp', 'mf',
                              'f', 'ff', 'fff', 'ffff', 'fffff'])

RNG = np.random.RandomState(1984)

DEFAULT_PIPELINE = ['dynamics_markings_rules',
                    'simple_accent_rules',
                    'polynomial_pitch_height_rules',
                    'motor_noise_rules'
]
                    
                    

class DynamicsRules(PerformanceRules):
    """
    Pipeline for computing dynamics-related expressive parameters
    (velocity_trend and velocity_dev) using rules
    """
    def __init__(self, pipeline=DEFAULT_PIPELINE, rng=RNG):
        super().__init__(pipeline=pipeline,
                         rng=rng,
                         module_globals=globals())

    def compute_parameters(self, part, W, names, *args, **kwargs):

        (pitch, onsets, duration, unique_onsets, unique_onset_idxs) = _get_score_info(part)

        velocity_trend = np.ones(len(W)) * 64
        velocity_dev = np.zeros(len(W))

        for func in self.pipeline:

            velocity_trend, velocity_dev = func(W=W,
                                                names=names,
                                                onsets=onsets,
                                                pitch=pitch,
                                                duration=duration,
                                                rng=self.rng,
                                                unique_onset_idxs=unique_onset_idxs,
                                                unique_onsets=unique_onsets,
                                                velocity_trend=velocity_trend,
                                                velocity_dev=velocity_dev,
                                                *args, **kwargs)

        return velocity_trend / 127, velocity_dev / 127


def dynamics_markings_rules(W, names, onsets, rng=RNG, unique_onset_idxs=None,
                            velocity_trend=None, velocity_dev=None,
                            *args, **kwargs):
    """
    Compute 
    """

    if unique_onset_idxs is None:
        unique_onset_idxs, unique_onsets = get_unique_onset_idxs(onsets, return_unique_onsets=True)
    else:
        unique_onsets = kwargs.get('unique_onsets', None)

        if unique_onsets is None:
            unique_onsets = np.array([onsets[uix].mean() for uix in unique_onset_idxs])
        else:
            assert len(unique_onsets) == len(unique_onset_idxs)

    # Get onsetwise features
    W_onsetwise = notewise_to_onsetwise(W, unique_onset_idxs)

    # Compute dynamics markings map
    (dynamics_markings_map, next_dynamics_markings_map,
     dm_array) = _get_constant_dynamics_markings_map(W_onsetwise=W_onsetwise,
                                                     names=names,
                                                     unique_onsets=unique_onsets,
                                                     rng=rng)
    # sample velocity per dynamics marking
    velocity_per_dm = _sample_velocity_per_dm(dm_array=dm_array, rng=rng)

    # Initialize velocity trend
    velocity_trend = _initialize_velocity_trend(unique_onsets, dm_array, velocity_per_dm)

    # Compute crescendo profile
    if 'loudness_incr' in names:
        crescendo = W_onsetwise[:, names.index('loudness_incr')]
    else:
        crescendo = np.zeros(len(unique_onsets))

    if 'loudness_decr' in names:
        diminuendo = W_onsetwise[:, names.index('loudness_decr')]
    else:
        diminuendo = np.zeros(len(unique_onsets))

    cresc_dim_profile = compute_crescendo_profile(crescendo=crescendo,
                                                  diminuendo=diminuendo,
                                                  unique_onsets=unique_onsets,
                                                  dynamics_markings_map=dynamics_markings_map,
                                                  dm_array=dm_array,
                                                  velocity_per_dm=velocity_per_dm)

    velocity_trend += cresc_dim_profile
    velocity_trend = onsetwise_to_notewise(velocity_trend, unique_onset_idxs)

    if velocity_dev is None:
        velocity_dev = np.zeros_like(velocity_trend)

    return velocity_trend, velocity_dev

def simple_accent_rules(W, names, velocity_trend=None, velocity_dev=None, rng=RNG,
                        *args, **kwargs):

    if velocity_trend is None:
        velocity_trend = np.ones(len(W)) * 50
    if velocity_dev is None:
        velocity_dev = np.zeros_like(velocity_trend)

    velocity_dev += compute_accents(W=W,
                                    names=names,
                                    velocity_trend=velocity_trend,
                                    rng=rng)

    return velocity_trend, velocity_dev

def polynomial_pitch_height_rules(pitch, velocity_trend=None, velocity_dev=None, *args, **kwargs):

    if velocity_trend is None:
        velocity_trend = np.ones(len(pitch)) * 50
    if velocity_dev is None:
        velocity_dev = np.zeros_like(velocity_trend)

    velocity_dev += (1 - centered_quadratic_pitch(pitch,
                                                  max_pitch=pitch.max(),
                                                  min_pitch=pitch.min())) * velocity_trend

    return velocity_trend, velocity_dev

def motor_noise_rules(velocity_trend, velocity_dev, rng=RNG, std=0.05, *args, **kwargs):

    velocity_dev += rng.normal(loc=0, scale=std, size=len(velocity_dev)) * velocity_trend

    return velocity_trend, velocity_dev


    
def _get_constant_dynamics_markings_map(W_onsetwise, names, unique_onsets, rng=RNG):
    """
    Returns a map of the dynamics marking for each onset, as well as a map of the next
    constant dynamics markings and an array with the position in beats of each
    constant dynamics markings.
    """
    # get starting point of all constant dynamics markings
    dm_array = []
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

    # if there are no constant dynamics markings, assume mf
    if len(dm_array) == 0:
        dm_array.append((unique_onsets.min(), int(np.where(CONSTANT_MARKINGS == 'mf')[0])))
    dm_array = np.array(dm_array)

    # assume mf if there is no dynamics marking at the begining of the piece
    if dm_array[0, 0] != unique_onsets.min():
        dm_array = np.insert(dm_array, 0, (unique_onsets.min(),
                                           int(np.where(CONSTANT_MARKINGS == 'mf')[0])))

    # Create dynamics markings map
    if len(dm_array) > 1:
        dynamics_markings_map = interp1d(dm_array[:, 0], dm_array[:, 1], kind='previous',
                                         bounds_error=False,
                                         fill_value=(dm_array[0, 1], dm_array[-1, 1]))
        next_dynamics_markings_map = interp1d(dm_array[:-1, 0], dm_array[1:, 1], kind='previous',
                                              bounds_error=False,
                                              fill_value=(dm_array[0, 1], dm_array[-1, 1]))
    # dummy map in case there is only one dynamics marking (avoid error with interp1d)
    else:
        dynamics_markings_map = next_dynamics_markings_map = lambda onset: dm_array[0, 1]


    return dynamics_markings_map, next_dynamics_markings_map, dm_array

def _sample_velocity_per_dm(dm_array, rng=RNG):
    """Sample a MIDI velocity for each dynamics marking
    """
    velocity_per_dm = [rng.randint(*DYNAMICS_MARKINGS_VELOCITY[CONSTANT_MARKINGS[int(dm)]])
                       for _, dm in dm_array]

    return velocity_per_dm

def _initialize_velocity_trend(unique_onsets, dm_array, velocity_per_dm):
    """
    Helper method to initialize the velocity_trend array with a constant
    velocity per dynamics marking
    """
    velocity_trend = np.zeros(len(unique_onsets))

    for i, (vel, (start, _)) in enumerate(zip(velocity_per_dm, dm_array)):
        try:
            end = dm_array[i + 1, 0]
        except IndexError:
            end = unique_onsets.max() + 1

        idxs = np.where(np.logical_and(unique_onsets >= start, unique_onsets <= end))[0]
        velocity_trend[idxs] = vel


    return velocity_trend
            

def compute_crescendo_profile(crescendo, diminuendo, unique_onsets, dynamics_markings_map,
                              dm_array, velocity_per_dm, default_dyn_diff=10):
    """
    Profile of crescendo and diminuendo

    Parameters
    ----------
    crescendo : np.ndarray
        Value from the crescendo basis function for each score onset (i.e., `unique_onsets`)
        The basis function evaluates to 0 at the start of the crescendo mark and 1 at the end and
        zero otherwise. See `basismixer.basisfunction.loudness_direction_basis`.
    diminuendo : np.ndarray
        Value from the diminuendo basis function for each score onset (i.e., `unique_onsets`)
        The basis function evaluates to 0 at the start of the diminuendo mark and 1 at the end and
        zero otherwise. See `basismixer.basisfunction.loudness_direction_basis`.    
    unique_onsets : np.ndarray
        Unique score positions in beats in the score
    dynamics_markings_map : callable
        A method which takes as inputs the score position in beats and returns the index of the dynamics
        marking
    dm_array : np.ndarray
        An array indicating the starting times and dynamics markings (as indices of CONSTANT_MARKINGS)
        in the score.
    velocity_per_dm : iterable
        A list with the value of the constant MIDI velocity for each instance of a dynamics marking
        in the score.
    default_dyn_diff : int or float
       Default MIDI velocity for crescendo or diminuendo (if there is only a single dynamics marking)

    Returns
    -------
    cresc_dim_profile : np.array
        An array with the change in MIDI velocity for each score onset resulting from
        crescendo and diminuendo markings.
    """
    crescendo_profile = []
    diminuendo_profile = []
    prev_crescendo = False
    prev_diminuendo = False
    prev_dyn = dynamics_markings_map(unique_onsets.min())
    reset_cresc = True

    # compute crescendi and diminuendo profiles
    for on, cresc, dim in zip(unique_onsets, crescendo, diminuendo):
        # current dynamics marking
        c_dyn = int(dynamics_markings_map(on))
        # next dynamics marking
        # n_dyn = int(next_dynamics_markings_map(on))

        # if the new dynamics marking is different than the previous one
        # reset crescendo profile
        if c_dyn != prev_dyn:
            reset_cresc = True
            # update prev_dyn
            prev_dyn = c_dyn
        else:
            # Otherwise do not reset profile
            reset_cresc = False

        # if there is a crescendo
        if cresc != 0:
            # start from scratch if the profile is reset
            if reset_cresc:
                crescendo_profile.append(cresc)
            # else continue from previous value
            else:
                crescendo_profile.append(crescendo_profile[-1] + cresc)
            prev_crescendo = True

        # keep the last value if the previous onset was
        # part of a crescendo, until the next dynamics markings
        elif cresc == 0 and prev_crescendo:
            # reset profile if there is a change in constant dynamics marking
            if reset_cresc:
                crescendo_profile.append(0)
            # else keep the last value as constant
            else:
                crescendo_profile.append(crescendo_profile[-1])

        # if there is no crescendo, or it does not follow from the
        # previous value, the profile is 0
        else:
            crescendo_profile.append(0)
            prev_crescendo = False

        # if there is a diminuendo do a similar procedure
        if dim != 0:
            if reset_cresc:
                diminuendo_profile.append(dim)
            else:
                diminuendo_profile.append(diminuendo_profile[-1] - dim)
            prev_diminuendo = True
        elif dim == 0 and prev_diminuendo:
            if reset_cresc:
                diminuendo_profile.append(0)
            else:
                diminuendo_profile.append(diminuendo_profile[-1])
        else:
            diminuendo_profile.append(0)
            prev_diminuendo = False
            
    crescendo_profile = np.array(crescendo_profile)
    diminuendo_profile = np.array(diminuendo_profile)

    # The full profile is the sum of both crescendo and diminuendo
    # TODO: Add a nonlinearity here?
    cresc_dim_profile = crescendo_profile + diminuendo_profile

    # scale according to dynamics markings:
    # how much it increases or decreases
    # depends on which is the next dynamics marking
    for i, (d_start, dm) in enumerate(dm_array):
        # constant MIDI velocity for the current section
        vel_start = velocity_per_dm[i]
        try:
            # get next midi velocity and onset of the next
            # dynamics marking
            d_end, _ = dm_array[i + 1]
            vel_end = velocity_per_dm[i + 1]
        except IndexError:
            # otherwise use the current one (for the end of the piece)
            d_end = unique_onsets.max() + 1
            # n_dm = dm
            vel_end = velocity_per_dm[i]

        # Indices of the dynamics markings
        dm_idxs = np.where(np.logical_and(unique_onsets >= d_start,
                                          unique_onsets < d_end))[0]
        # compute difference of MIDI velocity
        dyn_diff = vel_end - vel_start

        if abs(dyn_diff) != 0:
            # It will increse/decrease at most 90% of the difference
            dyn_multiplier = 0.9 * abs(dyn_diff)
        else:
            # It will increase/decrease at most 10 MIDI velocity units
            dyn_multiplier = default_dyn_diff

        # scale for the section
        scale = max(abs(cresc_dim_profile[dm_idxs]))
        if scale == 0:
            # if there is no crescendo or diminuendo
            scale = 1.0

        # Scale the profile
        cresc_dim_profile[dm_idxs] *= dyn_multiplier / scale

    return cresc_dim_profile

def compute_accents(W, names, velocity_trend, rng=RNG):

    velocity_dev = np.zeros(len(W))
    for i, (w, vt) in enumerate(zip(W, velocity_trend)):

        for accent_type in ACCENT_INCREASE:
            if accent_type in names:
                if w[names.index(accent_type)] != 0:
                    # import pdb
                    # pdb.set_trace()
                    velocity_dev[i] -= vt * rng.uniform(*ACCENT_INCREASE[accent_type])

    return velocity_dev

def centered_quadratic_pitch(pitch, min_proportion=0.8, max_pitch=96, min_pitch=24,
                             max_proportion=1.2):
    
    b = np.sqrt(max_proportion - min_proportion) * (1 + min_pitch / max_pitch) / (1 - min_pitch / max_pitch )
    a = (np.sqrt(max_proportion - min_proportion) + b) / max_pitch

    return np.clip((a * pitch - b) ** 2 + min_proportion, min_proportion, 1.0)


if __name__ == '__main__':

    import partitura
    import basismixer.basisfunctions as bf
    

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

    dr = DynamicsRules(rng=rng)

    velocity_trend, velocity_dev = dr.compute_dynamics(part, W_notewise, names)

    
    velocity_dev = compute_accents(W_notewise, names, velocity_trend, rng=rng)
