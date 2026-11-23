# Copyright (C) 2020-2026 Carlos Cancino-Chacón
#
# This file is part of RuMEx.
#
# RuMEx is free software: you can redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either version 3 of the License, or (at
# your option) any later version.
#
# RuMEx is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License along with RuMEx. If not, see
# <https://www.gnu.org/licenses/>.

from abc import ABC, abstractmethod

import partitura

import numpy as np
from scipy.interpolate import interp1d

from partitura.musicanalysis.performance_codec import (
    get_unique_onset_idxs,
    notewise_to_onsetwise,
    onsetwise_to_notewise,
)

RNG = np.random.RandomState(1984)


def minmax(x, lowest, highest):
    """Linearly rescale `x` to the range [`lowest`, `highest`] (all `lowest` if `x` is constant)."""
    x = np.asarray(x, dtype=float)
    x_min, x_max = x.min(), x.max()
    if np.isclose(x_max, x_min):
        x_hat = np.zeros_like(x)
    else:
        x_hat = (x - x_min) / (x_max - x_min)
    return (highest - lowest) * x_hat + lowest


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
    note_array = part.note_array()

    onset_unit, duration_unit = partitura.utils.music.get_time_units_from_note_array(note_array)
    onsets = note_array[onset_unit]
    duration = note_array[duration_unit]
    pitch = note_array['pitch']

    unique_onset_idxs, unique_onsets = get_unique_onset_idxs(onsets, return_unique_onsets=True)

    return pitch, onsets, duration, unique_onsets, unique_onset_idxs


def _get_constant_markings_array_and_maps(W_onsetwise, names, unique_onsets,
                                          constant_markings, default_marking):
    """
    Get an array indicating the starting points of constant markings, a map (a callable method) 
    of the constant marking for each onset, as well as the map of the next constant marking.
    """
    # get starting point of all constant markings
    cm_array = []
    for on, w in zip(unique_onsets, W_onsetwise):

        for ci, dm in enumerate(constant_markings):
            if dm in names:
                di = names.index(dm)
                if w[di] != 0:
                    try:
                        if cm_array[-1][1] != ci and cm_array[-1][0] != on:
                            # if cm_array[-1][0] != on:
                            #     # Case where there are two dynamics markings with the same
                            #     # onset
                            cm_array.append((on, ci))
                    except:
                        cm_array.append((on, ci))
                        
    # if there are no constant markings, assume `default_marking`
    if len(cm_array) == 0:
        cm_array.append((unique_onsets.min(), int(np.where(constant_markings == default_marking)[0][0])))
    cm_array = np.array(cm_array)

    # assume `default_marking` if there is no marking at the begining of the piece
    if cm_array[0, 0] != unique_onsets.min():
        cm_array = np.insert(cm_array, [0],
                             np.array((unique_onsets.min(),
                                       int(np.where(constant_markings == default_marking)[0][0]))),
                             axis=0)

    # Create constant markings map
    if len(cm_array) > 1:
        constant_markings_map = interp1d(cm_array[:, 0], cm_array[:, 1], kind='previous',
                                         bounds_error=False,
                                         fill_value=(cm_array[0, 1], cm_array[-1, 1]))
        if len(cm_array) > 2:
            next_constant_markings_map = interp1d(cm_array[:-1, 0], cm_array[1:, 1], kind='previous',
                                                  bounds_error=False,
                                                  fill_value=(cm_array[0, 1], cm_array[-1, 1]))
        else:
            # dummy map in case there are only two constant markings (avoid error with interp1d)
            next_constant_markings_map = lambda onset: cm_array[-1, 1]
    # dummy map in case there is only one constant marking (avoid error with interp1d)
    else:
        constant_markings_map = next_constant_markings_map = lambda onset: cm_array[0, 1]

    return cm_array, constant_markings_map, next_constant_markings_map


def _initialize_onsetwise_parameter(unique_onsets, cm_array, parameter_per_marking):
    """
    Helper method to initialize an onsetwise parameter with a constant
    value per (instance) of constant markings.
    """
    parameter = np.zeros(len(unique_onsets))

    for i, (vel, (start, _)) in enumerate(zip(parameter_per_marking, cm_array)):
        try:
            end = cm_array[i + 1, 0]
        except IndexError:
            end = unique_onsets.max() + 1

        idxs = np.where(np.logical_and(unique_onsets >= start, unique_onsets <= end))[0]
        parameter[idxs] = vel


    return parameter


def compute_increase_decrease_profile(increase, decrease, unique_onsets, constant_markings_map,
                                      cm_array, parameter_per_marking, default_diff,
                                      max_multiplier=0.9):
    """
    Profile of gradual increase and decrease of an onsetwise parameter

    Parameters
    ----------
    increase : np.ndarray
        Value of an `_incr` note feature for each score onset (i.e., `unique_onsets`)
        The note feature evaluates to 0 at the start of the crescendo mark and 1 at the end and
        zero otherwise. See `partitura.musicanalysis.note_features.loudness_direction_feature`.
    decrease : np.ndarray
        Value from the `_decr` note feature for each score onset (i.e., `unique_onsets`)
        The note feature evaluates to 0 at the start of the diminuendo mark and 1 at the end and
        zero otherwise. See `partitura.musicanalysis.note_features.loudness_direction_feature`.
    unique_onsets : np.ndarray
        Unique score positions in beats in the score
    constant_markings_map : callable
        A method which takes as inputs the score position in beats and returns the index of the constant
        marking
    cm_array : np.ndarray
        An array indicating the starting times and markings (as indices of CONSTANT_MARKINGS)
        in the score.
    parameter_per_marking : iterable
        A list with the value of the constant parameter for each instance of a constant marking
        in the score.
    default_diff : int or float
       Default maximal range for increasing or decreasing (if there is only a single constant marking)

    Returns
    -------
    cresc_dim_profile : np.array
        An array with the change in MIDI velocity for each score onset resulting from
        crescendo and diminuendo markings.
    """
    increase_profile = []
    decrease_profile = []
    prev_increase = False
    prev_decrease = False
    prev_mark = constant_markings_map(unique_onsets.min())
    reset_increase = True

    # compute increase and decrease profiles
    for on, incr, decr in zip(unique_onsets, increase, decrease):
        # current dynamics marking
        c_mark = int(constant_markings_map(on))

        # if the new dynamics marking is different than the previous one
        # reset increase profile
        if c_mark != prev_mark:
            reset_increase = True
            # update prev_mark
            prev_mark = c_mark
        else:
            # Otherwise do not reset profile
            reset_increase = False

        # if there is a increase
        if incr != 0:
            # start from scratch if the profile is reset
            if reset_increase:
                increase_profile.append(incr)
            # else continue from previous value
            else:
                increase_profile.append(increase_profile[-1] + incr)
            prev_increase = True

        # keep the last value if the previous onset was
        # part of a increase, until the next dynamics markings
        elif incr == 0 and prev_increase:
            # reset profile if there is a change in constant dynamics marking
            if reset_increase:
                increase_profile.append(0)
            # else keep the last value as constant
            else:
                increase_profile.append(increase_profile[-1])

        # if there is no increase, or it does not follow from the
        # previous value, the profile is 0
        else:
            increase_profile.append(0)
            prev_increase = False

        # if there is a decrease do a similar procedure
        if decr != 0:
            if reset_increase:
                decrease_profile.append(decr)
            else:
                decrease_profile.append(decrease_profile[-1] - decr)
            prev_decrease = True
        elif decr == 0 and prev_decrease:
            if reset_increase:
                decrease_profile.append(0)
            else:
                decrease_profile.append(decrease_profile[-1])
        else:
            decrease_profile.append(0)
            prev_decrease = False

    increase_profile = np.array(increase_profile)
    decrease_profile = np.array(decrease_profile)

    # The full profile is the sum of both increase and decrease
    # TODO: Add a nonlinearity here?
    incr_decr_profile = (increase_profile + decrease_profile).astype(float)

    # scale according to dynamics markings:
    # how much it increases or decreases
    # depends on which is the next dynamics marking
    for i, (d_start, dm) in enumerate(cm_array):
        # constant MIDI velocity for the current section
        vel_start = parameter_per_marking[i]
        try:
            # get next midi velocity and onset of the next
            # dynamics marking
            d_end, _ = cm_array[i + 1]
            vel_end = parameter_per_marking[i + 1]
        except IndexError:
            # otherwise use the current one (for the end of the piece)
            d_end = unique_onsets.max() + 1
            # n_dm = dm
            vel_end = parameter_per_marking[i]

        # Indices of the dynamics markings
        cm_idxs = np.where(np.logical_and(unique_onsets >= d_start,
                                          unique_onsets < d_end))[0]
        # compute difference of MIDI velocity
        dyn_diff = vel_end - vel_start

        if abs(dyn_diff) != 0:
            # It will increse/decrease at most 90% of the difference
            dyn_multiplier = max_multiplier * abs(dyn_diff)
        else:
            # It will increase/decrease at most 10 MIDI velocity units
            dyn_multiplier = default_diff

        # scale for the section
        # import pdb
        # pdb.set_trace()
        scale = float(max(abs(incr_decr_profile[cm_idxs])))
        if scale == 0:
            # if there is no increase or decrease
            scale = 1.0

        # Scale the profile        
        incr_decr_profile[cm_idxs] *= dyn_multiplier / scale

    return incr_decr_profile








