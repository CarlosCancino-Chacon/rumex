import numpy as np


def melody_lead(pitch, velocity, lead=0.02):
    """
    Compute melody lead
    """
    return (np.exp((pitch - 127) / 127.) *
            np.exp(-(velocity - 127) / 127) * lead)
