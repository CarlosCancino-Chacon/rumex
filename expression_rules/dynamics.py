import numpy as np

DYNAMICS_MARKINGS_VELOCITY = {"pp": 25,
                              "p": 40,
                              "mf": 50,
                              "f": 70,
                              "ff": 90}

                              

def polynomial_pitch_model(pitch, coefficients):

    X = np.column_stack([x ** i for i in range(len(coefficients))])

    Y = np.dot(X, coefficients)

    return Y

def dynamics_marking(onsets):

    
    return 1


def accents(notes, accent, accent_percentage=.5):

    return accent * accent_percentage


def dynamics(basis_functions):

    
    return 0


    
