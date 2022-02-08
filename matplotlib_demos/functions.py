"""
Functions for data clustering demonstrations
"""
import numpy as np

def Gaussian(x, mu, sigma):
    """Simple normal distribution with height 1"""
    return np.exp(-0.5*((x-mu)/sigma)**2)