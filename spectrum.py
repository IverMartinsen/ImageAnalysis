# -*- coding: utf-8 -*-
"""
Functions for spectral filtering and analysis.

Created on Mon Jun 21 12:28:51 2021

@author: iverm
"""

import numpy as np

def gaussian_ir(rectangle, spread, threshold):
    '''
    Produced gaussian impulse response filter to be
    multiplied by image spectrum.

    Parameters
    ----------
    rectangle : numpy.ndarray
        Image to be filtered.
    spread : float
        Gaussian standard deviation.
    threshold : float
        Maximum range of filter.

    Returns
    -------
    output : TYPE
        DESCRIPTION.

    '''    
    M, N = rectangle.shape
    
    output = np.ones((M, N))
    
    coordinates = np.array(np.meshgrid(np.arange(N), np.arange(M)))
    positions = np.array([np.tile(N/2, (M, N)), np.tile(M/2, (M, N))])
    
    d = np.linalg.norm(coordinates - positions, axis = 0)
    
    idx = np.where(d < threshold)
    
    output[idx] = (np.exp(-d[idx] / (2*spread**2)))
    
    return output

def gaussian_notch(rectangle, centers, spread):
    '''
    Produces gaussian notch filters centered at centers.

    Parameters
    ----------
    rectangle : numpy.ndarray
        Image to be filtered.
    centers : list
        List of center coordinates (tuples).
    spread : float
        Shape parameter for the normal distribution.

    Returns
    -------
    output : numpy.ndarray
        Filter to be multiplied by the image spectrum.

    '''
    M, N = rectangle.shape

    output = np.ones((M, N))

    for center in centers:
        v, u = center    
    
        coordinates = np.array(np.meshgrid(np.arange(N), np.arange(M)))
        positions1 = np.array([np.tile(N/2 + u, (M, N)), np.tile(M/2 + v, (M, N))])
        positions2 = np.array([np.tile(N/2 - u, (M, N)), np.tile(M/2 - v, (M, N))])
    
        d1 = np.linalg.norm(coordinates - positions1, axis = 0)
        d2 = np.linalg.norm(coordinates - positions2, axis = 0)
        
        output = output * (1 - np.exp(-d1 / (2*spread**2))) * (1 - np.exp(-d2 / (2*spread**2)))

    return output