# -*- coding: utf-8 -*-
"""
This module contains functions for filtering images

Created on Mon Jun 21 12:28:04 2021

@author: iverm
"""

import numpy as np

def median_filter(image, size):
    '''
    Produces an image filtered by a median filter.

    Parameters
    ----------
    image : numpy.ndarray
        Unpadded input image.
    size : tuple
        Filter size.

    Returns
    -------
    output : numpy.ndarray
        Filtered image with size as input.

    '''
    h, w = image.shape
    m, n = size
        
    image = np.pad(image, (((m - 1) // 2, (m - 1) // 2),
                           ((n - 1) // 2, (n - 1) // 2)))
    
    image_col = np.zeros((h*w, m*n))

    for i in range(h):
        for j in range(w):
            image_col[i*w + j, :] = image[i:i + m, j:j + n].reshape(-1)
    
    output = np.median(image_col, axis = 1).reshape(h, w)
    
    return output

def convolve2d(x, w, stride = 1, mode = "valid"):
    '''
    Performs 2d convolution between x and w.
    w is rotated and output is similar to scipy.signal.convolve2d.

    Parameters
    ----------
    x : numpy.ndarray
        2-D array to be filtered.
    w : numpy.ndarray
        2-D filter.
    stride : int
        movement/stride.
        The default is 1.
    mode : str, optional
        Options are "full", "valid" and "same".
        The default is "valid".

    Raises
    ------
    ValueError
        If unsupported mode.

    Returns
    -------
    y : numpy.ndarray
        2-D array of filtered values.

    '''
    filterheight, filterwidth = w.shape
    
    if mode == 'valid':
        pass
    
    elif mode == 'full':
        x = np.pad(x, (filterheight - 1,
                       filterwidth - 1))
        
    elif mode == 'same':
        x = np.pad(x, (filterheight // 2,
                       filterwidth // 2))
    
    else:
        raise ValueError
    
    H, W = x.shape
    
    H_out = (H - filterheight) // stride + 1
    W_out = (W - filterwidth) // stride + 1
    
    x_col = np.zeros([H_out * W_out, filterheight * filterwidth])

    w_col = np.flip(w.reshape(-1))

    for i in range(H_out):
       for j in range(W_out):
           patch = x[i*stride : i*stride + filterheight,
                     j*stride : j*stride + filterwidth]
           x_col[i*W_out + j, :] = np.reshape(patch, -1)
    
    y_col = np.matmul(x_col, w_col)
    
    y = y_col.reshape(H_out, W_out)
    
    return y

def spatial_filter(image, filtersize, method, d = 0):
    '''
    Produces spatially filtered image with size as input image.
    Assumes filtersize of odd numbers.

    Parameters
    ----------
    image : numpy.array
        Input image.
    filtersize : int or tuple
        Filtersize.
    method : str
        Filter method to be used. Options are
        'arithmetic', 'geometric', 'harmonic', 'alpha_trimmed'
        and 'median'.
    d : int
        Optional. To be used with the alpha trimmed mean.
        The default is 0.
    Raises
    ------
    NameError
        If method not found.

    Returns
    -------
    output : numpy.array
        Filtered image.

    '''

    if d > np.prod(filtersize):
        raise ValueError('d must be less than filterheight * filterwidth')

    h, w = image.shape
    try:
        m, n = filtersize
    except TypeError:
        m = filtersize
        n = filtersize
    
    image = np.pad(image, (((m - 1) // 2, (m - 1) // 2),
                           ((n - 1) // 2, (n - 1) // 2)))
    
    image_col = np.zeros((h*w, m*n))
    
    for i in range(h):
        for j in range(w):
            image_col[i*w + j, :] = image[i:i + m, j:j + n].reshape(-1)
    
    if method == 'arithmetic':
        output = np.mean(image_col, axis = 1).reshape(h, w)
    elif method == 'geometric':
        output = (np.prod(image_col, axis = 1) ** (1 / (m * n))).reshape(h, w)
    elif method == 'harmonic':
        output = (m*n / np.sum(1 / image_col, axis = 1)).reshape(h, w)
    elif method == 'median':
        output = np.median(image_col, axis = 1).reshape(h, w)
    elif method == 'alpha_trimmed':
        image_col = np.sort(image_col, axis = 1)[:, d:m*n - d]
        output = np.mean(image_col, axis = 1).reshape(h, w)
    else:
        raise NameError('method not found')
        
    return output