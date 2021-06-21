# -*- coding: utf-8 -*-
"""
Functions for image segmentation and analysis.

Created on Mon Jun 21 12:28:51 2021

@author: iverm
"""

import numpy as np
from filters import convolve2d

def threshold(image, channels, thresholds, color_space = 'rgb'):
    '''
    Performs thresholding of image along given channels
    for given threshold values

    Parameters
    ----------
    image : PIL.Image
        Input image.
    channels : tuple
        Tuple of string. Channels to be thresholded.
    thresholds : list
        List of floats. Threshold values.
    color_space : str
        Channel names. The default is 'rgb'.

    Returns
    -------
    output_image : numpy.ndarray
        Output image.

    '''    
    # normalise image
    image = np.array(image, dtype = np.float32) / np.max(image)
    
    if color_space == 'rgb':
        colors = 'r', 'g', 'b'
    elif color_space == 'hsi':
        colors = 'h', 's', 'i'
    else:
        raise ValueError('Channels not found')
    
    output_image = image
    
    for idx, color in enumerate(colors):
        for j, channel in enumerate(channels):
            if color == channel:
                output_image[:, :, idx] = (image[:, :, idx] > thresholds[j])
    
    return output_image

def image_hist(image, bins = 50, density = True,
               normalise = False, color_space = 'rgb'):
    '''
    Plots histogram of image channels.

    Parameters
    ----------
    image : PIL.Image
        Input image.
    bins : int, optional
        Number of bins. The default is 50.
    density : Boolean, optional
        If histogram should be normalized.
        The default is True.
    normalise : Boolean, optional
        If image should be normalized before plotting.
        The default is False.
    color_space : str, optional
        Color space titles for the axes.
        The default is 'rgb'.

    Returns
    -------
    None.

    '''
    # normalise image
    if normalise:
        image = np.array(image, dtype = np.float32) / np.max(image)
    else:
        image = np.array(image, dtype = np.float32)
    
    # ax titles
    if color_space == 'rgb':
        channels = 'Red', 'Green', 'Blue'
    elif color_space == 'hsi':
        channels = 'Hue', 'Saturation', 'Intensity'

    fig, axes = plt.subplots(1, 3)
    
    for i , channel in enumerate(channels):
    
        hist, bins = np.histogram(image[:, :, i].flatten(),
                                  bins = bins,
                                  density = density)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        axes[i].bar(center, hist, align = 'center', width = width)
        axes[i].set_title(channel + ' channel')
    
    plt.show()

def sampled_sd(image, size):
    '''
    Computes the standard deviation of image intensities,
    sampled by subimages of given size.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.
    size : tuple
        Size of image samples.

    Returns
    -------
    output : numpy.ndarray
        array of standard deviations.

    '''
    h, w = image.shape
    m, n = size
    
    image_col = np.zeros(((h // m) * (w // n), m*n))

    for i in range(h // m):
        for j in range(w // n):
            image_col[i*(w // n) + j, :] = image[i*m:(i + 1)*m, j*n:(j + 1)*n].reshape(-1)
    
    output = np.std(image_col, axis = 1)
    
    return output

def gradient_image(image):
    '''
    Takes image, returns (grad_x, grad_y, grad_image, directions)
    computed using sobel kernels.

    Parameters
    ----------
    image : numpy.ndarray
        Input image (H, W, C) or (H, W).

    Returns
    -------
    tuple

    '''    
    gx = np.zeros_like(image) 
    gy = np.zeros_like(image)      
      
    wx = np.array([-1, -2, -1,
                   0, 0, 0,
                   1, 2, 1]).reshape(3, 3)
            
    wy = np.array([-1, 0, 1,
                   -2, 0, 2,
                   -1, 0, 1]).reshape(3, 3)    
    
    try:
        num_channels = image.shape[2]
        
        for i in range(num_channels):
            gx[:, :, i] = convolve2d(image[:, :, i], wx, mode = 'same')
            gy[:, :, i] = convolve2d(image[:, :, i], wy, mode = 'same')

        gx = np.max(gx, axis = 2)
        gy = np.max(gy, axis = 2)
    
    except IndexError:
        gx = convolve2d(image, wx, mode = 'same')
        gy = convolve2d(image, wy, mode = 'same')
    
    # add constant to avoid zero division
    epsilon = 1e-9
    
    return (gx,
            gy,
            np.sqrt(gx**2 + gy**2),
            (np.arctan(gy / (gx + epsilon)) * 180 / np.pi) % 180)

def HOG(grad_mag, grad_dir, num_bins = 9):
    '''
    Takes gradient magnitudes and directions,
    and returns bins and values for the HOG descriptor.

    Parameters
    ----------
    grad_mag : numpy.ndarray
        Matrix of gradient magnitudes.
    grad_dir : numpy.ndarray
        Matrix of gradient directions.
    num_bins : int, optional
        DESCRIPTION. The default is 9.

    Returns
    -------
    bins : numpy.ndarray
        Histogram bins.
    vals : numpy.ndarray
        Histogram values.

    '''
    bins = np.linspace(0, 180 - 180 / num_bins, num_bins)
    vals = np.zeros_like(bins)

    angles = grad_dir.flatten()
    magnitudes = grad_mag.flatten()
    
    
    for i in range(len(angles)):
        
        j = np.floor(angles[i]*num_bins/180).astype(int)
        
        vals[j] += magnitudes[i]*(
            angles[i] - bins[j])*num_bins/180
    
        k = num_bins % (j + 1)
        
        vals[k] += magnitudes[i]*(
            bins[j] + 180/num_bins - angles[i])*num_bins/180
        
    return bins, vals

def segment_image(image, subregion, threshold, batch_size=1000):
    '''
    Segments image based on pixels in subregion.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.
    subregion : numpy.ndarray
        Subregion of input image.
    threshold : float
        Segmentation threshold.
    batch_size : int, optional
        Computes distances in batches for memory handling.
        The default is 1000.

    Returns
    -------
    numpy.ndarray
        Segmented image.

    '''
    # image dimensions
    H, W, C = image.shape

    # mean and covariance of pixels in subregion
    a = np.mean(subregion, axis = (0, 1))    
    c = np.cov(np.vstack([subregion[..., i].flatten() for i in range(C)]))
    c_inv = np.linalg.inv(c)

    y = np.vstack([(image - a)[..., i].flatten() for i in range(C)])
    d = np.zeros(y.shape[-1])
    
    # compute distances in batches
    for i in range(np.ceil(y.shape[-1] / batch_size).astype(int)):
        lo = i*batch_size
        up = (i+1)*batch_size
        subset = y[:, lo:up]
        d[lo:up] = np.sqrt(np.diagonal(subset.transpose() @ c_inv @ subset))
        
    return (d < threshold).repeat(3).reshape(H, W, C) * image