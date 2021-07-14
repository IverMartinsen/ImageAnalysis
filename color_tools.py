# -*- coding: utf-8 -*-
"""
Functions for color images.

Created on Mon Jun 21 12:28:51 2021

@author: iverm
"""

import numpy as np
import matplotlib.pyplot as plt

def rgb_to_hsi(image):
    '''
    Transforms image from RGB space to HSI space.

    Parameters
    ----------
    image : PIL.Image
        Input RGB image.

    Returns
    -------
    numpy.ndarray
        Output HSI image.

    '''
    
    # normalize image    
    image = np.array(image, dtype = np.float32) / np.max(image)
    
    height, width, _ = image.shape

    R, G, B = [i.reshape(height, width) for i in np.split(image, 3, 2)]

    # small constant to avoid dividing by zero
    epsilon = 1e-3
    
    theta = (np.arccos(((R - G) + (R - B)) / 2) /
             (np.sqrt((R - G)**2 + (R - B)*(G - B)) + epsilon))
    
    # transformations
    H = (B <= G) * theta + (B > G) * (360 - theta)
    S = 1 - 3 * np.min(image, axis = 2) / (R + G + B)
    I = (R + G + B) / 3
    
    return np.dstack((H / np.max(H), S / np.max(S), I / np.max(I)))

def channel_plot(image, color_space = 'rgb'):
    '''
    Displays image along with channels in a 2-by-2 grid.

    Parameters
    ----------
    image : PIL.Image
        Input image.
    color_space : str, optional
        Color space channels to display. The default is 'rgb'.

    Returns
    -------
    None.

    '''
    
    # normalize image
    original = np.array(image, dtype = np.float32) / np.max(image)
    
    if color_space == 'rgb':
        image = original
        channel_names = 'Red', 'Green', 'Blue'
        
    elif color_space == 'hsi':
        image = rgb_to_hsi(image)
        channel_names = 'Hue', 'Saturation', 'Intensity'
    
    height, width, _ = image.shape
    
    # unpack channels
    channels = [i.reshape(height, width) for i in np.split(image, 3, 2)]
    
    # plot image
    fig, axes = plt.subplots(2, 2)
    axes = np.array(axes).flatten()
    
    axes[0].imshow(original)
    axes[0].axis('off')
    axes[0].set_title('Original')
        
    for i, channel in enumerate(channels):
        channel /= np.max(channel)
        axes[i + 1].imshow(channel, 'gray')
        axes[i + 1].axis('off')
        axes[i + 1].set_title(channel_names[i])
    
    plt.show()
    
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])