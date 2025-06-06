import json
import math
import os
import random
import re
import sys
import warnings
from glob import glob
from itertools import chain
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import pydicom
import seaborn as sns
import skimage
from scipy import ndimage, stats
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson
from scipy.ndimage import center_of_mass, zoom
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation, Add, Layer
from tqdm import tqdm
from keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, LeakyReLU, SpatialDropout3D,
    Activation, GlobalMaxPooling3D, concatenate, Reshape, Lambda
)
from tensorflow.keras.optimizers import Adam
import volumentations as V



skip = 5
red = '#FF4E32'
blue = '#57C7FF'
green = '#64BB12' #'#A2F55A'
purple = '#CD68FF'
yellow = '#FF771C'
vessels = ['lpa','rpa','ao','svc','ivc']

vessels_dict = {'lpa':1, 'rpa':2, 'ao':3, 'svc':4, 'ivc':5}
vessels_dict_r = {v:k for k,v in vessels_dict.items()}
image_size = 128
image_frames = 32
min_timesteps = 10
max_timesteps = 65

colormaps = {
    'lpa': ListedColormap([red]),
    'rpa': ListedColormap([blue]),
    'ao': ListedColormap([green]),
    'svc': ListedColormap([purple]),
    'ivc': ListedColormap([yellow])
}
colormaps = {vessel:colormaps[vessel] for vessel in vessels}


data_dictionary = {
    'lpa': ['lpa', 'l pa', 'lt ', 'left'],
    'rpa': ['rpa', 'r pa', 'rt ', 'right'],
    'ivc': ['ivc', 'dao', 'inferior', 'fontan', 'font','inf '],
    'svc': ['svc', 'rsvc', 'lsvc', 'sup', 'superior', 'glenn', 'vfc', 'bdg'],
    'ao': ['aorta', 'ao', 'aor', 'asc', 'aso', 'neo', 'aa', 'aao', 'native', 'aov', 'av ', 'qs', 'stj', 'dks'],
    'other': ['pv', 'avv', 'vein', 'lpv', 'rpv']
}


description_match_dict = {'lpa':['lpa'],
                          'rpa':['rpa'],
                          'ivc':['ivc'],
                          'svc':['svc'],
                          'ao':['asc','aa','aao','ao','neo','stj','dks']}



def convert_time_to_minutes(time_str):
    '''
    Converts a time string in the format HHMMSS.FFFFFF into minutes.
    '''
    # Parse the time components
    time_str = time_str.split('.')[0]
    hours = int(time_str[:2])
    minutes = int(time_str[2:4])
    seconds = int(time_str[4:6])
    
    # Calculate total time in minutes
    total_minutes = hours * 60 + minutes + seconds / 60 
    return round(total_minutes)

def is_token_a_substring_in_dictionary(data_dictionary, description):
    """
    Check if any substring in the data dictionary lists is a part of any string in the description.
    """
    matches = []
    for key, value_list in data_dictionary.items():
        for substring in value_list:
            for string in description:
                if substring in string:
                    matches.append(key)
    return matches



def assign_group(group):
    sum_values = group['image'].apply(np.sum)
    max_index = sum_values.idxmax()
    min_index = sum_values.idxmin()
    group['group'] = ['mag' if i == max_index else 'diff' for i in group.index]
    return group

def phase2angle(value, venc, to_range=(-np.pi, np.pi)):
    from_min = -venc
    from_max = venc
    to_min, to_max = to_range
    mapped_value = (value - from_min) * (to_max - to_min) / (from_max - from_min) + to_min
    return mapped_value

def create_complex_image(magnitude, phase):
    real_part = magnitude * np.cos(phase)
    imag_part = magnitude * np.sin(phase)
    
    # Create complex image
    complex_image = np.stack((real_part, imag_part), axis=-1)
    
    return complex_image


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def calculate_curve(mask, phase_image, vessel):
    ps = 1 if mask.shape[0] == 256 else 2
    pixel_area = ps **2 / 100  # convert mm2 to cm2
    phase = mask * phase_image
    v_curve = np.sum(np.sum(phase*pixel_area ,0),0) # cm3/s
    if np.mean(v_curve) < 0:
        v_curve = -v_curve
    return v_curve

def interpolate_curve(curve, phase_vessel_rr):
    x_original = np.linspace(0, round(phase_vessel_rr), len(curve))
    y_original = curve
    cs = CubicSpline(x_original, y_original)
def get_volumentation(image_size, frames, vessel):
    transforms = [V.RandomBrightnessContrast(p  = 0.5), 
                  V.Flip(axis = 1, p = 0.4),
                  V.Rotate((0, 0), (0, 0), (-45, 45), p=0.5),
                  ]
    crop_val = random.randint(25,50)
    pad_factor = random.randint(0,20)

    if random.random()< 0.5:
        transforms.append(V.PadIfNeeded(shape = (image_size + pad_factor,image_size + pad_factor, frames), p =0.3))
    else:    
        transforms.append(V.CenterCrop(shape = (image_size- crop_val,image_size - crop_val, frames), p = 0.3))
        
    transforms.extend([
    V.Resize(shape = (image_size,image_size, frames), p = 1)
    ])
    return V.Compose(transforms, p=1.0)
    x_new = np.arange(0, round(phase_vessel_rr))
    return cs(x_new)

def format_func(value, tick_number):
    return value / 1000



def normalise(image):
    if np.max(image) != 0:
        norm = (image - np.min(image)) / np.max(image)
    return norm



def remove_non_propagating_components(masks):
    H, W, T = masks.shape
    
    # Label connected components in the 3D mask (H, W, T)
    labeled_mask, num_components = ndimage.label(masks)
    
    # Create an empty mask to store the valid components
    final_mask = np.zeros_like(masks)
    
    # Iterate over all connected components
    for component_id in range(1, num_components + 1):
        # Find the time points where the component exists
        component_coords = np.argwhere(labeled_mask == component_id)
        
        # Extract the time points for this component
        time_points = np.unique(component_coords[:, 2])
        
        # Check if the component spans all time points
        if len(time_points) == T:
            # If it does, keep the component in the final mask
            final_mask[labeled_mask == component_id] = 1
    
    return final_mask

def keep_largest_component_per_time_point(masks):
    H, W, T = masks.shape
    
    # Create a new mask to store the result
    final_mask = np.zeros_like(masks)
    
    for t in range(T):
        # Extract the 2D mask at the current time point
        mask_at_t = masks[:, :, t]
        
        # Label connected components for this 2D slice
        labeled_mask, num_components = ndimage.label(mask_at_t)
        
        if num_components > 0:
            # Find the size of each connected component
            component_sizes = np.bincount(labeled_mask.ravel())
            
            # Ignore the background component (component 0)
            component_sizes[0] = 0
            
            # Find the largest component (largest size)
            largest_component_id = component_sizes.argmax()
            
            # Keep only the largest component in the final mask
            final_mask[:, :, t] = (labeled_mask == largest_component_id).astype(int)
    
    return final_mask


def calculate_segmentation_quality_metric(mask):
    labeled_mask, num_features = ndimage.label(mask)
    components = [(labeled_mask == i).astype(np.uint8) for i in range(1, num_features + 1)]
    mask_quality_metrics = []
    for component in components:
        differences = []
        areas = []
        for i in range(component.shape[-1]):
            difference = math.dist(np.array(center_of_mass(component))[:2], np.array(center_of_mass(component[...,i])))
            areas.append(component[...,i].sum())
            differences.append(difference)
        mask_quality_metric = 1 - np.max(differences)/(np.min(areas))
        mask_quality_metrics.append(mask_quality_metric)
    segmentation_quality_metric = np.mean(mask_quality_metrics)
    return segmentation_quality_metric

def keep_largest_n_components(binary_mask, n):
    """
    Keep the n largest connected components in a binary mask.
    
    Parameters:
    - binary_mask: numpy array of shape (M, N) representing the binary mask.
    - n: Number of largest connected components to keep.
    
    Returns:
    - new_mask: numpy array of the same shape as binary_mask with only the n largest components.
    """
    # Label the connected components
    labeled_array, num_features = ndimage.label(binary_mask)

    # If there are no components, return an empty mask
    if num_features == 0:
        return np.zeros_like(binary_mask, dtype=np.uint8)  # No components, return empty mask

    # If there's only one component, return it as is
    if num_features == 1 and n==1:
        return binary_mask

    # If there are exactly two components, return both of them without removing any
    if num_features == 2 and n==2:
        return binary_mask

    # Calculate the size of each component (component 0 is the background)
    component_sizes = np.bincount(labeled_array.ravel())

    # Get the indices of the n largest components (ignoring the background, index 0)
    largest_components_indices = np.argsort(component_sizes[1:])[-n:] + 1  # Offset by 1 to ignore background

    # Create a new mask with only the largest components
    new_mask = np.isin(labeled_array, largest_components_indices).astype(np.uint8)
    
    return new_mask

def check_double_aorta(mask):
    dilated_mask = ndimage.binary_dilation(mask, iterations = 10)
    dilated_labels, num_components_after = ndimage.label(dilated_mask)
    if num_components_after == 1:
        return True  # Double aorta
    else:
        return False  # No double aorta

def clean_mask(mask, vessel):
    mask= remove_non_propagating_components(mask)
    double_aorta = False
    if vessel == 'ao':
        double_aorta = check_double_aorta(mask)
    if vessel == 'svc' or (vessel == 'ao' and double_aorta):
        mask = keep_largest_n_components(mask, n=2)
    else:
        mask = keep_largest_component_per_time_point(mask)
        mask = keep_largest_n_components(mask, n=1)
    mask= remove_non_propagating_components(mask)
    return mask


def clean_channels(y_pred):
    for vessel, channel_index in vessels_dict.items():
        pred_mask = y_pred[..., channel_index]
        pred_mask = clean_mask(pred_mask, vessel)
        y_pred[..., channel_index] = pred_mask
    return y_pred


def find_substring_matches(substrings, strings):
    matches = []
    for sub in substrings:
        for s in strings:
            if sub in s:
                matches.append(s)
    return matches
