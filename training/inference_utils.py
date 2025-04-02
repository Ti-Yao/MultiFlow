# Standard library imports
import os
import sys
import math
import random
import json
import warnings
import glob
import re
from pathlib import Path
from itertools import chain

# Third-party libraries
import numpy as np
import pandas as pd
import pydicom
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
from scipy import ndimage, stats
from scipy.ndimage import zoom, center_of_mass
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
from sklearn.model_selection import train_test_split

# TensorFlow & Keras
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Add, Activation

# Neptune for experiment tracking
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

# External libraries
import skimage
import volumentations as V


random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

warnings.filterwarnings('ignore')



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
    'rpa': ['rpa', 'r pa', 'rt ','right'],
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


def description_one_hot(description):
    if description == '': # if description is empty
        label = ''
        one_hot_input = tf.one_hot(0, 6)[np.newaxis] 

    else:
        description = description.replace('_',' ').replace('.',' ').replace('x','').replace('  ',' ').split(' ')
        print(description)
        labels = is_token_a_substring_in_dictionary(data_dictionary, description) 
        if len(labels) == 0:
            label = 0
        else:
            labels = pd.Series(labels)
            if (labels == 'other').any():
                label = 'other'
            else:
                label = labels.value_counts().index[0]
    
    one_hot = vessels_dict[label] if label in vessels_dict.keys() else 0 # tunable input 
    one_hot_input = tf.one_hot(one_hot, 6)[np.newaxis] 
    return one_hot_input
    
def inference(model,mag_image, phase_image, patient, venc_df, vessel, series_description_df, type):
    vessel_index = vessels_dict[vessel]
    mag_image[mag_image<1e-10] = 0                
    max_val = np.max(phase_image)

    venc = venc_df.loc[(venc_df['patient'] == patient) & (venc_df['vessel'] == vessel)].venc.values[0]
    angles = phase2angle(phase_image, venc)
    mag_image = (mag_image - np.min(mag_image))/(np.max(mag_image))
    mag_image[mag_image>=1] = 1

    mag_image = skimage.exposure.equalize_adapthist(mag_image)
    complex_image = create_complex_image(mag_image, angles)
    real_image, imaginary_image = complex_image[...,0],complex_image[...,1]
    mag_image = normalise(mag_image)        
    imaginary_image = normalise(imaginary_image)        

    X = np.stack([mag_image, imaginary_image], -1).astype('float32')[np.newaxis]
    y = np.zeros((image_size, image_size, 32, 6), dtype='uint8')[np.newaxis] # dummy input

    cgm_input = tf.zeros((6))[np.newaxis] 

    if type == 'empty':
        description = ''
        one_hot_input = description_one_hot(description)    

    if type == 'actual':
        description = series_description_df.loc[patient,vessel].seriesdescription[0]
        one_hot_input = description_one_hot(description)    

    if type == 'random':
        array = np.arange(0, 6)
        array = np.delete(array, vessel_index)
        array = np.delete(array, 0)
        random_val = random.choice(array)
        one_hot_input =  tf.one_hot(random_val, 6)[np.newaxis] 

    if type == 'vanilla':
        y_pred = model.predict({'image_input':X, 'cgm_input': cgm_input,'mask_input': y})[-1][0]
        print(y_pred.shape)
        pred_label = vessels_dict_r[np.argmax(np.sum(y_pred, axis=(0, 1, 2))[1:]) + 1]

    else:
        print(one_hot_input)
        y_pred, probability = model.predict({'image_input':X, 'cgm_input': cgm_input,'one_hot_input':one_hot_input,'mask_input': y})
        pred_label = vessels_dict_r[np.argmax(probability)]
        y_pred = y_pred[-1][0]

    y_pred = get_one_hot(np.argmax(y_pred,axis = -1), 6).astype('uint8')
    y_pred = clean_channels(y_pred)

    return y_pred, pred_label



# def plot_gif(mag_image, true_mask, dice_val, vessel, y_pred, patient):
#     fig, axs = plt.subplots(1,2, figsize = (9,5))
#     fig.suptitle(f'Dice = {dice_val:.2f}')

#     frames = []
#     for i in range(mag_image.shape[-1]):
#         p1 = axs[0].imshow(mag_image[...,i],cmap = 'gray', vmin = np.min(mag_image), vmax = np.max(mag_image))
#         p2 = axs[1].imshow(mag_image[...,i],cmap = 'gray', vmin = np.min(mag_image), vmax = np.max(mag_image))
#         p4 = axs[1].imshow(true_mask[...,i],alpha = true_mask[...,i] * 0.7, cmap = colormaps[vessel])
#         text = axs[0].text(0,-5,f'Time = {i}')

#         artists = [p1, p2, p4, text]
#         for j, label in enumerate(list(colormaps.keys())[1:]):
#             cmap = colormaps[label]
#             if np.sum(y_pred[..., j+1]) > 0:
#                 artists.append(axs[0].imshow(y_pred[..., i, j+1],alpha = y_pred[..., i, j+1] * 0.7, cmap=cmap))
#         frames.append(artists)
#         legend_patches = [mpatches.Patch(color=plt.cm.get_cmap(colormaps[label])(0.5), label=label) for label in colormaps.keys()]
#         fig.legend(handles=legend_patches, loc='lower center', ncol=5, fontsize='large', bbox_to_anchor=(0.5, 0))
#     fig.tight_layout()
#     plt.subplots_adjust(hspace=0.5, bottom=0.1)
#     ani = animation.ArtistAnimation(fig, frames)


#     Path(f'results/{model_name}/{vessel}').mkdir(parents=True, exist_ok=True)
#     ani.save(f'results/{model_name}/{vessel}/{patient}.gif', fps=mag_image.shape[0]/2)
# #             run[f'results/{cohort}/{patient}/{vessel}'].upload(f'results/{model_name}/{vessel}/{patient}.gif')
#     plt.close()

def single_dice(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    if im1.sum() + im2.sum() == 0:
        return 1.0  # If both arrays are empty, they are identical
    

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())

def get_model(inference_model_name):
    # model =  tf.keras.models.load_model(f'models/{inference_model_name}.h5', compile = False)
    model_path = f'models/{inference_model_name}.h5'
    model =  tf.keras.models.load_model(model_path, compile = False) # segmentation part

    # Identify the classification layer
    classification_layer = None
    for layer in model.layers:
        if layer.name == "tf.nn.softmax":  
            classification_layer = layer.output # classification part
            break

    # Create a single model that outputs both segmentation and classification
    model = tf.keras.Model(inputs=model.inputs, outputs=[model.output, classification_layer])
    return model




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

def create_complex_image(magnitude, phase): # magnitude is a real number tensor; phase is a tensor of radiant angles
#     if np.max(phase) > (np.pi+1e-2) or np.min(phase) < -(np.pi+1e-2):
#         print('Not right about phase')
    # Calculate real and imaginary parts
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
