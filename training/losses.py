
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy.ndimage import distance_transform_edt as distance
from tensorflow import Tensor
from typing import List

# ========================= #
# Dice loss and variants
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

# ========================= #
# Tversky loss and variants

def tversky(y_true, y_pred, alpha=0.5, const=K.epsilon()):
    '''
    Tversky Loss.
    
    tversky(y_true, y_pred, alpha=0.5, const=K.epsilon())
    
    ----------
    Hashemi, S.R., Salehi, S.S.M., Erdogmus, D., Prabhu, S.P., Warfield, S.K. and Gholipour, A., 2018. 
    Tversky as a loss function for highly unbalanced image segmentation using 3d fully convolutional deep networks. 
    arXiv preprint arXiv:1803.11078.
    
    Input
    ----------
        alpha: tunable parameter within [0, 1]. Alpha handles imbalance classification cases.
        const: a constant that smooths the loss gradient and reduces numerical instabilities.
        
    '''
    # tf tensor casting
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    
    # <--- squeeze-out length-1 dimensions.
    y_pred = tf.squeeze(y_pred[...,1:])
    y_true = tf.squeeze(y_true[...,1:])
    
    loss_val = 1 - tversky_coef(y_true, y_pred, alpha=alpha, const=const)
    
    return loss_val

def tversky_coef(y_true, y_pred, alpha=0.5, const=K.epsilon()):
    '''
    Weighted Sørensen–Dice coefficient.
    
    Input
    ----------
        y_true, y_pred: predicted outputs and targets.
        const: a constant that smooths the loss gradient and reduces numerical instabilities.
        
    '''
    
    # flatten 2-d tensors
    y_true_pos = tf.reshape(y_true, [-1])
    y_pred_pos = tf.reshape(y_pred, [-1])
    
    # get true pos (TP), false neg (FN), false pos (FP).
    true_pos  = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1-y_pred_pos))
    false_pos = tf.reduce_sum((1-y_true_pos) * y_pred_pos)
    
    # TP/(TP + a*FN + b*FP); a+b = 1
    coef_val = (true_pos + const)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + const)
    
    return coef_val

def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [1,2,3]
    # Two dimensional
    elif len(shape) == 4 : return [1,2]
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')

# def focal_tversky_loss(alpha, gamma, smooth=0.000001):
#     """A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation
#     Link: https://arxiv.org/abs/1810.07842
#     Parameters
#     ----------
#     gamma : float, optional
#         focal parameter controls degree of down-weighting of easy examples, by default 0.75
#     """
#     def loss_function(y_true, y_pred):
#         # Clip values to prevent division by zero error
#         epsilon = K.epsilon()
#         y_pred = K.clip(y_pred, epsilon, 1. - epsilon) 
#         axis = identify_axis(y_true.get_shape())
#         # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
#         tp = K.sum(y_true * y_pred, axis=axis)
#         fn = K.sum(y_true * (1-y_pred), axis=axis)
#         fp = K.sum((1-y_true) * y_pred, axis=axis)
#         tversky_class = (tp + smooth)/(tp + alpha*fn + (1-alpha)*fp + smooth)
#         # Average class scores
#         focal_tversky_loss = K.mean(K.pow((1-tversky_class), gamma))

#         return focal_tversky_loss

#     return loss_function

import tensorflow as tf

def categorical_crossentropy(y_true, y_pred):
    # Clip the predictions to prevent log(0) errors
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    
    # Calculate cross entropy
    crossentropy = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
    
    return tf.reduce_mean(crossentropy)

def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
    """
    Focal Tversky loss for multi-class 3D segmentation.

    Args:
    y_true: tensor of shape [B, D, H, W, C]
    y_pred: tensor of shape [B, D, H, W, C]
    alpha: controls the penalty for false positives
    beta: controls the penalty for false negatives
    gamma: focal parameter to down-weight easy examples
    smooth: smoothing constant to avoid division by zero

    Returns:
    loss: computed Focal Tversky loss
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, smooth, 1.0 - smooth)  # Clipping to avoid log(0)
    
    num_classes = 6
    loss = 0.0
    
    for c in range(num_classes):
        y_true_c = y_true[..., c]
        y_pred_c = y_pred[..., c]
        
        true_pos = tf.reduce_sum(y_true_c * y_pred_c)
        false_neg = tf.reduce_sum(y_true_c * (1 - y_pred_c))
        false_pos = tf.reduce_sum((1 - y_true_c) * y_pred_c)
        
        tversky_index = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
        loss_c = tf.pow((1 - tversky_index), gamma)
        loss += loss_c
    
    loss /= tf.cast(num_classes, tf.float32)  # Averaging over all classes
    return loss



# ========================= #
# MS-SSIM

def ms_ssim(y_true, y_pred, **kwargs):
    """
    Multiscale structural similarity (MS-SSIM) loss.
    
    ms_ssim(y_true, y_pred, **tf_ssim_kw)
    
    ----------
    Wang, Z., Simoncelli, E.P. and Bovik, A.C., 2003, November. Multiscale structural similarity for image quality assessment. 
    In The Thrity-Seventh Asilomar Conference on Signals, Systems & Computers, 2003 (Vol. 2, pp. 1398-1402). Ieee.
    
    ----------
    Input
        kwargs: keywords of `tf.image.ssim_multiscale`
                https://www.tensorflow.org/api_docs/python/tf/image/ssim_multiscale
                
        *Issues of `tf.image.ssim_multiscale`refers to:
                https://stackoverflow.com/questions/57127626/error-in-calculation-of-inbuilt-ms-ssim-function-in-tensorflow
    
    """
    
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)
    
    tf_ms_ssim = tf.image.ssim_multiscale(y_true, y_pred, **kwargs)
        
    return 1 - tf_ms_ssim

# ======================== #

def iou_box_coef(y_true, y_pred, mode='giou', dtype=tf.float32):
    
    """
    Inersection over Union (IoU) and generalized IoU coefficients for bounding boxes.
    
    iou_box_coef(y_true, y_pred, mode='giou', dtype=tf.float32)
    
    ----------
    Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I. and Savarese, S., 2019. 
    Generalized intersection over union: A metric and a loss for bounding box regression. 
    In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 658-666).
    
    ----------
    Input
        y_true: the target bounding box. 
        y_pred: the predicted bounding box.
        
        Elements of a bounding box should be organized as: [y_min, x_min, y_max, x_max].

        mode: 'iou' for IoU coeff (i.e., Jaccard index);
              'giou' for generalized IoU coeff.
        
        dtype: the data type of input tensors.
               Default is tf.float32.

    """
    
    zero = tf.convert_to_tensor(0.0, dtype)
    
    # subtrack bounding box coords
    ymin_true, xmin_true, ymax_true, xmax_true = tf.unstack(y_true, 4, axis=-1)
    ymin_pred, xmin_pred, ymax_pred, xmax_pred = tf.unstack(y_pred, 4, axis=-1)
    
    # true area
    w_true = tf.maximum(zero, xmax_true - xmin_true)
    h_true = tf.maximum(zero, ymax_true - ymin_true)
    area_true = w_true * h_true
    
    # pred area
    w_pred = tf.maximum(zero, xmax_pred - xmin_pred)
    h_pred = tf.maximum(zero, ymax_pred - ymin_pred)
    area_pred = w_pred * h_pred
    
    # intersections
    intersect_ymin = tf.maximum(ymin_true, ymin_pred)
    intersect_xmin = tf.maximum(xmin_true, xmin_pred)
    intersect_ymax = tf.minimum(ymax_true, ymax_pred)
    intersect_xmax = tf.minimum(xmax_true, xmax_pred)
    
    w_intersect = tf.maximum(zero, intersect_xmax - intersect_xmin)
    h_intersect = tf.maximum(zero, intersect_ymax - intersect_ymin)
    area_intersect = w_intersect * h_intersect
    
    # IoU
    area_union = area_true + area_pred - area_intersect
    iou = tf.math.divide_no_nan(area_intersect, area_union)
    
    if mode == "iou":
        
        return iou
    
    else:
        
        # encolsed coords
        enclose_ymin = tf.minimum(ymin_true, ymin_pred)
        enclose_xmin = tf.minimum(xmin_true, xmin_pred)
        enclose_ymax = tf.maximum(ymax_true, ymax_pred)
        enclose_xmax = tf.maximum(xmax_true, xmax_pred)
        
        # enclosed area
        w_enclose = tf.maximum(zero, enclose_xmax - enclose_xmin)
        h_enclose = tf.maximum(zero, enclose_ymax - enclose_ymin)
        area_enclose = w_enclose * h_enclose
        
        # generalized IoU
        giou = iou - tf.math.divide_no_nan((area_enclose - area_union), area_enclose)

        return giou

def iou_box(y_true, y_pred, mode='giou', dtype=tf.float32):
    """
    Inersection over Union (IoU) and generalized IoU losses for bounding boxes. 
    
    iou_box(y_true, y_pred, mode='giou', dtype=tf.float32)
    
    ----------
    Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I. and Savarese, S., 2019. 
    Generalized intersection over union: A metric and a loss for bounding box regression. 
    In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 658-666).
    
    ----------
    Input
        y_true: the target bounding box. 
        y_pred: the predicted bounding box.
        
        Elements of a bounding box should be organized as: [y_min, x_min, y_max, x_max].

        mode: 'iou' for IoU coeff (i.e., Jaccard index);
              'giou' for generalized IoU coeff.
        
        dtype: the data type of input tensors.
               Default is tf.float32.
        
    """
    
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred = tf.cast(y_pred, dtype)
    
    y_true = tf.cast(y_true, dtype)
    
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)

    return 1 - iou_box_coef(y_true, y_pred, mode=mode, dtype=dtype)


def iou_seg(y_true, y_pred, dtype=tf.float32):
    """
    Inersection over Union (IoU) loss for segmentation maps. 
    
    iou_seg(y_true, y_pred, dtype=tf.float32)
    
    ----------
    Rahman, M.A. and Wang, Y., 2016, December. Optimizing intersection-over-union in deep neural networks for 
    image segmentation. In International symposium on visual computing (pp. 234-244). Springer, Cham.
    
    ----------
    Input
        y_true: segmentation targets, c.f. `tf.keras.losses.categorical_crossentropy`
        y_pred: segmentation predictions.
        
        dtype: the data type of input tensors.
               Default is tf.float32.
        
    """

    # tf tensor casting
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred = tf.cast(y_pred, dtype)
    y_true = tf.cast(y_true, y_pred.dtype)

    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)
    
    y_true_pos = tf.reshape(y_true, [-1])
    y_pred_pos = tf.reshape(y_pred, [-1])

    area_intersect = tf.reduce_sum(tf.multiply(y_true_pos, y_pred_pos))
    
    area_true = tf.reduce_sum(y_true_pos)
    area_pred = tf.reduce_sum(y_pred_pos)
    area_union = area_true + area_pred - area_intersect
    
    return 1-tf.math.divide_no_nan(area_intersect, area_union)

# ========================= #
# Semi-hard triplet

def triplet_1d(y_true, y_pred, N, margin=5.0):
    
    '''
    (Experimental)
    Semi-hard triplet loss with one-dimensional vectors of anchor, positive, and negative.
    
    triplet_1d(y_true, y_pred, N, margin=5.0)
    
    Input
    ----------
        y_true: a dummy input, not used within this function. Appeared as a requirment of tf.tf.keras.loss function format.
        y_pred: a single pass of triplet training, with `shape=(batch_num, 3*embeded_vector_size)`.
                i.e., `y_pred` is the ordered and concatenated anchor, positive, and negative embeddings.
        N: Size (dimensions) of embedded vectors
        margin: a positive number that prevents negative loss.
        
    '''
    
    # anchor sample pair separations.
    Embd_anchor = y_pred[:, 0:N]
    Embd_pos = y_pred[:, N:2*N]
    Embd_neg = y_pred[:, 2*N:]
    
    # squared distance measures
    d_pos = tf.reduce_sum(tf.square(Embd_anchor - Embd_pos), 1)
    d_neg = tf.reduce_sum(tf.square(Embd_anchor - Embd_neg), 1)
    loss_val = tf.maximum(0., margin + d_pos - d_neg)
    loss_val = tf.reduce_mean(loss_val)
    
    return loss_val


# import cv2 as cv
# import numpy as np


# def calc_dist_map(seg):
#     res = np.zeros_like(seg)
#     posmask = seg.astype(np.bool)

#     if posmask.any():
#         negmask = ~posmask
#         res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

#     return res


# def calc_dist_map_batch(y_true):
#     y_true_numpy = y_true.numpy()
#     return np.array([calc_dist_map(y)
#                      for y in y_true_numpy]).astype(np.float32)


# def boundary_loss(y_true, y_pred):
#     y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
#                                      inp=[y_true],
#                                      Tout=tf.float32)
#     multipled = y_pred * y_true_dist_map
#     return K.mean(multipled)


def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res

def calc_dist_map_batch(y_true):
    y_true_numpy = y_true.numpy()
    return np.array([calc_dist_map(y)
                     for y in y_true_numpy]).astype(np.float32)


# @tf.function
def boundary_loss(y_true, y_pred):
    y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
                                     inp=[y_true],
                                     Tout=tf.float32)
    multipled = y_pred * y_true_dist_map
    return K.mean(multipled)

def multiclass_boundary_loss(y_true, y_pred, num_classes = 6, ignore_background=True):
    total_loss = 0.0
    class_count = 0.0
    
    start_class = 1 if ignore_background else 0
    
    for c in range(start_class, num_classes):
        y_true_c = y_true[:, :, :, c] 
        y_pred_c = y_pred[:, :, :, c]  # Assuming channels last format
        
        if tf.reduce_sum(y_true_c) > 0:  # Proceed only if there are true pixels for the class
            class_loss = boundary_loss(y_true_c, y_pred_c)
            total_loss += class_loss
            class_count += 1
    
    if class_count > 0:
        return total_loss / class_count
    else:
        return tf.constant(0.0)


def simplex(tensor: tf.Tensor) -> bool:
    return tf.reduce_all(tf.equal(tf.reduce_sum(tensor, axis=-1), 1))

class GeneralizedDice:
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
        assert simplex(probs) and simplex(target)

        pc = tf.cast(tf.gather(probs, self.idc, axis=-1), tf.float32)
        tc = tf.cast(tf.gather(target, self.idc, axis=-1), tf.float32)

        w = 1 / (tf.pow(tf.reduce_sum(tc, axis=[1, 2]) + 1e-10, 2))
        intersection = w * tf.reduce_sum(pc * tc, axis=[1, 2])
        union = w * (tf.reduce_sum(pc, axis=[1, 2]) + tf.reduce_sum(tc, axis=[1, 2]))

        divided = 1 - 2 * (tf.reduce_sum(intersection, axis=-1) + 1e-10) / (tf.reduce_sum(union, axis=-1) + 1e-10)

        loss = tf.reduce_mean(divided)

        return loss


def contour(x):
    '''
    Differentiable approximation of contour extraction
    '''
    min_pool_x = -tf.nn.max_pool2d(-x, (3, 3), strides=(1, 1), padding='SAME')
    max_min_pool_x = tf.nn.max_pool2d(min_pool_x, (3, 3), strides=(1, 1), padding='SAME')
    contour = tf.nn.relu(max_min_pool_x - min_pool_x)
    return contour



# class ContourLoss(tf.tf.keras.losses.Loss):
#     '''
#     inputs shape  (batch, height, width, channel).
#     Calculate the contour loss
#     Because pred and target at moment of loss calculation will be TensorFlow tensors,
#     it is preferable to calculate target_skeleton on the step of batch forming,
#     when it will be in numpy array format by means of opencv
#     '''
#     def __init__(self, **kwargs):
#         super().__init__()
#         # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
#         self.idc: List[int] = kwargs["idc"]
#         self.name = kwargs.get("name", "ContourLoss")

#     def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
#         pc = tf.cast(tf.gather(y_pred, indices=self.idc, axis=-1), tf.float32)
#         tc = tf.cast(tf.gather(y_true, indices=self.idc, axis=-1), tf.float32)

#         b, w, h, _ = pc.shape
#         cl_pred = tf.reduce_sum(contour(pc), axis=(1, 2)) #/tf.reduce_sum(pc)
#         target_contour = tf.reduce_sum(contour(tc), axis=(1, 2))#/tf.reduce_sum(tc)
#         big_pen = tf.square(cl_pred - target_contour)
#         contour_loss = big_pen / (w * h)
    
#         return tf.reduce_mean(contour_loss, axis=0) * 5 * 100
    
# class IsoLoss(tf.tf.keras.losses.Loss):
#     '''
#     inputs shape  (batch, height, width, channel).
#     Calculate the contour loss
#     Because pred and target at moment of loss calculation will be TensorFlow tensors,
#     it is preferable to calculate target_skeleton on the step of batch forming,
#     when it will be in numpy array format by means of opencv
#     '''
#     def __init__(self, **kwargs):
#         super().__init__()
#         # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
#         self.idc: List[int] = kwargs["idc"]
#         self.name = kwargs.get("name", "Iso")

#     def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
#         y_pred = tf.cast(tf.gather(y_pred, indices=self.idc, axis=-1), tf.float32)

#         pc = contour(y_pred)
        
#         b, w, h, _ = pc.shape
        

#         pP = tf.reduce_sum(pc, axis=(1, 2)) + 1e-10
#         pA = tf.reduce_sum(y_pred, axis=(1, 2))
#         p_iso = (4 * np.pi * pA) / tf.square(pP)
#         loss =  1 - tf.reduce_max(p_iso, axis=0)
#         loss = tf.maximum(0., loss)
#         return loss
    
    
def hybrid_loss(losses, lambdas):
    def loss(y_true, y_pred):
        loss = 0
        if 'focal' in losses:
            index = losses.index('focal')
            lamb = lambdas[index]
            focal_tversky_fn = focal_tversky_loss(alpha=0.7, gamma=4/3)
            loss_focal = focal_tversky_fn(y_true, y_pred)
            loss += loss_focal * lamb
#         if 'cce' in losses:
#             index = losses.index('cce')
#             lamb = lambdas[index]
#             cce = tf.tf.keras.losses.CategoricalCrossentropy()
#             loss_cce = cce(y_true, y_pred)#.numpy()
#             loss_dice = dice(y_true, y_pred)
#             loss += loss_cce  * lamb + loss_dice* 1e-10
        if 'boundary' in losses:
            index = losses.index('boundary')
            lamb = lambdas[index]
            loss_boundary = multiclass_boundary_loss(y_true, y_pred)
            loss += loss_boundary * lamb
        if 'perimeter' in losses:
            index = losses.index('perimeter')
            lamb = lambdas[index]
            perimeter_fn = ContourLoss(idc=[1,2,3,4,5])
            loss_perimeter = perimeter_fn(y_true, y_pred)
            loss += loss_perimeter * lamb      
        if 'iso' in losses:
            index = losses.index('iso')
            lamb = lambdas[index]
            iso_fn = IsoLoss(idc=[1,2,3,4,5])
            loss_iso = iso_fn(y_true, y_pred)
            loss += loss_iso * lamb      
        return loss
    return loss



def iou(y_true, y_pred, smooth=1):
    """
    Calculate intersection over union (IoU) between images.
    Input shape should be Batch x Height x Width x #Classes (BxHxWxN).
    Using Mean as reduction type for batch values.
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3])
    union = union - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def iou_loss(y_true, y_pred):
    """
    Jaccard / IoU loss
    """
    return 1 - iou(y_true, y_pred)


def focal_loss(y_true, y_pred):
    """
    Focal loss
    """
    gamma = 2.
    alpha = 4.
    epsilon = 1.e-9

    y_true_c = tf.convert_to_tensor(y_true, tf.float32)
    y_pred_c = tf.convert_to_tensor(y_pred, tf.float32)

    model_out = tf.add(y_pred_c, epsilon)
    ce = tf.multiply(y_true_c, -tf.math.log(model_out))
    weight = tf.multiply(y_true_c, tf.pow(
        tf.subtract(1., model_out), gamma)
                         )
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=-1)
    return tf.reduce_mean(reduced_fl)


def ssim_loss(y_true, y_pred):
    """
    Structural Similarity Index loss.
    Input shape should be Batch x Height x Width x #Classes (BxHxWxN).
    Using Mean as reduction type for batch values.
    """
    ssim_value = tf.image.ssim(y_true, y_pred, max_val=1)
    return K.mean(1 - ssim_value, axis=0)


def dice_coef(y_true, y_pred, smooth=1.e-9):
    """
    Calculate dice coefficient.
    Input shape should be Batch x Height x Width x #Classes (BxHxWxN).
    Using Mean as reduction type for batch values.
    """
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def unet3p_hybrid_loss(y_true, y_pred):
    """
    Hybrid loss proposed in
    UNET 3+ (https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf)
    Hybrid loss for segmentation in three-level hierarchy – pixel,
    patch and map-level, which is able to capture both large-scale
    and fine structures with clear boundaries.
    """
    f_loss = focal_loss(y_true, y_pred)
#     jacard_loss = iou_loss(y_true, y_pred)
    ms_ssim = tfmri.metrics.SSIMMultiscale(max_val = 1)
    ms_ssim_loss = ms_ssim(y_true, y_pred)
    
    focal_tversky_fn = focal_tversky_loss(alpha=0.7, gamma=4/3)
    loss_focal = focal_tversky_fn(y_true, y_pred)
    return f_loss + ms_ssim_loss + loss_focal