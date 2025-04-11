import tensorflow as tf
import numpy as np
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


def categorical_crossentropy(y_true, y_pred):
    # Clip the predictions to prevent log(0) errors
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    
    # Calculate cross entropy
    crossentropy = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
    
    return tf.reduce_mean(crossentropy)