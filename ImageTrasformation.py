"""
Conversion of XYZ trajectory to RGB images.

This module provides a function `xyz2rgb` that converts a 3D XYZ trajectory to
a sequence of RGB images.
"""

import tensorflow as tf
import numpy as np

def xyz2rgb(traj):
    """Convert XYZ trajectory to RGB images."""
    no_frames = traj.shape[0]
    rgb = [np.array(tf.keras.preprocessing.image.array_to_img(traj[frame]))
           for frame in range(no_frames)]
    return np.array(rgb)

if __name__ == "__main__":
    pass
