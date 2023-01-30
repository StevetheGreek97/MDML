import tensorflow as tf
import numpy as np

def xyz2rgb(traj):
    no = traj.shape[0]
    rgb = [np.array(tf.keras.preprocessing.image.array_to_img(traj[frame])) for frame in range(no)]
    return np.array(rgb)

if __name__ == "__main__":
    pass
