import numpy as np
import tensorflow as tf

def gaussian_kernel(size=5, sigma=2.0):
    """
    size: int
    sigma: blur factor

    return: (normalized) Gaussian kernel size*size
    """
    x_points = np.arange(-(size - 1) // 2, (size - 1) // 2 + 1, 1)
    y_points = x_points[::-1]
    xs, ys = np.meshgrid(x_points, y_points)
    kernel = np.exp(-(xs ** 2 + ys ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    return kernel / kernel.sum()


def tf_gaussian_blyr(input, size=5, sigma=1.5):
    kernel = gaussian_kernel(size=size, sigma=sigma)
    kernel = kernel[:, :, np.newaxis, np.newaxis]  # height,width, channel_in, channel_out
    return tf.nn.conv2d(input, kernel, strides=[1, 1, 1, 1], padding='SAME')