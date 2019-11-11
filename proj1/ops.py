# uncompyle6 version 3.5.0
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.4 (default, Oct  4 2019, 06:57:26) 
# [GCC 9.2.0]
# Embedded file name: /home/eyek/Desktop/cv-project/proj1/ops.py
# Size of source mod 2**32: 3378 bytes
import numpy as np, cv2

def edge_pad(orig):
    """
    an impl of edge(orig, (1,1), "edge") which is neither intuitive nor elegant,
    but what can I do?
    """
    inter = np.ndarray((orig.shape[0] + 2, orig.shape[1] + 2))
    inter[1:-1, 1:-1] = orig
    inter[0] = orig[0]
    inter[-1] = orig[(-1)]
    inter[:, 0] = orig[:, 0]
    inter[:, -1] = orig[:, -1]
    inter[(0, 0)] = orig[(0, 0)]
    inter[(0, -1)] = orig[(0, -1)]
    inter[(-1, 0)] = orig[(-1, 0)]
    inter[(-1, -1)] = orig[(-1, -1)]
    return inter


def corner_edge_pad(orig, width):
    inter = np.ndarray((orig.shape[0] + width, orig.shape[1] + width))
    inter[width:, width:] = orig
    inter[0:width, width:] = orig[0]
    inter[width:, 0:width] = orig[:, 0].repeat(width).reshape((orig.shape[0], width))
    inter[0:width, 0:width] = orig[(0, 0)]
    return inter


def conv2d(orig, kern, kernel_size=None):
    assert orig.ndim == 2 and kern.ndim == 2
    assert kern.shape[0] == kern.shape[1]
    if kernel_size == None:
        kernel_size = kern.shape[0]
    print('conv...')
    inter = corner_edge_pad(orig, kernel_size - 1)
    ret = np.ndarray(orig.shape)
    for i in range(orig.shape[0]):
        for j in range(orig.shape[1]):
            ret[(i, j)] = np.sum(np.multiply(kern, inter[i:i + kernel_size, j:j + kernel_size]))

    return ret


def gradient(gx, gy):
    return np.sqrt(gx ** 2 + gy ** 2)


def roberts(orig):
    roberts_y = np.array([[0, -1], [1, 0]])
    roberts_x = np.array([[-1, 0], [0, 1]])
    return gradient(conv2d(orig, roberts_x), conv2d(orig, roberts_y))


def prewitt(orig):
    prewitt_y = np.array([[-1, 0, 1]] * 3)
    prewitt_x = prewitt_y.T.copy()
    return gradient(conv2d(orig, prewitt_x), conv2d(orig, prewitt_y))


def sobel(orig):
    sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_x = sobel_y.T.copy()
    return gradient(conv2d(orig, sobel_x), conv2d(orig, sobel_y))


def gaussian(orig, kernel_size=3, sigma=None):
    assert kernel_size % 2 == 1
    if sigma == None:
        sigma = (kernel_size - 1) / 2
    ax = np.linspace((1 - kernel_size) // 2, (kernel_size - 1) // 2, kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / 2 / np.square(sigma))
    kernel /= np.max(kernel)
    result = conv2d(orig, kernel)
    return result * 255 / np.max(result)


def mean(orig, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size ** 2
    return conv2d(orig, kernel)


def median(orig, kernel_size=3):
    print('calcing median')
    orig = corner_edge_pad(orig, kernel_size - 1)
    ret = np.ndarray(orig.shape)
    for i in range(orig.shape[0]):
        for j in range(orig.shape[1]):
            ret[(i, j)] = np.median(orig[i:i + kernel_size, j:j + kernel_size])

    return ret.reshape(orig.shape)


edge_kernels = {
    'Sobel':sobel, 
    'Roberts':roberts, 
    'Prewitt':prewitt
}
mean_kernels = {
    'Gaussian':gaussian, 
    'Mean':mean, 
    'Median':median
}
# okay decompiling ops.cpython-37.pyc
