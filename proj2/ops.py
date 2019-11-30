"""
This file contains all algorithms impled for proj2
"""

import cv2
import numpy as np
import cv2
from matplotlib import pyplot as plt

def load_image(img_name):
    ret = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    m = max(ret.shape[0], ret.shape[1])
    if m > 128:
        ret = cv2.resize(ret, (ret.shape[1] * 128 // m, ret.shape[0] * 128 // m))
    return ret

def get_image_bw(img):
    _, im_bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return im_bw

# default square kernel for binary image
binary_kernel = np.array([[1,1,1]] * 3, 'uint8')
# default square kernel for gray image
gray_kernel = np.array([[-3,0,-3]] * 3, 'int8')

def b_dilation(img_, kernel=binary_kernel):
    assert(kernel.shape[0] == kernel.shape[1])
    sz = kernel.shape[0]
    ew = sz // 2
    ret = np.zeros((img_.shape[0] + sz - 1, img_.shape[1] + sz - 1), 'uint8')
    for i in range(0, img_.shape[0]):
        for j in range(0, img_.shape[1]):
            if img_[i, j]:
                ret[i:i + sz, j:j + sz] = np.logical_or(ret[i:i + sz, j:j + sz], kernel)
    ret[ret > 0] = 255
    return ret[ew : -ew, ew: -ew]

def b_erosion(img_, kernel=binary_kernel):
    assert(kernel.shape[0] == kernel.shape[1])
    sz = kernel.shape[0]
    ew = sz // 2
    
    img = np.pad(img_, (ew, ew), 'edge')
    # square kernel
    nkern = np.logical_not(kernel)
    ret = np.zeros(img_.shape, 'uint8')
    for i in range(0, img_.shape[0]):
        for j in range(0, img_.shape[1]):
            # stands for P->Q(logical implication)
            ret[i, j] = np.logical_or(nkern, img[i:i + sz, j:j + sz]).all()
    
    ret[ret > 0] = 255
    return ret

def b_opening(img_, kernel=binary_kernel):
    return b_dilation(b_erosion(img_, binary_kernel), binary_kernel)

def b_closing(img_, kernel=binary_kernel):
    return b_erosion(b_dilation(img_, binary_kernel), binary_kernel)

# edge detection
def edge_surrounding(img):
    return b_dilation(img) - b_erosion(img)

def edge_internal(img):
    return img - b_erosion(img)

def edge_external(img):
    return b_dilation(img) - img

# morphological reconstruction
def get_disc_kernel(ksize = 3):
    ret = np.ndarray((ksize, ksize))
    center = (ksize - 1) / 2
    disq = center ** 2
    
    for i in range(ksize):
        for j in range(ksize):
            ret[i,j] = (i - center) ** 2 + (j - center) ** 2 <= disq
    
    return ret

def get_marker(img_, ite_count = 20):
    from skimage.measure import label
    # generate marker image
    i = 0
    pimg = img_
    img = img_.copy()
    while np.any(img) and i < ite_count:
        pimg = img
        img = b_erosion(img)
        i = i + 1
    
    # mark the image
    return label(pimg).astype('uint8')

def conditional_dilation(marker, mask, kernel=binary_kernel, max_itecnt=50):
    assert(kernel.shape[0] == kernel.shape[1])
    sz = kernel.shape[0]
    ew = sz // 2
    ret = np.pad(marker, (ew, ew), 'constant', constant_values=(0, 0))
    mask = np.pad(mask, (ew, ew), 'edge')
    mask[mask > 0] = 1
    pimg = None
    itecnt = 0
    while True:
        itecnt = itecnt + 1
        pimg = ret.copy()
        for i in range(0, marker.shape[0]):
            for j in range(0, marker.shape[1]):
                if pimg[i + ew, j + ew]:
                    ret[i:i + sz, j:j + sz] += (ret[i:i + sz, j:j + sz] == 0) * kernel
        ret = mask * ret
        if np.all(pimg == ret) or itecnt > max_itecnt:
            break
    return ret[ew:-ew, ew:-ew]

def g_dilation(img_, kernel=binary_kernel):
    assert(kernel.shape[0] == kernel.shape[1])
    sz = kernel.shape[0]
    ew = sz // 2
    img = np.pad(img_, (ew, ew), 'edge').astype('int16')
    ret = np.ndarray(img_.shape, 'int16')
    for i in range(0, img_.shape[0]):
        for j in range(0, img_.shape[1]):
            ret[i, j] = np.max(img[i:i + sz, j:j + sz] + kernel)
    ret[ret > 255] = 255
    ret[ret < 0] = 0
    return ret.astype("uint8")

def g_erosion(img_, kernel=binary_kernel):
    assert(kernel.shape[0] == kernel.shape[1])
    sz = kernel.shape[0]
    ew = sz // 2
    img = np.pad(img_, (ew, ew), 'edge').astype('int16')
    ret = np.ndarray(img_.shape).astype('int16')
    for i in range(0, img_.shape[0]):
        for j in range(0, img_.shape[1]):
            ret[i, j] = np.min(img[i:i + sz, j:j + sz] - kernel)
    ret[ret > 255] = 255
    ret[ret < 0] = 0
    return ret.astype("uint8")

def g_opening(img_, kernel=binary_kernel):
    return g_dilation(g_erosion(img_, kernel), kernel)

def g_closing(img_, kernel=binary_kernel):
    return g_erosion(g_dilation(img_, kernel), kernel)

# morphological reconstruction
def g_reconstruct_dilation(img_, kernel=gray_kernel, ite_count=10):
    img = img_.copy()
    for _ in range(ite_count):
        img = g_erosion(img, kernel)
    # reconstruct the image
    i = 25
    while True and i > 0:
        pimg = img.copy()
        img = np.minimum(img_, g_dilation(img, kernel))
        if np.all(img == pimg):
            break
        i = i - 1
    return img
    
def g_reconstruct_erosion(img_, kernel=gray_kernel, ite_count=10):
    img = img_.copy()
    for _ in range(ite_count):
        img = g_dilation(img, kernel)
    # reconstruct the image
    i = 25
    while True and i > 0:
        pimg = img.copy()
        img = np.maximum(img_, g_erosion(img, kernel))
        if np.all(img == pimg):
            break
        i = i - 1
    return img

# morphological gradient
def gradient_external(img, kernel=gray_kernel):
    return 0.5 * (g_dilation(img, kernel) - img)

def gradient_internal(img, kernel=gray_kernel):
    return 0.5 * (img - g_erosion(img, kernel))

def gradient_surrounding(img, kernel=gray_kernel):
    return 0.5 * (g_dilation(img, kernel) - g_erosion(img, kernel))


operations = {
    "original image": lambda x:x,
    "convert to binary image": get_image_bw,
    "binary dilation": b_dilation,
    "binary erosion": b_erosion,
    "binary opening": b_opening,
    "binary closing": b_closing,
    "binary edge detection(surrounding)": edge_surrounding,
    "binary edge detection(external)": edge_external,
    "binary edge detection(internal)": edge_internal,
    "binary get labelled marker image": get_marker,
    "binary conditional dilation": conditional_dilation,
    "gray image dilation": g_dilation,
    "gray image erosion": g_erosion,
    "gray image opening": g_opening,
    "gray image closing": g_closing,
    "gray image reconstruction(OBR)": g_reconstruct_dilation,
    "gray image reconstruction(CBR)": g_reconstruct_erosion,
    "gray image edge detection(external)": gradient_external,
    "gray image edge detection(internal)": gradient_internal,
    "gray image edge detection(surrounding)": gradient_surrounding
}
