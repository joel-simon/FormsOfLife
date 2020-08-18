import os, sys, copy, time, copy
import numpy as np
from tqdm.notebook import tqdm
import math
import numpy as np
import PIL
import PIL.Image
import IPython.display
from io import BytesIO
import matplotlib.pyplot as plt
import cv2
plt.rcParams['figure.dpi'] = 100

def imgrid(images, cols=5, pad=1, pad_v=0):
    images = [
       np.pad(im, pad, "constant", constant_values=pad_v)
       for im in images
    ]
    if len(images) <= cols:
        return np.hstack(images)
    rows = math.ceil(len(images) / cols)
    h, w = images[0].shape[:2]
    assert all(im.shape[0] == h for im in images), "images are not all the same size"
    assert all(im.shape[1] == w for im in images), "images are not all the same size"
    result = np.zeros((h*rows, w*cols))
    idx = 0
    for r in range(rows):
        for c in range(cols):
            result[r*h:(r+1)*h, c*w:(c+1)*w] = images[idx]
            idx += 1
            if idx == len(images):
                break
    return result

def imshow(a, scale=True, format='JPEG'):
    if scale:
        amin, amax = a.min(), a.max()
        if amin != amax:
            a = (a - amin) / (amax - amin) * 255
    a = a.astype('uint8')
    image = PIL.Image.fromarray(a, 'L' if len(a.shape) == 2 else 'RGB')
    buffered = BytesIO()
    image.save(buffered, format=format, quality=90)
    im_data = buffered.getvalue()
    disp = IPython.display.display(IPython.display.Image(im_data))
    return disp

# def pltshow(img):
#     plt.axis('off')
#     plt.imshow(img)