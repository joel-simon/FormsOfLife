import numpy as np
import cv2
import random
from src.img_utils import subdivide
from src.mutation import mutate
from src.repair import repair
from src.config import img_size

################################################################################
# Utils.

def keep_max_cc(img):
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(img.astype('uint8'), connectivity=4)
    max_area = np.argmax(stats[1:, -1]) +1
    return (labels_im == max_area).astype('i')

def norm(a):
    return (a - a.min()) / (a.max() - a.min())
    
def gkern(kernlen=32, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

################################################################################

def sample(size=img_size, m=8, target=.6):
    img = np.zeros((8, 8), dtype='i')
    img[2:6, 2:6] = 1
    img = mutate(img, m)[0]
    subs = 0
    while img.shape[0] < size:
        subs += 1
        img = subdivide(img)
        img = mutate(img, m*subs)[0]
    img = keep_max_cc(img)
    return img.astype('i')

from src.perlin import generate_fractal_noise_2d
import scipy.stats as st

kerns = { s: norm(gkern(s)) for s in (32, 64, 128) }

def sample_perlin(size=32):
    _size = 32
    img = norm(generate_fractal_noise_2d((_size, _size), (4, 4), 3))
    img = np.power(img, random.uniform(.5, .8))
    k = np.power(kerns[_size], random.uniform(.2, .8))
    img = img * k
    img = cv2.resize(img, (size, size))
    return (img > random.uniform(.2, .5)).astype('i')