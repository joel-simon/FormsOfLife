import sys
import cv2
import numpy as np
import random
from src.config import img_size
def make_seed(seed=1, n=64, radius=4, p=.5):
    random.seed(1)
    arr = np.zeros((n, n), dtype='i')
    for i in range((n//2)-radius, (n//2)+radius+1):
        for j in range((n//2)-radius, (n//2)+radius+1):
            if random.random() < p:
                arr[i, j] = 1
    return arr

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1

img = make_seed(seed, img_size)
cv2.imwrite(f'seed_{seed}.png', img*255)