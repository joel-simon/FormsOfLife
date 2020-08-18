import time
import random
import cv2
import numpy as np
import taichi as ti
from src.config import img_size
ti.init(arch=ti.gpu)

n = img_size
batch_size = 200
alive = ti.field(ti.i32, shape=(n*batch_size, n))
count = ti.field(ti.i32, shape=(n*batch_size, n))
mask = ti.field(ti.i32, shape=(n*batch_size, n))
life_count = ti.field(ti.i32, shape=(batch_size,))

@ti.func
def get_count(i, j):
    return (alive[i - 1, j] + alive[i + 1, j] + alive[i, j - 1] +
            alive[i, j + 1] + alive[i - 1, j - 1] + alive[i + 1, j - 1] +
            alive[i - 1, j + 1] + alive[i + 1, j + 1])

@ti.kernel
def run():
    for i, j in alive:
        if not mask[i, j]:
            continue
        count[i, j] = get_count(i, j)
    for i, j in alive:
        if not mask[i, j]:
            continue
        if alive[i, j]:
            if (count[i, j] != 2 and count[i, j] != 3):
                alive[i, j] = 0
            # else:
                # Count the amount of new life preserved.
                # life_count[i//ti.static(n)] += 1
        elif count[i, j] == 3:
            alive[i, j] = 1
            # Count the amount of new life created.
            life_count[i//ti.static(n)] += 1

@ti.kernel
def init():
    for i in life_count:
        life_count[i] = 0
    for i, j in alive:
        if not mask[i, j]:
            alive[i, j] = 0

def make_seed(seed=1, radius=4, p=.1):
    myrandom = random.Random(seed)
    arr = np.zeros((n, n), dtype='i')
    for i in range((n//2)-radius, (n//2)+radius+1):
        for j in range((n//2)-radius, (n//2)+radius+1):
            if myrandom.random() < 0.5:
                arr[i, j] = 1
    return arr

def evaluate_batch(seed, images, steps):
    assert seed.shape == (n, n), seed.shape
    assert images.shape[1:] == (n, n), images.shape
    for i in range(images.shape[0]):
        images[i, 0, :] = 0
        images[i, -1, :] = 0
        images[i, :, 0] = 0
        images[i, :, -1] = 0

    masks_arr = np.vstack(images)

    if masks_arr.shape[0] < batch_size * n:
        diff = (batch_size*n) - masks_arr.shape[0]
        derp = np.pad(masks_arr, ((0, diff), (0, 0)), constant_values=0)
        mask.from_numpy(derp)
    else:
        mask.from_numpy(masks_arr)
    
    seeds = np.vstack([ seed.copy() for _ in range(batch_size)])
    masks_arr = np.vstack(masks_arr)
    alive.from_numpy(seeds)
    
    init()
    for _ in range(steps):
        run()
    return life_count.to_numpy()[:images.shape[0]]

# if __name__ == '__main__':
#     import sys
#     steps = 2000
#     start = time.time()
#     seed = make_seed()
#     masks_arr = np.ones((batch_size, n, n), dtype='i')
#     total_life = evaluate_batch(seed, masks_arr, steps)
#     print(total_life)
#     print(time.time() - start)