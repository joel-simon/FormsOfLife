import numpy as np
from src.config import img_size, gol_steps, max_pixels
from src.gol_batch import evaluate_batch, batch_size

def fitness(images, seed, steps=gol_steps, max_pixels=max_pixels):
    assert images.min() >= 0.0 and images.max() <= 1.0
    total_life = evaluate_batch(seed, images, steps=steps)
    f = total_life / (steps*img_size*img_size)
    if max_pixels is not None:
        # Apply constraint penalty
        means = images.mean(axis=(1, 2))
        f[means > max_pixels] *= .005 / (means[means > max_pixels])**2

    assert f.min() >= 0.0 and f.max() <= 1.0
    return f