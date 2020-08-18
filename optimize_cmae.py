import cma
import cv2
import torch
import numpy as np
import os, pickle
from src.fitness import fitness
from src.gol_batch import make_seed
from src.gan import Generator
from src.config import img_size, latent_size, root
from src.gan_eval import make_images
from src.fitness import fitness
from src.utils import rand_key

es = cma.CMAEvolutionStrategy([0]*latent_size, 0.2, {
    'popsize': 100,
    'maxfevals': 200 * 400,
    'maxiter': np.inf
})

key = f'still_{img_size}_{rand_key()}'
dout = f'{root}/gan_cmae/{key}'

os.makedirs(dout)
seed = make_seed()
history = []

def save_best():
    z = [ es.result.xfavorite ]
    img = make_images(z) * 255
    cv2.imwrite(f'{dout}/{es.result.evaluations:05}.png', img)

last_best = 0
while not es.stop(): # Use the ask an tell API to do batched.
    solutions = es.ask()
    images = make_images(solutions)
    es.tell(solutions, -1*fitness(images, seed))
    res = es.result
    if res.fbest < last_best:
        print(key, res.evaluations, res.fbest)
        history.append((res.evaluations, res.fbest))
        save_best()
        last_best = res.fbest

res = es.result
history.append((res.evaluations, res.fbest))

with open(f'{dout}/result.p', 'wb') as f:
    pickle.dump((es.result.xbest, es.result.fbest, history), f)

print(key, 'done')