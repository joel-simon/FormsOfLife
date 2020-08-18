import os, sys, random, time, string
import numpy as np
import cv2
import pymoo
import torch
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_sampling, get_crossover, get_mutation

from src.gan_eval import make_images
from src.fitness import fitness
from src.gol_batch import make_seed, batch_size
from src.repair import repair
from src.utils import rand_key
from src.config import img_size, latent_size, root

class FOLProblem(pymoo.model.problem.Problem):
    def __init__(self):
        super().__init__(
            n_var=latent_size,
            type_var=np.float32,
            n_obj=1,
            n_constr=0,
        )
        self.seed = make_seed()
    def _evaluate(self, Z, out, *args, **kwargs):        
        images = make_images(Z)
        out['F'] = -1 * fitness(images, self.seed)

class FOLProblem2OBJ(pymoo.model.problem.Problem):
    def __init__(self):
        super().__init__(n_var=latent_size, type_var=np.float32, n_obj=2)
        self.seed = make_seed()
    def _evaluate(self, Z, out, *args, **kwargs):
        images = make_images(Z)
        out['F'] = np.stack([
            -1 * fitness(images, self.seed),
            images.mean(axis=(1, 2))
        ], axis=-1)

class LatentMutation(pymoo.model.mutation.Mutation):
    def _do(self, problem, X, **kwargs):
        X = X.copy()
        for x in X:
            if random.random() < .3:
                x *= random.uniform(.8, 1.2)
        p = random.uniform(.001, 3.0)
        mask = np.random.rand(X.shape[0]) > random.uniform(.1, 1.0)
        X[mask] += np.random.uniform(low=-p, high=p, size=X.shape)[mask]
        # X += np.random.normal(scale=p, size=X.shape)
        # # X += np.random.normal(scale=p, size=X.shape)
        return X

class LatentSampling(pymoo.model.sampling.Sampling):
    def _do(self, problem, n_samples, **kwargs):
        # return np.random.normal(scale=1, size=(n_samples, latent_size))
        return torch.randn(n_samples, latent_size).numpy() * 2.0

if __name__ == '__main__':
    multi_obj = False
    key = f'{img_size}_{rand_key()}'
    if multi_obj:
        key = '2o_' + key
    os.makedirs(f'{root}/ga_latent/{key}')
    last_best = 0
    history = []

    def callback(algorithm):
        global last_best
        F = algorithm.pop.get("F")
        X = algorithm.pop.get("X")
        best_f = F.max()
        best_i = np.argmax(F)
        if best_f < last_best:
            img = make_images(X[best_i:best_i+1]) * 255
            assert img.shape == (img_size, img_size), img.shape
            cv2.imwrite(f'{root}/ga_latent/{key}/{algorithm.n_gen:04}.png', img)
            last_best = best_f
            print(key, algorithm.n_gen, F.mean(axis=0))
        print(algorithm.n_gen)
        if not multi_obj:
            history.append(best_f)

    algorithm = (NSGA2 if multi_obj else GA)(
        pop_size=200,
        eliminate_duplicates=True,
        sampling=LatentSampling(),
        mutation=LatentMutation(),
        # mutation=get_mutation('real_pm', eta=40, prob=1.0),
        crossover=get_crossover('real_ux'),
        callback=callback
    )
    res = minimize(
        FOLProblem2OBJ() if multi_obj else FOLProblem(),
        algorithm,
        termination=('n_gen', 400),
        seed=None,
        verbose=False,
        save_history=False
    )
    np.save(f'{root}/ga_latent/{key}/F', res.F)
    np.save(f'{root}/ga_latent/{key}/X', res.X)
    np.save(f'{root}/ga_latent/{key}/history', np.array(history))
