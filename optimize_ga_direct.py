import os, sys, random, time, math
import numpy as np
import cv2
import pymoo
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize

from src.sample import sample_perlin
from src.fitness import fitness
from src.gol_batch import make_seed, batch_size
from src.utils import rand_key
from src.config import img_size, root
img_r = img_size//2

def mutate(img, m=4):
    img = img.copy()
    for _ in range(random.randint(1, m)):
        r = random.randint(1, int(math.sqrt(img_size)))
        x = np.random.randint(0, img_size-r)
        y = np.random.randint(0, img_size-r)
        y1 = y+r
        x1 = x+r
        if random.random() < 0.5:
            img[y:y1, x:x1] = 0
        else:
            img[y:y1, x:x1] = 1
    return img

def crossover(p1, p2):
    c1 = p1.copy()
    c2 = p2.copy()
    x = np.random.randint(0, img_size)
    if random.random() < 0.5:
        c1[:x] = p2[:x]
        c2[:x] = p1[:x]
    else:
        c1[:, :x] = p2[:, :x]
        c2[:, :x] = p1[:, :x]
    return c1, c2

class FOLProblem(pymoo.model.problem.Problem):
    def __init__(self):
        super().__init__(n_var=img_size*img_size, type_var=np.int32, n_obj=1, n_constr=0)
        self.seed = make_seed()
    def _evaluate(self, X, out, *args, **kwargs):
        images = X.reshape(-1, img_size, img_size)
        assert 1.0 > images.mean() > 0.0
        out['F'] = - 1 * fitness(images, self.seed)
        # out['F'] = - 1 * fitness(images, self.seed) / np.power(images.mean(axis=(1, 2)), .75)
        # out['G'] = images.mean(axis=(1, 2)) > .50

class FOLAntiProblem(pymoo.model.problem.Problem):
    def __init__(self):
        super().__init__(n_var=img_size*img_size, type_var=np.int32, n_obj=1, n_constr=0)
        self.seed = make_seed()
    def _evaluate(self, X, out, *args, **kwargs):
        images = X.reshape(-1, img_size, img_size).copy()
        for img in images:
            img[img_r-6: img_r+6, img_r-6: img_r+6] = 1.0
        out['F'] = 1 * fitness(images, self.seed)

class FOLProblem2OBJ(pymoo.model.problem.Problem):
    def __init__(self):
        super().__init__(n_var=img_size*img_size, type_var=np.int32, n_obj=2, n_constr=0)
        self.seed = make_seed()
    def _evaluate(self, X, out, *args, **kwargs):
        images = X.reshape(-1, img_size, img_size)
        t = time.time()
        out['F'] = np.stack([
            -1 * fitness(images, self.seed),
            images.mean(axis=(1, 2))
        ], axis=-1)
        # out['G'] = out['F'][:, 1] < .05

class FOLCrossover(pymoo.model.crossover.Crossover):
    def __init__(self, **kwargs):
        super().__init__(n_parents=2, n_offsprings=2, **kwargs)
    def _do(self, problem, X, **kwargs):
        X = X.reshape(2, -1, img_size, img_size).copy()
        for i in range(X.shape[1]):
            c1, c2 = crossover(X[0, i], X[1, i])
            X[0, i] = c1
            X[1, i] = c2
        return X.reshape(2, -1, img_size*img_size)

# class NoCross(pymoo.model.crossover.Crossover):
#     def __init__(self, **kwargs):
#         super().__init__(n_parents=1, n_offsprings=1, **kwargs)
#     def _do(self, problem, X, **kwargs):
#         return X.copy()

class FOLMutation(pymoo.model.mutation.Mutation):
    def _do(self, problem, X, **kwargs):
        X = X.reshape(-1, img_size, img_size).copy()
        for i in range(len(X)):
            X[i] = mutate(X[i], 1)
        return X.reshape(-1, img_size*img_size)

class FOLSampling(pymoo.model.sampling.Sampling):
    def _do(self, problem, n_samples, **kwargs):        
        return np.stack([
            sample_perlin(img_size).flatten()
            for _ in range(n_samples)
        ])

if __name__ == '__main__':
    multi_obj = False
    anti = False
    key = f'{img_size}_{rand_key()}'
    if anti:
        key = 'anti_' + key
    if multi_obj:
        key = '2o_' + key
    os.makedirs(f'./{root}/ga_direct/{key}')
    last_best = np.inf
    history = []

    def callback(algorithm):
        global last_best
        F = algorithm.pop.get("F")
        X = algorithm.pop.get("X")
        best_f = F.max()
        best_i = np.argmax(F)
        
        print(key, algorithm.n_gen)
        
        if (not multi_obj) and best_f < last_best:
            img = X[best_i].reshape(img_size, img_size) * 255
            if anti:
                img[img_r-6: img_r+6, img_r-6: img_r+6] = 255
                img[0, :] = 255
                img[-1, :] = 255
                img[:, 0] = 255
                img[:, -1] = 255
            cv2.imwrite(f'./{root}/ga_direct/{key}/{algorithm.n_gen:04}.png', img)
            last_best = best_f
            print('NEW BEST', F.mean(axis=0))
        
        if not multi_obj:
            history.append(best_f)

    algorithm = (NSGA2 if multi_obj else GA)(
        pop_size=200,
        eliminate_duplicates=False,
        sampling=FOLSampling(),
        mutation=FOLMutation(),
        crossover=FOLCrossover(),
        callback=callback
    )
    
    if anti:
        prob = FOLAntiProblem()
    elif multi_obj:
        prob = FOLProblem2OBJ()
    else:
        prob = FOLProblem()
        
    res = minimize(
        prob,
        algorithm,
        termination=('n_gen', 400),
        seed=None,
        verbose=False,
        save_history=False
    )
    np.save(f'./{root}/ga_direct/{key}/F', res.F)
    np.save(f'./{root}/ga_direct/{key}/X', res.X)
    np.save(f'./{root}/ga_direct/{key}/history', np.array(history))
