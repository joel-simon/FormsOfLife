import os, sys
import torch
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
from qdpy import algorithms, containers, benchmarks, plots

from src.gol import evaluate, make_seed
from src.vae import VAE

img_shape = (32, 32)
latent_size = 50
encoder_layers = [ 1024, 512 ]   
decoder_layers = [ 512, 1024 ]

model = VAE(img_shape, 2, encoder_layers, latent_size, decoder_layers)
model.load_state_dict(torch.load('saved_models/vae'))

model_2d = VAE(img_shape, 2, encoder_layers, 4, decoder_layers)
model_2d.load_state_dict(torch.load('saved_models/vae_d2'))

generator = model.decoder.cuda()
classifer = model_2d.encoder.cuda()

seed = make_seed()

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std

def evaluate_genome(z, steps=500):
    z = np.array([z])
    assert 0 <= z.min() <= 1.0
    assert 0 <= z.max() <= 1.0
    z = (z * 10) - 5
    z = torch.from_numpy(z).float().cuda()
    
    samples = generator(z)
    
    features = reparameterize(*classifer(samples)).detach().cpu().numpy()[0]
    # print(features)
    features = (features + 10) / 20
    # print(features)

    img = samples.detach().cpu().numpy().argmax(axis=1)[0]
    total_life = evaluate(seed, img, steps=steps)
    fitness = total_life / (steps * img.shape[0] * img.shape[1])
    
    assert features.shape == (4,)
    return (fitness,), features

grid = containers.Grid(
    shape=(6, 6, 6, 6),
    max_items_per_bin=1,
    fitness_domain=((0., 1.),),
    features_domain=((0., 1.),(0., 1.),(0., 1.),(0., 1.))
)
algo = algorithms.RandomSearchMutPolyBounded(
    grid,
    budget=5000,
    batch_size=50,
    dimension=50,
    optimisation_task="maximisation"
)
logger = algorithms.TQDMAlgorithmLogger(algo)
best = algo.optimise(evaluate_genome)
# Print results info
print(algo.summary())

# Plot the results
plots.default_plots_grid(logger)

print("All results are available in the '%s' pickle file." % logger.final_filename)
