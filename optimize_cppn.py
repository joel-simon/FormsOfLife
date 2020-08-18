import os, math
import neat
import cv2
import numpy as np

from src import neat_visualize as visualize
from src.fitness import fitness
from src.gol_batch import make_seed, batch_size
from src.config import img_size, root
from src.utils import batch, rand_key

seed = make_seed()
key = rand_key()
dout = f'./{root}/cppn/{key}'
os.makedirs(dout)
history = []
class Reporter(neat.reporting.BaseReporter):
    def start_generation(self, generation):
        self.generation = generation
    def post_evaluate(self, config, population, species, best_genome):
        history.append(best_genome.fitness)
        img = make_cppn(best_genome, config).astype('uint8')
        print('best', img.mean())
        cv2.imwrite(f'{dout}/{self.generation:04}.png', (img*255))
        node_names = {-3:'x', -2:'y', -1:'r', 0:'out'}
        visualize.draw_net(
            config, best_genome, view=False, node_names=node_names,
            filename=f'{dout}/{self.generation:04}_net'
        )
        print(key, self.generation, max(g.fitness for g in population.values()))

def dist(p1, p2):
    (x1, y1) = p1
    (x2, y2) = p2
    return math.hypot(x2-x1, y2-y1)

def make_cppn(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    mask = np.zeros((img_size, img_size), dtype='i')
    c = img_size//2
    
    for i in range(1, img_size-1):
        for j in range(1, img_size-1):
            output = net.activate([
                (2 * i / img_size) - 1,
                (2 * j / img_size) - 1,
                (2 * dist((i, j), (c, c)) / img_size) - 1
            ])
            mask[i, j] = output[0] > 0.0

    return mask

def eval_genomes(genomes, config):
    for g_batch in batch(genomes, 200):
        images = np.stack([ make_cppn(genome, config) for _, genome in g_batch ])
        F = fitness(images, seed)
        for f, img, (genome_id, genome) in zip(F, images, g_batch):
            genome.fitness = f

def run(config_file):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))
    p.add_reporter(Reporter())
    winner = p.run(eval_genomes, 200)

    visualize.plot_stats(stats, ylog=False, view=False)
    visualize.plot_species(stats, view=False)
    np.save(f'{dout}/history', np.array(history))
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_config.txt')
    run(config_path)