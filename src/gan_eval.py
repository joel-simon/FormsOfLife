import torch
import numpy as np
from src.gan import Generator
from src.config import img_size, latent_size
from src.repair import repair
generator = Generator(img_size, latent_size)
generator.load_state_dict(torch.load(f'/home/joel/Projects/FormsOfLife/saved_models/dcgan_gen_{img_size}'))
generator.cuda()
generator.eval()

def make_images(z):
    z = torch.tensor(z).float().cuda()
    imgs = generator(z).detach().cpu().numpy().squeeze()
    imgs = (imgs > 0).astype('i')
    if len(imgs.shape) == 3:
        imgs = np.array([ repair(img) for img in imgs ])
    else:
        imgs = repair(imgs)
    return imgs.astype('uint8')

