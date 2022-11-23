# %%
import warnings

from epe.dataset.sim_dataset import SimDataset
from epe.experiment.BaseExperiment import toggle_grad
warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed')
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from argparse import ArgumentParser
import logging
from time import time
from pathlib import Path
import random

import imageio
import numpy as np
from skimage.transform import resize
import torch
import torch.utils.data
from torch import autograd
from torchvision.utils import make_grid, save_image

import kornia as K

import epe.utils
import epe.dataset as ds
import epe.network as nw
import epe.experiment as ee
from epe.matching import MatchedCrops, IndependentCrops
import os
from tqdm import tqdm

from epe.utils.dataset_collector import uniform_sampling

# For debugging
# TODO: turn off for faster training?
# torch.autograd.set_detect_anomaly(True)
#  %%
parser = ArgumentParser()
parser.add_argument('model_name', type=str)
parser.add_argument('--iteration', type=str, default='latest')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--start_index', type=int, default=0)
args = parser.parse_args()

# %%
device = f'cuda:{args.gpu}'
# %%
import wandb
api = wandb.Api()
model_name = args.model_name
iteration = args.iteration
start_index = args.start_index
runs = api.runs(path="wayve-ai/EPE", filters={"display_name": model_name})
run = runs[0]
# %%

# gen_cfg = dict(self.cfg.get('generator', {}))
# self.gen_cfg = dict(gen_cfg.get('config', {}))

dataset_meta_path = '/home/kacper/data/datasets/rider_v0'
out_dir = '/home/kacper/data/datasets/out'
inference_path = os.path.join(dataset_meta_path, 'sim_files.csv')

SAMPLING_RATE = 1
car_filter = lambda x: 'ningaloo' in x

uniform_sampling(inference_path, SAMPLING_RATE, start_index=start_index, data_root=dataset_meta_path, car_filter=car_filter)

data_root = '/home/kacper/data/datasets/rider_v0'
g_buffers = ['depth', 'normal']
sim_data_modes = ['rgb', 'segmentation', *g_buffers]

gbuf_stats  = torch.load(os.path.join(dataset_meta_path, 'gbuf_stats.pt'))

dataset_fake_val = SimDataset(ds.utils.read_azure_filelist(inference_path,
	sim_data_modes), data_root=data_root, gbuffers=g_buffers, inference=True,
	gbuf_mean=gbuf_stats['gbuf_mean'], gbuf_std=gbuf_stats['gbuf_std'])

def seed_worker(id):
	random.seed(torch.initial_seed() % np.iinfo(np.int32).max)
	np.random.seed(torch.initial_seed() % np.iinfo(np.int32).max)
	pass

# %%
collate_fn_val   = ds.EPEBatch.collate_fn

loader_fake = torch.utils.data.DataLoader(dataset_fake_val, \
	batch_size=args.batch_size, shuffle=False, \
	num_workers=8, pin_memory=True, drop_last=False, worker_init_fn=seed_worker, collate_fn=collate_fn_val, prefetch_factor=6)


# %%
gen_cfg = dict(run.config.get('generator', {}))
gen_cfg = dict(gen_cfg.get('config', {}))
gen_cfg['num_classes']          = dataset_fake_val.num_classes
gen_cfg['num_gbuffer_channels'] = dataset_fake_val.num_gbuffer_channels
gen_cfg['cls2gbuf']             = dataset_fake_val.cls2gbuf

print(gen_cfg)
generator_type     = run.config['generator']['type']

if generator_type == 'hr':
	generator = nw.ResidualGenerator(nw.make_ienet(gen_cfg))
elif generator_type == 'hr_new':
	generator = nw.ResidualGenerator(nw.make_ienet2(gen_cfg))

generator.to(device)
generator.eval()
toggle_grad(generator, False)
# %%
weights_name = 'gen-network.pth.tar' if iteration == 'latest' else f'{iteration}_gen-network.pth.tar'
weight_path = f'/home/kacper/data/EPE/weights/{model_name}/{weights_name}'
generator.load_state_dict(torch.load(weight_path, map_location=device))


# %%
def _forward_generator_fake(batch_fake):
	""" Run the generator without any loss computation. """

	rec_fake = generator(batch_fake)
	return {'rec_fake':rec_fake.detach(), 'fake':batch_fake.img.detach()}

with torch.no_grad():
	for bi, batch_fake in enumerate(tqdm(loader_fake)):                
		gen_vars = _forward_generator_fake(batch_fake.to(device))

		for i in range(len(gen_vars['rec_fake'])):
			rec_fake = gen_vars['rec_fake'][i].detach()
			fake = gen_vars['fake'][i].detach()
			
			rec_fake = gen_vars['rec_fake'][i].detach()
			save_path = os.path.join(out_dir, model_name, iteration, batch_fake.path[i])
			os.makedirs(os.path.dirname(save_path), exist_ok=True)
			save_image(rec_fake, save_path)

			# sim_save_path = os.path.join(out_dir, 'sim', batch_fake.path[i])
			# os.makedirs(os.path.dirname(sim_save_path), exist_ok=True)
			# save_image(fake, sim_save_path)
#
