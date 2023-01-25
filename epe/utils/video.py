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

from epe.utils.dataset_collector import uniform_sampling_dataset, uniform_sampling_run

# For debugging
# TODO: turn off for faster training?>
parser = ArgumentParser()
parser.add_argument('model_name', type=str)
parser.add_argument('--iteration', type=str, default='latest')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1)
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

dataset_meta_path = '/mnt/remote/data/users/kacper/datasets/somers-town_weather_v1'

out_dir = '/home/kacper/data/out'


run_path = '/tmp/kacper/wayve/reefshark/somerstown_data_collection--2023-01-04--16-41-50/ningaloo--3_6_190--jaguaripacenoslipdynamics--ningaloo_av_2_0/2023-01-04--16-41-50--somerstown-aft-loop-anti-clockwise-v1--af07df57deaf357b--307454da'
# data_root = '/tmp/kacper/wayve/reefshark/somerstown_data_collection--2023-01-04--16-41-50'
data_root = '/'.join(run_path.split('/')[:-1])
run_id = run_path.split('/')[-1]
video_test_path = f'/tmp/kacper/video_test_{run_id}_{start_index}.csv'

# sim_somers_town = '/home/kacper/code/EPE/datasets/somers_town/sim_files.csv'
# sim_car_filter = lambda car: 'ningaloo' in car
path_filter = lambda x: run_id in x 
SAMPLING_RATE = 6
FRAME_RATE = 5.0
assert FRAME_RATE.is_integer()

# uniform_sampling_dataset(video_test_path, SAMPLING_RATE, start_index=start_index, path_filter=path_filter, data_root=data_root)
uniform_sampling_run(video_test_path, run_path, sampling_rate=SAMPLING_RATE)

g_buffers = ['depth', 'normal']
sim_data_modes = ['rgb', 'segmentation', *g_buffers]

# gbuf_mean = []
# gbuf_std = []

gbuf_stats  = torch.load(os.path.join(dataset_meta_path, 'gbuf_stats.pt'))

dataset_fake_val = SimDataset(ds.utils.read_filelist(video_test_path,
	sim_data_modes, dataset_name='', is_sim=True), data_root=data_root, gbuffers=g_buffers, inference=True,
	gbuf_mean=gbuf_stats['gbuf_mean'], gbuf_std=gbuf_stats['gbuf_std'])

def seed_worker(id):
	random.seed(torch.initial_seed() % np.iinfo(np.int32).max)
	np.random.seed(torch.initial_seed() % np.iinfo(np.int32).max)
	pass

# %%
collate_fn_val   = ds.EPEBatch.collate_fn

loader_fake = torch.utils.data.DataLoader(dataset_fake_val, \
	batch_size=1, shuffle=False, \
	num_workers=4, pin_memory=True, drop_last=False, worker_init_fn=seed_worker, collate_fn=collate_fn_val)


# %%
gen_cfg = dict(run.config.get('generator', {}))
gen_cfg = dict(gen_cfg.get('config', {}))
gen_cfg['num_classes']          = dataset_fake_val.num_classes
gen_cfg['num_gbuffer_channels'] = dataset_fake_val.num_gbuffer_channels
gen_cfg['cls2gbuf']             = dataset_fake_val.cls2gbuf

print(gen_cfg)
generator_type     = run.config['generator']['type']

# %%
class PassthruGenerator(torch.nn.Module):
	def __init__(self, network):
		super(PassthruGenerator, self).__init__()
		self.network = network
		pass

	def forward(self, epe_batch):
		return torch.sigmoid(self.network(epe_batch))

if generator_type == 'hr':
	generator = nw.ResidualGenerator(nw.make_ienet(gen_cfg))
elif generator_type == 'hr_new':
	print('HR_NEW:')
	generator = nw.ResidualGenerator(nw.make_ienet2(gen_cfg))
elif generator_type == 'hr_new_pass_thru':
	generator = PassthruGenerator(nw.make_ienet2(gen_cfg))

generator.to(device)
generator.eval()
toggle_grad(generator, False)
# %%
artifact = api.artifact(f'wayve-ai/EPE/{model_name}:latest')
assert artifact.logged_by().name == model_name
iteration = f'{(int(str(artifact.version)[1:]) + 1) * 100}k'
artifact_save_path = '/tmp/kacper/weights/'
weight_path = artifact.download(os.path.join(artifact_save_path, 'artifacts', model_name, artifact.version))
weight_path = os.path.join(weight_path, f'{model_name}_gen-network.pth.tar')
generator.load_state_dict(torch.load(weight_path, map_location=device))
# %%
def _forward_generator_fake(batch_fake):
	""" Run the generator without any loss computation. """

	rec_fake = generator(batch_fake)
	return {'rec_fake':rec_fake.detach(), 'fake':batch_fake.img.detach()}

print('output directory:')
print(os.path.join(out_dir, model_name, iteration))
print()
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

			sim_save_path = os.path.join(out_dir, 'sim', batch_fake.path[i])
			os.makedirs(os.path.dirname(sim_save_path), exist_ok=True)
			save_image(fake, sim_save_path)
#

# %%
import subprocess

for bi, batch_fake in enumerate(tqdm(loader_fake)):                
	folder = batch_fake.path[0].split('cameras')[0]
	print(folder)

	epe_base_dir  = os.path.join(out_dir, model_name, iteration, folder)
	epe_video_dir = os.path.join(epe_base_dir, 'videos')
	epe_video_save_path = os.path.join(epe_video_dir, 'front-forward--rgb-epe.mp4')
	print(epe_video_save_path)
	os.makedirs(os.path.dirname(epe_video_save_path), exist_ok=True)


	command = f'ffmpeg -framerate {FRAME_RATE}  -pattern_type glob -i "{epe_base_dir}/cameras/front-forward--rgb/*.jpeg"  -c:v libx264 -r {FRAME_RATE} {epe_video_save_path}'
	subprocess.run(command, shell=True)

	sim_base_dir  = os.path.join(out_dir, 'sim', folder)
	sim_video_dir = os.path.join(sim_base_dir, 'videos')
	sim_video_save_path = os.path.join(sim_video_dir, 'front-forward--rgb.mp4')
	if not os.path.exists(sim_video_save_path) or True:
		print(sim_video_save_path)
		os.makedirs(os.path.dirname(sim_video_save_path), exist_ok=True)

		command = f'ffmpeg -framerate {FRAME_RATE}  -pattern_type glob -i "{sim_base_dir}/cameras/front-forward--rgb/*.jpeg"  -c:v libx264 -r {FRAME_RATE} {sim_video_save_path}'
		subprocess.run(command, shell=True)

	print(sim_video_save_path)
	
	# save_image(rec_fake, save_path)
	break
# %%
