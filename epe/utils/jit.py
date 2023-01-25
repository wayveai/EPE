# %%
import torch
from argparse import ArgumentParser
from epe.experiment.BaseExperiment import toggle_grad
import random
import numpy as np
from epe.dataset.sim_dataset import SimDataset
import epe.dataset as ds
import epe.network as nw
import os
import wandb
# %%
# %%
parser = ArgumentParser()
parser.add_argument('model_name', type=str)
parser.add_argument('--iteration', type=str, default='latest')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--start_index', type=int, default=0)
if 'get_ipython' in dir(__builtins__):
    args = parser.parse_args('jolly-glade-289'.split(' '))
else:
    args = parser.parse_args()
# %%
print(args.model_name)
# %%
device = f'cuda:{args.gpu}'
# %%
api = wandb.Api()
model_name = args.model_name
iteration = args.iteration
start_index = args.start_index
runs = api.runs(path="wayve-ai/EPE", filters={"display_name": model_name})
run = runs[0]
# %%
dataset_meta_path = '/mnt/remote/data/users/kacper/datasets/somers-town_weather_v1'
out_dir = '/home/kacper/data/out'

car = 'ningaloo--3_6_190--jaguaripacenoslipdynamics--ningaloo_av_2_0'
data_root = dataset_meta_path
video_test_path = os.path.join(dataset_meta_path, 'sim_files.csv')

SAMPLING_RATE = 1
start_index = 0
FRAME_RATE = 30.0
assert FRAME_RATE.is_integer()

g_buffers = ['depth', 'normal']
sim_data_modes = ['rgb', 'segmentation', *g_buffers]

# gbuf_mean = []
# gbuf_std = []

gbuf_stats  = torch.load(os.path.join(dataset_meta_path, 'gbuf_stats.pt'))
# %%
print(gbuf_stats)
# %%
print([x.item() for x in list(gbuf_stats['gbuf_mean'])])
print([x.item() for x in list(gbuf_stats['gbuf_std'])])
# %%

dataset_fake_val = SimDataset(ds.utils.read_filelist(video_test_path,
    sim_data_modes), data_root=data_root, gbuffers=g_buffers, inference=True,
    gbuf_mean=gbuf_stats['gbuf_mean'], gbuf_std=gbuf_stats['gbuf_std'], 
    )
# %%
print(len(dataset_fake_val))
# %%

def seed_worker(id):
    random.seed(torch.initial_seed() % np.iinfo(np.int32).max)
    np.random.seed(torch.initial_seed() % np.iinfo(np.int32).max)
    pass

collate_fn_val   = ds.EPEBatch.collate_fn

loader_fake = torch.utils.data.DataLoader(dataset_fake_val, \
    batch_size=1, shuffle=False, \
    num_workers=1, pin_memory=True, drop_last=False, worker_init_fn=seed_worker, collate_fn=collate_fn_val, prefetch_factor=1)

# %%
print(len(loader_fake))
# %%
example_input = None
for batch in loader_fake:
    # example_input = (example_batch.img, example_batch.gbuffers, example_batch.gt_labels)
    # example_input = {'img': example_batch.img, 'gbuffers': example_batch.gbuffers, 'gt_labels': example_batch.gt_labels}
    example_input = batch
    break

epe_batch = {}
epe_batch['img'] = example_input['img'].to(device)
epe_batch['gbuffers'] = example_input['gbuffers'].to(device)
epe_batch['gt_labels'] = example_input['gt_labels'].to(device)


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
artifact = api.artifact(f'wayve-ai/EPE/{model_name}:latest')
assert artifact.logged_by().name == model_name
artifact_save_path = '/home/kacper/code/EPE/wandb'
weight_path = artifact.download(os.path.join(artifact_save_path, 'artifacts', model_name, artifact.version))
weight_path = os.path.join(weight_path, f'{model_name}_gen-network.pth.tar')

# weights_name = 'gen-network.pth.tar' if iteration == 'latest' else f'{iteration}_gen-network.pth.tar'
# weight_path = f'/home/kacper/data/EPE/weights/{model_name}/{weights_name}'
iteration = artifact.version
generator.load_state_dict(torch.load(weight_path, map_location=device))
# %%
from torchvision import transforms
to_pil = transforms.ToPILImage()
# %%
def _forward_generator_fake(batch_fake):
	""" Run the generator without any loss computation. """

	rec_fake = generator(batch_fake)
	return {'rec_fake':rec_fake.detach(), 'fake':batch_fake.img.detach()}
# %%
output = _forward_generator_fake(example_input.to(device))
# %%
to_pil(output['fake'][0])
# %%
to_pil(output['rec_fake'][0])
# %%
traced_script_module = torch.jit.trace(generator, (epe_batch))
# %/tmp/ailib/requirements.txt%
out = traced_script_module(epe_batch)
# %%
print(out.shape)
# %%
to_pil(out[0])

# %%
jit_save_path = os.path.join(out_dir, model_name, f'{model_name}.pt')
traced_script_module.save(jit_save_path)
# %%
m = torch.jit.load(jit_save_path)
if m is not None:
    print("saved properly")
# %%
out = m(epe_batch)
print(out.shape)
to_pil(out[0])
# %%
