# %%
import torch
import random
import numpy as np
from epe.dataset.sim_dataset import SimDataset
import epe.dataset as ds
import epe.network as nw
import os
# %%
gen_cfg = {'encoder_type': 'ENCODER', 'stem_norm': 'group', 'num_stages': 5, 'other_norm': 'group', 'gbuffer_norm': 'RAC', 'gbuffer_encoder_norm': 'residual2', 'num_gbuffer_layers': 3, 'num_classes': 12, 'num_gbuffer_channels': 4, 'cls2gbuf': {}}
generator = nw.ResidualGenerator(nw.make_ienet2(gen_cfg))
# %%
dataset_meta_path = '/mnt/remote/data/users/kacper/datasets/somers-town_weather_v0'
data_root = '/mnt/remote/data/users/kacper/datasets'
out_dir = '/home/kacper/data/out'
video_test_path = f'{dataset_meta_path}/sim_files.csv'

# sim_somers_town = '/home/kacper/code/EPE/datasets/somers_town/sim_files.csv'
# sim_car_filter = lambda car: 'ningaloo' in car
# path_filter = lambda x: '2022-10-26--10-48-33--somerstown-aft-loop-anti-clockwise-v1--1ceea0c3ea91f1ef--3955de33' in x 
# path_filter = lambda x: True

SAMPLING_RATE = 1
start_index = 0
FRAME_RATE = 30.0
assert FRAME_RATE.is_integer()

g_buffers = ['depth', 'normal']
sim_data_modes = ['rgb', 'segmentation', *g_buffers]

# gbuf_mean = []
# gbuf_std = []

gbuf_stats  = torch.load(os.path.join(dataset_meta_path, 'gbuf_stats.pt'))

dataset_fake_val = SimDataset(ds.utils.read_azure_filelist(video_test_path,
    sim_data_modes), data_root=data_root, gbuffers=g_buffers, inference=True,
    gbuf_mean=gbuf_stats['gbuf_mean'], gbuf_std=gbuf_stats['gbuf_std'], 
    )

def seed_worker(id):
    random.seed(torch.initial_seed() % np.iinfo(np.int32).max)
    np.random.seed(torch.initial_seed() % np.iinfo(np.int32).max)
    pass

collate_fn_val   = ds.EPEBatch.collate_fn

loader_fake = torch.utils.data.DataLoader(dataset_fake_val, \
    batch_size=1, shuffle=False, \
    num_workers=8, pin_memory=True, drop_last=False, worker_init_fn=seed_worker, collate_fn=collate_fn_val, prefetch_factor=6)

# %%
print(len(loader_fake))
# %%
example_input = None
for batch in loader_fake:
    # example_input = (example_batch.img, example_batch.gbuffers, example_batch.gt_labels)
    # example_input = {'img': example_batch.img, 'gbuffers': example_batch.gbuffers, 'gt_labels': example_batch.gt_labels}
    example_input = batch.as_input()
    break


# %%
for k in example_input.keys():
    print(k, example_input[k].shape)
# %%
traced_script_module = torch.jit.trace(generator, (example_input))
# %%
out = traced_script_module(example_input)
# %%
print(out.shape)
# %%

# Import the necessary packages
from torchvision import transforms


# Create the ToPILImage object
to_pil = transforms.ToPILImage()

# Apply the object to the tensor to convert it to a PIL image
pil_image = to_pil(out[0])
# %%
pil_image

# %%
traced_script_module.save("traced_resnet_model.pt")
# %%
