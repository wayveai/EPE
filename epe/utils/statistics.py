# %%
import epe.dataset as ds
from epe.dataset.sim_dataset import SimDataset
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import random

from epe.utils.dataset_collector import uniform_sampling
# %%

g_buffers = ['depth', 'normal']
sim_data_modes = ['rgb', 'segmentation', 'segmentation-mseg', *g_buffers]
real_data_modes = ['rgb', 'segmentation-mseg']
data_root = '/mnt/remote/data/users/kacper/datasets'
dataset_meta_path = '/mnt/remote/data/users/kacper/datasets/somers-town_weather_v0'
fake_path = os.path.join(dataset_meta_path, 'sim_files.csv')

# %%
# uniform_sampling(fake_path, 1, data_root=dataset_meta_path, car_filter=lambda x: 'ningaloo' in x, )

# %%


# validation
dataset_fake = SimDataset(ds.utils.read_azure_filelist(fake_path, sim_data_modes), data_root=data_root, gbuffers=g_buffers)

def compute_gbuffer_statistics(ds):
    indices = list(range(len(ds)))
    random.shuffle(indices)
    acc_std = 0
    acc_mean = 0
    counter = 0

    print('Computing gbuffer statistics...')
    for i in tqdm(indices[:1000]):
        batch = ds[i]
        gbuffers = batch.gbuffers
        std, mean = torch.std_mean(gbuffers, (0, 2, 3))
        acc_std += std
        acc_mean += mean
        counter += 1

    gbuf_mean = acc_mean / counter
    gbuf_std  = acc_std / counter
    return gbuf_mean, gbuf_std

# %%
gbuf_mean, gbuf_std = compute_gbuffer_statistics(dataset_fake)
print(gbuf_mean)
print(gbuf_std)
# %%
gbuf_stats = {'gbuf_mean': gbuf_mean, 'gbuf_std': gbuf_std}
torch.save(gbuf_stats, os.path.join(dataset_meta_path, 'gbuf_stats.pt'))
# %%
