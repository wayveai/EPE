# %%
import epe.dataset as ds
from epe.dataset.sim_dataset import SimDataset
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import wandb
wandb.init(project='EPE', mode="disabled")
# %%


sim_data_modes = ['rgb', 'segmentation', 'seg_robust', 'depth', 'normal']
real_data_modes = ['rgb', 'seg_robust']
real_data_root = '/home/kacper/data/datasets'
sim_data_root = '/home/kacper/data/datasets'
g_buffers = 'depth'
fake_path = '/home/kacper/data/EPE/somers_town/sim_files.csv'

# validation
dataset_fake = SimDataset(ds.utils.read_azure_filelist(fake_path, sim_data_modes), data_root=sim_data_root, gbuffers=g_buffers)
# %%
x = dataset_fake[8]
print(x.gbuffers)
x.gbuffers.mean()
print(x.img.shape)
# %%
print(x.gbuffers.min(), x.gbuffers.max())
for i in range(x.gbuffers.shape[1]):
    v = x.gbuffers[:, i]
    print(v.mean(), v.std(), v.min(), v.max())

# %%
black_images = []
for i in tqdm(range(len(dataset_fake)), total=len(dataset_fake)):
    x = dataset_fake[i]
    if torch.eq(x.img, torch.zeros_like(x.img)).all():
        black_images.append(x.path)

print(len(black_images))
# %%
run_ids = set()
for x in black_images:
    run_id = '/'.join(x.split('/')[:-3])
    # print(run_id)
    run_ids.add(run_id)

print(len(run_ids))
# %%
counter = 0
for run_id in run_ids:
    dir = os.path.join(sim_data_root, run_id, 'cameras', 'front-forward--rgb')
    counter += len(os.listdir(dir))
    print(len(os.listdir(dir)))

print("counter: ", counter)
print("len black images: ", len(black_images))
# %%
print(run_ids)

# %%

# with open('/home/kacper/black_sim.txt')
# %%

# # %%
# max = 0
# for x in tqdm(dataset_fake):
#     max = torch.max(x.gbuffers) if torch.max(x.gbuffers) > max else max
# print('max: ', max.item())

# # %%
# max = 0
# for x in tqdm(dataset_fake):
#     max = torch.max(x.gbuffers) if torch.max(x.gbuffers) > max else max
# print('clipped max: ', max.item())

# # %%
# mean_acc = 0
# std_acc = 0
# for x in tqdm(dataset_fake):
#     mean_acc += torch.mean(x.gbuffers/max)
#     std_acc += torch.std(x.gbuffers/max)
#     break

# mean = mean_acc / len(dataset_fake)
# std = std_acc / len(dataset_fake)

# print('mean: ', mean.item())
# print('std: ', std.item())

# # %%
# x = dataset_fake[0]
# values, counts = np.unique(x.gbuffers.numpy(), return_counts=True)
# plt.plot(values, counts)
# plt.show()
# # %%
# # %%
# x = dataset_fake[0]
# values, counts = np.unique(((x.gbuffers - mean) / std).numpy(), return_counts=True)
# plt.plot(values, counts)
# plt.show()

# # %%

# %%
