# %%
from epe.dataset.sim_dataset import SimDataset
import epe.dataset as ds
import os
from PIL import Image
from torchvision.transforms import ToPILImage
to_pil_image = ToPILImage()
import numpy as np
# %%
fake_path = '/app/datasets/urban-driving/sim_files.csv'
# %%
g_buffers = ['depth', 'normal']
frames = ds.utils.read_filelist(fake_path) 
dataset_fake = SimDataset(frames, gbuffers=g_buffers, crop_undistortions=True)
# %%
frame = frames[0]
frame.camera_id = 'front-right-rightward'
frame.camera_id = 'front-forward'

# %%
x = next(iter(dataset_fake))

# %%
img = x['img'][0]
print(img.shape, img.min(), img.max())
# %%
rgb = to_pil_image(img)
# %%
# %%
normal = to_pil_image(x.gbuffers[0, 1:])
# %%
# permute x.gbuffers from cwh to hwc
Image.blend(rgb, normal, 0.2)
# %%
