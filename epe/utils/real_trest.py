# %%
from epe.dataset.sim_dataset import SimDataset
from epe.dataset.image_datasets import ImageDataset
from epe.dataset.robust_labels import RobustlyLabeledDataset
import epe.dataset as ds
import os
from PIL import Image
from torchvision.transforms import ToPILImage
to_pil_image = ToPILImage()
import numpy as np
from wayve.ai.lib.data import fetch_label, fetch_image
from wayve.core.ai.segmentation.convert_colour_map import convert_colour_map_numpy
import matplotlib.cm as cm
from wayve.ai.lib.data import load_camera_calibrations
from wayve.ai.lib import undistort
import torch
from torchvision.transforms import ToPILImage
to_pil_image = ToPILImage()

# %%
fake_path = '/app/datasets/urban-driving/real_files.csv'
# %%
frames = ds.utils.read_filelist(fake_path) 
# %%
robust_dataset = RobustlyLabeledDataset('real', frames)
# %%
x = next(iter(robust_dataset))
img = x.img[0]
print(img.shape)
# %%
to_pil_image(img)
# %%
seg = x.robust_labels[0][0]
# %%
cmap = cm.get_cmap('jet', 28)
seg_col = Image.fromarray(np.uint8(cmap(seg)*255))
img_col = to_pil_image(img)
print(img.size)
print(seg_col.size)
Image.blend(img_col.convert('RGB'), seg_col.convert('RGB'), 0.2)
# %%


