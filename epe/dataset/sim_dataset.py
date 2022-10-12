import logging
from pathlib import Path

import imageio
import numpy as np
from skimage.transform import resize
import scipy.io as sio
import torch
from epe.dataset.azure_loader import AzureImageLoader
import os
import random
import wandb
from tqdm import tqdm
from torchvision.transforms import Resize


from .batch_types import EPEBatch
from .synthetic import SyntheticDataset
from .utils import mat2tensor, normalize_dim

from .image_conversion import normal_to_normalised_normal, np_inverse_depth_invm_to_depth_m_normalized, np_inverse_depth_normalized_to_depth_m

def center(x, m, s):
	for i in range(x.shape[0]):
		x[i,:,:] = (x[i,:,:] - m[i]) / s[i]
	return x


def material_from_gt_label(gt_labelmap):
	""" Merges several classes. Using wayve_v3 classes """

	h,w = gt_labelmap.shape
	shader_map = np.zeros((h, w, 12), dtype=np.float32)
	shader_map[:,:,0] = (gt_labelmap == 24).astype(np.float32) # sky
	shader_map[:,:,1] = (np.isin(gt_labelmap, [10, 11, 12, 13, 13, 15, 16, 17])).astype(np.float32) # Driveable_road,  Sidewalk_or_footpath,  Curb_or_elevated_part_of_traffic_island,  Parking,  Parking,  Pedestrian_crossing,  Road_marking,  Restricted_lane 
	shader_map[:,:,2] = (np.isin(gt_labelmap, [3, 4, 5, 6, 7])).astype(np.float32) # Car, Large_vehicle, Bus, Motorcycle, Bicycle
	shader_map[:,:,3] = (gt_labelmap == 23).astype(np.float32) # Terrain, 
	shader_map[:,:,4] = (gt_labelmap == 22).astype(np.float32) # vegetation
	shader_map[:,:,5] = (np.isin(gt_labelmap, [0 , 1])).astype(np.float32) # Person, Rider, 
	shader_map[:,:,6] = (np.isin(gt_labelmap, [19, 21])).astype(np.float32) # Traffic_cone, Other_street_furniture, 
	shader_map[:,:,7] = (gt_labelmap == 9).astype(np.float32) # Traffic_light, 
	shader_map[:,:,8] = (gt_labelmap == 8).astype(np.float32) # Traffic_sign, 
	shader_map[:,:,9] = (gt_labelmap == 25).astype(np.float32) # Ego_car, 
	shader_map[:,:,10] = (np.isin(gt_labelmap, [18, 20])).astype(np.float32) # Building_or_wall_or_bridge_or_tunnel, Fence_or_guard_rail, 
	shader_map[:,:,11] = (np.isin(gt_labelmap, [2])).astype(np.float32) # unlabeled: animal
	return shader_map


class SimDataset(SyntheticDataset):
	def __init__(self, paths, transform=None, gbuffers='fake', data_root='', shape=(600, 960), mean=None, std=None):
		"""


		# Note that last arguments will be gbuffer PATHS! to allow for flexibility
		paths -- list of tuples with (img_path, robust_label_path, gt_label_path, gbuffer_paths)
		"""

		super(SimDataset, self).__init__('GTA')

		# assert gbuffers in ['all', 'img', 'no_light', 'geometry', 'fake', 'depth']

		self.transform = transform
		self.gbuffers  = gbuffers
		self.data_root = data_root
		self.shape = shape

		# self.shader    = class_type

		self._paths    = paths
		self._path2id  = {Path(p[0]).stem:i for i,p in enumerate(self._paths)}
		if self._log.isEnabledFor(logging.DEBUG):
			self._log.debug(f'Mapping paths to dataset IDs (showing first 30 entries):')
			for i,(k,v) in zip(range(30),self._path2id.items()):
				self._log.debug(f'path2id[{k}] = {v}')
				pass
			pass

		self.azure_loader = AzureImageLoader()

		self._log.info(f'Found {len(self._paths)} samples.')

		self.resize = Resize(self.shape)

		self.gbuf_mean = mean
		self.gbuf_std  = std
		if mean is None or std is None:
			self.compute_gbuffer_statistics()





	@property
	def num_gbuffer_channels(self):
		""" Number of image channels the provided G-buffers contain."""
		channels = 0
		for buffer in self.gbuffers:
			channels += {'depth': 1, 'normal':4}[buffer]

		return channels
		# return {'fake':32, 'all':26, 'img':0, 'no_light':17, 'geometry':8, 'depth': 1}[self.gbuffers]


	@property
	def num_classes(self):
		""" Number of classes in the semantic segmentation maps."""
		return 12
		# return {'fake':12, 'all':12, 'img':0, 'no_light':0, 'geometry':0, 'depth': 12}[self.gbuffers]


	@property
	def cls2gbuf(self):
		if self.gbuffers == 'all':
			# all: just handle sky class differently
			return {\
				0:lambda g:g[:,15:21,:,:]}
		else:
			return {}

	def compute_gbuffer_statistics(self):
		indices = list(range(self.__len__()))
		random.shuffle(indices)
		acc_std = 0
		acc_mean = 0
		counter = 0

		print('Computing gbuffer statistics...')
		for i in tqdm(indices[:1000]):
			batch = self[i]
			gbuffers = batch.gbuffers
			std, mean = torch.std_mean(gbuffers, (0, 2, 3))
			acc_std += std
			acc_mean += mean
			counter += 1

		self.gbuf_mean = acc_mean / counter
		self.gbuf_std  = acc_std / counter

		data = zip(self.gbuf_mean.tolist(), self.gbuf_std.tolist())
		data = list(data)
		table = wandb.Table(columns=["mean", "std"], data=data)
		wandb.log({'mean_std': table})



	def get_id(self, img_filename):
		return self._path2id.get(Path(img_filename).stem)

	def load_file(self, path):
		local_path = os.path.join(self.data_root, path)
		if os.path.exists(local_path):
			img = imageio.imread(local_path)

			if '--depth' in local_path:
				img = img.astype(np.uint16)
			else:
				img = img.astype(np.float32)
		else:
			img = np.array(self.azure_loader.load_img_from_path_and_resize(path, *self.shape))

		return img


	def __getitem__(self, index):
		# TODO: check if either local or azure file exists, else skip

		index  = index % self.__len__()
		img_path, gt_label_path, robust_label_path = self._paths[index][:3]
		g_buffer_paths = self._paths[index][3:]


		g_data = []
		for g_buffer_path in g_buffer_paths:
			buffer = self.load_file(g_buffer_path)
			if '--depth' in g_buffer_path:
				buffer = np.array(buffer, dtype=np.uint16)
				buffer = np_inverse_depth_normalized_to_depth_m(buffer)
				buffer = np.expand_dims(buffer, axis=-1)
				# buffer = (np.clip(buffer, 0, max_depth_m) / max_depth_m) * 2 - 1
			if '--normal' in g_buffer_path:
				buffer = normal_to_normalised_normal(buffer)

			buffer = np.transpose(buffer, (2, 0, 1))
			g_data.append(buffer)
		# TODO Double check the dimensions of this
		g_data = np.concatenate(g_data)

		img = mat2tensor(self.load_file(img_path).astype(np.float32) / 255.0)
		img = self.resize(img)
		gbuffers = torch.from_numpy(g_data.astype(np.float32)).float()
		gt_labels = mat2tensor(material_from_gt_label(self.load_file(gt_label_path)))
		# gt_labels = mat2tensor(self.load_file(gt_label_path).astype(np.float32))

		if torch.max(gt_labels) > 128:
			gt_labels = gt_labels / 255.0
			pass

		if self.gbuf_mean is not None:
			gbuffers = center(gbuffers, self.gbuf_mean, self.gbuf_std)
			pass

		robust_labels = self.load_file(robust_label_path)
		robust_labels = torch.LongTensor(robust_labels[:,:]).unsqueeze(0)

		return EPEBatch(img, gbuffers=gbuffers, gt_labels=gt_labels, robust_labels=robust_labels, path=img_path, coords=None)


	def __len__(self):
		return len(self._paths)
