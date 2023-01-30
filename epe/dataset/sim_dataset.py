import logging
from pathlib import Path

import imageio
import numpy as np
from skimage.transform import resize
import scipy.io as sio
import torch
import os
import random
from tqdm import tqdm
from torchvision.transforms import Resize

from .batch_types import EPEBatch,ImageBatch
from .synthetic import SyntheticDataset
from .utils import mat2tensor, Frame

from .image_conversion import normal_to_normalised_normal, np_inverse_depth_invm_to_depth_m

from wayve.ai.lib.data import fetch_label, fetch_image
from wayve.ai.lib import undistort

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
	def __init__(self, frames, transform=None, gbuffers='fake', data_root='', shape=(600, 960), just_image=False, inference=False,
				gbuf_mean=None,
				gbuf_std=None, crop_undistortions=True):
		super(SimDataset, self).__init__('SIM')

		self.transform = transform
		self.gbuffers  = gbuffers
		self.data_root = data_root
		self.shape = shape
		self.inference = inference

		self._frames    = frames
		self._frame2id  = {f:i for i,f in enumerate(self._frames)}

		self._log.info(f'Found {len(self._frames)} samples.')

		self.resize = Resize(self.shape)

		self.gbuf_mean = gbuf_mean
		self.gbuf_std = gbuf_std
		self.gbuffer_loaders = {'normal': self.load_normal, 'depth': self.load_depth}

		self.crop_undistortions = crop_undistortions
		self.coords = (72, shape[0] - 72, 0, shape[1]) if crop_undistortions else None

		self.just_image = just_image

		self.order = 'hwc'

		# Hard coding sim distortion parameters as they are constant
		intrinsics=np.array([[1.13407e+03,0.00000e+00,1.02016e+03],
         [0.00000e+00,1.13520e+03,6.40390e+02],
         [0.00000e+00,0.00000e+00,1.00000e+00]], dtype=np.float32)

		distortion=np.array([-0.03952656, 0.0064852 ,-0.02202639, 0.00956348, 0.
		, 0, 0, 0, 0, 0, 0.  , 0, 0, 0, 2], dtype=np.float32)

		outrinsics = np.array([[360., 0., 480],
                   [  0., 360., 300],
                   [  0.,   0.,   1.]], dtype=np.float32)

		outrinsics = np.expand_dims(outrinsics, 0)
		intrinsics = np.expand_dims(intrinsics, 0)
		distortion = np.expand_dims(distortion, 0)

		self.intrinsics = torch.tensor(intrinsics)
		self.outrinsics = torch.tensor(outrinsics)
		self.distortion = torch.tensor(distortion)



	@property
	def num_gbuffer_channels(self):
		""" Number of image channels the provided G-buffers contain."""
		channels = 0
		for buffer in self.gbuffers:
			channels += {'depth': 1, 'normal':3}[buffer]

		return channels

	@property
	def num_classes(self):
		""" Number of classes in the semantic segmentation maps."""
		return 12

	@property
	def cls2gbuf(self):
		if self.gbuffers == 'all':
			# all: just handle sky class differently
			return {\
				0:lambda g:g[:,15:21,:,:]}
		else:
			return {}

	def get_id(self, img_filename):
		return self._frame2id.get(img_filename)


	def load_normal(self, frame):
		normal = fetch_label(run_id=frame.run_id, camera=frame.camera_id,
		timestamp=frame.timestamp, label_type='normal', image_source='sim')
		normal = normal.astype(np.float32)
		normal = normal[:, :, :3]
		normal = normal_to_normalised_normal(normal)
		normal = np.transpose(normal, (2, 0, 1))
		return normal
	
	def load_depth(self, frame):
		depth = fetch_label(run_id=frame.run_id, camera=frame.camera_id,
        timestamp=frame.timestamp, label_type='depth', image_source='sim')
		print(depth.max(), depth.min(), depth.dtype)
		# TODO check dtype
		depth = np_inverse_depth_invm_to_depth_m(depth)
		depth = np.transpose(depth, (2, 0, 1))
		return depth

	def load_segmentation(self, frame):
		return fetch_label(run_id=frame.run_id, camera=frame.camera_id,
        timestamp=frame.timestamp, label_type='segmentation', image_source='sim')

	def load_image(self, frame):
		# load image
		max_num_tries = 4
		for num_try in range(max_num_tries):
			try:
				img = fetch_image(run_id=frame.run_id, camera=frame.camera_id,
				timestamp=frame.timestamp, image_source='sim', order=self.order)
				img = img.astype(np.float32) / 255.0
				img = mat2tensor(img)

				# undistort
				img = img.unsqueeze(0)
				img = undistort(img, intrinsics=self.intrinsics, distortion=self.distortion, outrinsics=self.outrinsics)
				img = img.squeeze()
				img = img[:, :self.shape[0], :self.shape[1]]
				return img
			except:
				continue
		return None

	def crop(self, x):
		if x is None:
			return None
		return x[:, 72:-72]

	def __getitem__(self, index):
		index  = index % self.__len__()
		frame = self._frames[index]

		img = self.load_image(frame)
		if self.just_image:
			if self.crop_undistortions:
				img = self.crop(img)
			return ImageBatch(img, frame=frame)

		gbuffers = [self.gbuffer_loaders[g](frame) for g in self.gbuffers]
		gbuffers = np.concatenate(gbuffers, axis=0)
		gbuffers = torch.from_numpy(gbuffers.astype(np.float32)).float()

		segmentation = self.load_segmentation(frame)
		gt_labels = mat2tensor(material_from_gt_label(segmentation))

		if gt_labels is not None and torch.max(gt_labels) > 128:
			gt_labels = gt_labels / 255.0

		robust_labels = None
		if not self.inference:
			robust_labels = segmentation
			robust_labels = torch.LongTensor(robust_labels[:,:]).unsqueeze(0)

		if self.gbuf_mean is not None:
			gbuffers = center(gbuffers, self.gbuf_mean, self.gbuf_std)

		if self.crop_undistortions:
			img = self.crop(img)
			gbuffers = self.crop(gbuffers)
			gt_labels = self.crop(gt_labels)
			robust_labels = self.crop(robust_labels)

		return EPEBatch(img, gbuffers=gbuffers, gt_labels=gt_labels, robust_labels=robust_labels, frame=frame)

	def __len__(self):
		return len(self._frames)
