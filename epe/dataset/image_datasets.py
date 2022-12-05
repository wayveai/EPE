import logging
import os
from pathlib import Path
import random
from turtle import shape

import imageio
import numpy as np
import scipy.io as sio
import torch
import torch.utils.data

from .batch_types import ImageBatch
from .utils import mat2tensor
from epe.dataset.azure_loader import AzureImageLoader
from torchvision.transforms import Resize

logger = logging.getLogger(__file__)

class ImageDataset(torch.utils.data.Dataset):
	def __init__(self, name, img_paths, transform=None, shape=(600, 960), data_root=''):
		"""

		name -- Name used for debugging, log messages.
		img_paths - an iterable of paths to individual image files. Only JPG and PNG files will be taken.
		transform -- Transform to be applied to images during loading.
		"""

		img_paths  = [Path(p[0] if type(p) is tuple else p) for p in img_paths]
		self.paths = sorted([p for p in img_paths if p.suffix in ['.jpg', '.jpeg', '.png']])
		
		self._path2id    = {p.stem:i for i,p in enumerate(self.paths)}
		self.transform   = transform

		self.azure_loader = AzureImageLoader()
		self.shape = shape
		
		self.name = name
		self._log = logging.getLogger(f'epe.dataset.{name}')
		self._log.info(f'Found {len(self.paths)} images.')
		self.data_root = data_root

		self.resize = Resize(shape)
		pass


	def _load_img(self, path):
		path = str(path)
		try:
			local_path = os.path.join(self.data_root, path)
			if os.path.exists(local_path):
				path = local_path
				img = imageio.imread(path).astype(np.float32)

			else:
				img = np.array(self.azure_loader.load_img_from_path_and_resize(path, *self.shape))
			# if len(img.shape) == 2:
			# 	img = np.expand_dims(img, 2)
			if '.jpeg' in path or '.jpg' in path:
				img = np.clip(img / 255.0, 0.0, 1.0)
			return img
		except:
			logging.exception(f'Failed to load {path}.')
			raise
		pass


	def get_id(self, path):
		return self._path2id.get(Path(path))


	def __getitem__(self, index):
		
		idx  = index % self.__len__()
		path = self.paths[idx]
		img  = self._load_img(path)

		if self.transform is not None:
			img = self.transform(img)
			pass

		img = mat2tensor(img)   
		# TODO: reformat for a cleaner way of resizing images when loading
		# img = self.resize(img)
		return ImageBatch(img, path)


	def __len__(self):
		return len(self.paths)
