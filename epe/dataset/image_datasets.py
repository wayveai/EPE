import logging
import os
import random

import imageio
import numpy as np
import scipy.io as sio
import torch
import torch.utils.data

from .batch_types import ImageBatch
from .utils import mat2tensor
from wayve.ai.lib import undistort
from wayve.ai.lib.data import load_camera_calibrations
from wayve.ai.lib.data import fetch_image

logger = logging.getLogger(__file__)

class ImageDataset(torch.utils.data.Dataset):
	def __init__(self, name, frames, transform=None, shape=(600, 960), data_root=''):
		"""

		name -- Name used for debugging, log messages.
		frames - list of frames to load.
		transform -- Transform to be applied to images during loading.
		"""

		self.frames = frames
		
		self.frame2id  = {f:i for i,f in enumerate(self.frames)}
		self.transform   = transform

		self.shape = shape
		
		self.name = name
		self._log = logging.getLogger(f'epe.dataset.{name}')
		self._log.info(f'Found {len(self.frames)} images.')
		self.data_root = data_root

	def _load_img(self, frame):
		img = fetch_image(run_id=frame.run_id, camera=frame.camera_id, timestamp=frame.timestamp, order='hwc', write_to_cache=False)
		calibration = load_camera_calibrations(
			run_id=frame.run_id,
			camera_names=(frame.camera_id,),
		)

		img = torch.Tensor(np.moveaxis(img, 2, 0))
		# undistort currently only works on batches of imgs
		img = img.unsqueeze(0)

		outrinsics = torch.tensor(
		[
			[360, 0, 480],
			[0, 360, 300],
			[0, 0.0, 1.0],
		]
		).broadcast_to(1, 3, 3)

		img = undistort(img, intrinsics=torch.Tensor(calibration.intrinsics), distortion=torch.Tensor(calibration.distortion), outrinsics=outrinsics)

		img = img[:, :, :600, :960]
		img = img[:, :, 76:76+448, :960]

		# format back for dispay
		img = img[0].numpy()
		img = np.moveaxis(img, 0, 2).astype(np.float32) / 255.0
		return img


	def get_id(self, frame):
		return self.frame2id.get(frame)

	def __getitem__(self, index):
		
		idx  = index % self.__len__()
		frame = self.frames[idx]
		img  = self._load_img(frame)

		if self.transform is not None:
			img = self.transform(img)
			pass

		img = mat2tensor(img)

		return ImageBatch(img, frame=frame)


	def __len__(self):
		return len(self.frames)

