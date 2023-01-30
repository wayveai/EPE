import logging

import imageio
import torch

from .batch_types import EPEBatch
from .image_datasets import ImageDataset
from .utils import mat2tensor

from wayve.ai.lib.data import fetch_label

logger = logging.getLogger('epe.dataset.robust')

class RobustlyLabeledDataset(ImageDataset):
	def __init__(self, name, frames, img_transform=None, label_transform=None, shape=(600, 960)):
		""" Create an image dataset with robust labels.

		name -- Name of dataset, used for debug output and finding corresponding sampling strategy
		img_and_robust_label_frames -- Iterable of tuple containing image frame and corresponding frame to robust label map. Assumes that filenames are unique!
		img_transform -- Transform (func) to apply to image during loading
		label_transform -- Transform (func) to apply to robust label map during loading
		"""
		self._log = logging.getLogger(f'epe.dataset.{name}')

		self.shape = shape

		self.frames = frames
		self.frame2id  = {f:i for i,f in enumerate(self.frames)}
			
		self.transform       = img_transform
		self.label_transform = label_transform
		self.name            = name

		self._log.info(f'Found {len(self.frames)} images.')
		if len(self.frames) < 1:
			self._log.warn('Dataset is empty!')
			pass
		pass


	def get_id(self, img_filename):
		""" Get dataset ID for sample given img_filename."""
		return self.frame2id.get(img_filename)

	def load_label(self, frame):
		return fetch_label(run_id=frame.run_id, camera=frame.camera_id, timestamp=frame.timestamp, label_type='segmentation')

	def __getitem__(self, index):
		
		idx      = index % self.__len__()
		frame = self.frames[idx]
		img      = self._load_img(frame)

		if self.transform is not None:
			img = self.transform(img)
			pass

		img = mat2tensor(img)

		robust_labels = self.load_label(frame)

		if self.label_transform is not None:
			robust_labels = self.label_transform(robust_labels)
			pass

		robust_labels = torch.LongTensor(robust_labels).unsqueeze(0)

		return EPEBatch(img, frame=frame, robust_labels=robust_labels)
	pass

