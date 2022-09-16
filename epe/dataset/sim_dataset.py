import logging
from pathlib import Path

import imageio
import numpy as np
from skimage.transform import resize
import scipy.io as sio
import torch
from dataset.azure_loader import AzureImageLoader


from .batch_types import EPEBatch
from .synthetic import SyntheticDataset
from .utils import mat2tensor, normalize_dim

def center(x, m, s):
	x[0,:,:] = (x[0,:,:] - m[0]) / s[0]
	x[1,:,:] = (x[1,:,:] - m[1]) / s[1]
	x[2,:,:] = (x[2,:,:] - m[2]) / s[2]
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
	def __init__(self, paths, transform=None, gbuffers='fake'):
		"""


		paths -- list of tuples with (img_path, robust_label_path, gbuffer_path, gt_label_path)
		"""

		super(SimDataset, self).__init__('GTA')

		assert gbuffers in ['all', 'img', 'no_light', 'geometry', 'fake']

		self.transform = transform
		self.gbuffers  = gbuffers
		# self.shader    = class_type

		self._paths    = paths
		self._path2id  = {p[0].stem:i for i,p in enumerate(self._paths)}
		if self._log.isEnabledFor(logging.DEBUG):
			self._log.debug(f'Mapping paths to dataset IDs (showing first 30 entries):')
			for i,(k,v) in zip(range(30),self._path2id.items()):
				self._log.debug(f'path2id[{k}] = {v}')
				pass
			pass

		self.azure_loader = AzureImageLoader()

		try:
			data = np.load(Path(__file__).parent / 'pfd_stats.npz')
			# self._img_mean  = data['i_m']
			# self._img_std   = data['i_s']
			self._gbuf_mean = data['g_m']
			self._gbuf_std  = data['g_s']
			self._log.info(f'Loaded dataset stats.')
		except:
			# self._img_mean  = None
			# self._img_std   = None
			self._gbuf_mean = None
			self._gbuf_std  = None
			pass

		self._log.info(f'Found {len(self._paths)} samples.')
		pass


	@property
	def num_gbuffer_channels(self):
		""" Number of image channels the provided G-buffers contain."""
		return {'fake':32, 'all':26, 'img':0, 'no_light':17, 'geometry':8}[self.gbuffers]


	@property
	def num_classes(self):
		""" Number of classes in the semantic segmentation maps."""
		return {'fake':12, 'all':12, 'img':0, 'no_light':0, 'geometry':0}[self.gbuffers]


	@property
	def cls2gbuf(self):
		if self.gbuffers == 'all':
			# all: just handle sky class differently
			return {\
				0:lambda g:g[:,15:21,:,:]}
		else:
			return {}


	def get_id(self, img_filename):
		return self._path2id.get(Path(img_filename).stem)


	def __getitem__(self, index):

		index  = index % self.__len__()
		img_path, robust_label_path, gbuffer_path, gt_label_path = self._paths[index]

		if not gbuffer_path.exists():
			self._log.error(f'Gbuffers at {gbuffer_path} do not exist.')
			raise FileNotFoundError
			pass

		data = np.load(gbuffer_path)

		if self.gbuffers == 'fake':
			img       = mat2tensor(imageio.imread(img_path).astype(np.float32) / 255.0)
			gbuffers  = mat2tensor(data['data'].astype(np.float32))
			gt_labels = material_from_gt_label(imageio.imread(gt_label_path))
			if gt_labels.shape[0] != img.shape[-2] or gt_labels.shape[1] != img.shape[-1]:
				gt_labels = resize(gt_labels, (img.shape[-2], img.shape[-1]), anti_aliasing=True, mode='constant')
			gt_labels = mat2tensor(gt_labels)
			pass
		else:
			img       = mat2tensor(data['img'].astype(np.float32) / 255.0)
			gbuffers  = mat2tensor(data['gbuffers'].astype(np.float32))
			gt_labels = mat2tensor(data['shader'].astype(np.float32))
			pass

		if torch.max(gt_labels) > 128:
			gt_labels = gt_labels / 255.0
			pass

		if self._gbuf_mean is not None:
			gbuffers = center(gbuffers, self._gbuf_mean, self._gbuf_std)
			pass

		if not robust_label_path.exists():
			self._log.error(f'Robust labels at {robust_label_path} do not exist.')
			raise FileNotFoundError
			pass

		robust_labels = imageio.imread(robust_label_path)
		robust_labels = torch.LongTensor(robust_labels[:,:]).unsqueeze(0)

		return EPEBatch(img, gbuffers=gbuffers, gt_labels=gt_labels, robust_labels=robust_labels, path=img_path, coords=None)


	def __len__(self):
		return len(self._paths)


        {
            "id": 0,
            "name": "Person",
            "colour": "0xdc143c",
            "hasInstance": true,
            "dynamic_class": 0,
            "static_class": 255,
            "sim_meta": {
                "has_bbox_label": true
            },
            "conversion": {
                "cityscapes": 11,
                "wayve": 11,
                "wayve_v2": 0,
                "dreamer_v1": 5,
                "wayve_bev": 7,
                "universal": 125 
            }
        },
        {
            "id": 1,
            "name": "Rider",
            "colour": "0xff0000",
            "hasInstance": true,
            "dynamic_class": 1,
            "static_class": 255,
            "conversion": {
                "cityscapes": 12,
                "wayve": 11,
                "wayve_v2": 1,
                "dreamer_v1": 5,
                "wayve_bev": 255,
                "universal": 127 
            }
        },
        {
            "id": 2,
            "name": "Animal",
            "colour": "0xa52a2a",
            "hasInstance": true,
            "dynamic_class": 2,
            "static_class": 255,
            "conversion": {
                "cityscapes": 255,
                "wayve": 11,
                "wayve_v2": 2,
                "dreamer_v1": 5,
                "wayve_bev": 255,
                "universal": 16
            }
        },
        {
            "id": 3,
            "name": "Car",
            "colour": "0x00008e",
            "hasInstance": true,
            "dynamic_class": 3,
            "static_class": 255,
            "sim_meta": {
                "has_bbox_label": true
            },
            "conversion": {
                "cityscapes": 13,
                "wayve": 12,
                "wayve_v2": 3,
                "dreamer_mini": 2,
                "dreamer_v1": 0,
                "wayve_bev": 0,
                "universal": 176
            }
        },
        {
            "id": 4,
            "name": "Large_vehicle",
            "colour": "0x00008e",
            "hasInstance": true,
            "dynamic_class": 4,
            "static_class": 255,
            "conversion": {
                "cityscapes": 14,
                "wayve": 12,
                "wayve_v2": 4,
                "dreamer_v1": 5,
                "wayve_bev": 3,
                "universal": 182
            }
        },
        {
            "id": 5,
            "name": "Bus",
            "colour": "0x003c64",
            "hasInstance": true,
            "dynamic_class": 5,
            "static_class": 255,
            "conversion": {
                "cityscapes": 15,
                "wayve": 12,
                "wayve_v2": 5,
                "dreamer_v1": 5,
                "wayve_bev": 1,
                "universal": 180
            }
        },
        {
            "id": 6,
            "name": "Motorcycle",
            "colour": "0x0000e6",
            "hasInstance": true,
            "dynamic_class": 6,
            "static_class": 255,
            "conversion": {
                "cityscapes": 17,
                "wayve": 13,
                "wayve_v2": 6,
                "dreamer_v1": 5,
                "wayve_bev": 5,
                "universal": 178

            }
        },
        {
            "id": 7,
            "name": "Bicycle",
            "colour": "0x770b20",
            "hasInstance": true,
            "dynamic_class": 7,
            "static_class": 255,
            "conversion": {
                "cityscapes": 18,
                "wayve": 13,
                "wayve_v2": 7,
                "dreamer_v1": 5,
                "wayve_bev": 4,
                "universal": 175
            }
        },
        {
            "id": 8,
            "name": "Traffic_sign",
            "colour": "0xdcdc00",
            "hasInstance": true,
            "dynamic_class": 255,
            "static_class": 8,
            "conversion": {
                "cityscapes": 7,
                "wayve": 7,
                "wayve_v2": 8,
                "dreamer_v1": 5,
                "wayve_bev": 255,
                "universal": 135
            }
        },
        {
            "id": 9,
            "name": "Traffic_light",
            "colour": "0xfaaa1e",
            "hasInstance": true,
            "dynamic_class": 255,
            "static_class": 9,
            "sim_meta": {
                "has_bbox_label": true
            },
            "conversion": {
                "cityscapes": 6,
                "wayve": 6,
                "wayve_v2": 9,
                "dreamer_v1": 5,
                "wayve_bev": 255,
                "universal": 136
            }
        },
        {
            "id": 10,
            "name": "Driveable_road",
            "colour": "0x804080",
            "hasInstance": false,
            "dynamic_class": 255,
            "static_class": 10,
            "conversion": {
                "cityscapes": 0,
                "wayve": 0,
                "dreamer_mini": 0,
                "wayve_v2": 10,
                "dreamer_v1": 1,
                "wayve_bev": 9,
                "universal": 98
            }
        },
        {
            "id": 11,
            "name": "Sidewalk_or_footpath",
            "colour": "0xf423e8",
            "hasInstance": false,
            "dynamic_class": 255,
            "static_class": 11,
            "conversion": {
                "cityscapes": 1,
                "wayve": 2,
                "wayve_v2": 11,
                "dreamer_v1": 2,
                "wayve_bev": 10,
                "universal": 100
            }
        },
        {
            "id": 12,
            "name": "Curb_or_elevated_part_of_traffic_island",
            "colour": "0xc4c4c4",
            "hasInstance": false,
            "dynamic_class": 255,
            "static_class": 12,
            "conversion": {
                "cityscapes": 255,
                "wayve": 255,
                "wayve_v2": 12,
                "dreamer_v1": 3,
                "wayve_bev": 12,
                "universal": 100
            }
        },
        {
            "id": 13,
            "name": "Parking",
            "colour": "0xfaaaa0",
            "hasInstance": false,
            "dynamic_class": 255,
            "static_class": 13,
            "conversion": {
                "cityscapes": 0,
                "wayve": 0,
                "wayve_v2": 13,
                "dreamer_v1": 5,
                "wayve_bev": 9,
                "universal": 194
            }
        },
        {
            "id": 14,
            "name": "Speedbump_or_manhole_or_pothole",
            "colour": "0xaaaaaa",
            "hasInstance": false,
            "dynamic_class": 255,
            "static_class": 14,
            "conversion": {
                "cityscapes": 0,
                "wayve": 0,
                "wayve_v2": 14,
                "dreamer_v1": 5,
                "wayve_bev": 9,
                "universal": 98
            }
        },
        {
            "id": 15,
            "name": "Pedestrian_crossing",
            "colour": "0x8c8cc8",
            "hasInstance": false,
            "dynamic_class": 255,
            "static_class": 15,
            "conversion": {
                "cityscapes": 0,
                "wayve": 0,
                "wayve_v2": 15,
                "dreamer_v1": 5,
                "wayve_bev": 9,
                "universal": 98
            }
        },
        {
            "id": 16,
            "name": "Road_marking",
            "colour": "0xffffff",
            "hasInstance": false,
            "dynamic_class": 255,
            "static_class": 16,
            "conversion": {
                "cityscapes": 0,
                "wayve": 1,
                "wayve_v2": 16,
                "dreamer_v1": 5,
                "wayve_bev": 11,
                "universal": 98
            }
        },
        {
            "id": 17,
            "name": "Restricted_lane",
            "colour": "0x8040ff",
            "hasInstance": false,
            "dynamic_class": 255,
            "static_class": 17,
            "conversion": {
                "cityscapes": 0,
                "wayve": 0,
                "wayve_v2": 17,
                "dreamer_v1": 5,
                "wayve_bev": 9,
                "universal": 98
            }
        },
        {
            "id": 18,
            "name": "Building_or_wall_or_bridge_or_tunnel",
            "colour": "0x464646",
            "hasInstance": false,
            "dynamic_class": 255,
            "static_class": 18,
            "conversion": {
                "cityscapes": 2,
                "wayve": 3,
                "wayve_v2": 18,
                "dreamer_v1": 5,
                "wayve_bev": 13,
                "universal": 35
            }
        },
        {
            "id": 19,
            "name": "Traffic_cone",
            "colour": "0xc5e79e",
            "hasInstance": false,
            "dynamic_class": 255,
            "static_class": 19,
            "conversion": {
                "cityscapes": 255,
                "wayve": 255,
                "wayve_v2": 19,
                "dreamer_v1": 5,
                "wayve_bev": 255,
                "universal": 194
            }
        },
        {
            "id": 20,
            "name": "Fence_or_guard_rail",
            "colour": "0xbe9999",
            "hasInstance": false,
            "dynamic_class": 255,
            "static_class": 20,
            "conversion": {
                "cityscapes": 4,
                "wayve": 4,
                "wayve_v2": 20,
                "dreamer_v1": 5,
                "wayve_bev": 255,
                "universal": 144
            }
        },
        {
            "id": 21,
            "name": "Other_street_furniture",
            "colour": "0xffff80",
            "hasInstance": false,
            "dynamic_class": 255,
            "static_class": 21,
            "conversion": {
                "cityscapes": 255,
                "wayve": 255,
                "wayve_v2": 21,
                "dreamer_v1": 5,
                "wayve_bev": 255,
                "universal": 194
            }
        },
        {
            "id": 22,
            "name": "Vegetation",
            "colour": "0x6b8e23",
            "hasInstance": false,
            "dynamic_class": 255,
            "static_class": 22,
            "conversion": {
                "cityscapes": 8,
                "wayve": 8,
                "wayve_v2": 22,
                "dreamer_v1": 5,
                "wayve_bev": 255,
                "universal": 174
            }
        },
        {
            "id": 23,
            "name": "Terrain",
            "colour": "0x40aa40",
            "hasInstance": false,
            "dynamic_class": 255,
            "static_class": 23,
            "conversion": {
                "cityscapes": 9,
                "wayve": 9,
                "wayve_v2": 23,
                "dreamer_v1": 5,
                "wayve_bev": 13,
                "universal": 102
            }
        },
        {
            "id": 24,
            "name": "Sky",
            "colour": "0x4682b4",
            "hasInstance": false,
            "dynamic_class": 255,
            "static_class": 255,
            "conversion": {
                "cityscapes": 10,
                "wayve": 10,
                "wayve_v2": 24,
                "dreamer_v1": 4,
                "wayve_bev": 255,
                "universal": 142
            }
        },
        {
            "id": 25,
            "name": "Ego_car",
            "colour": "0x4e63ab",
            "hasInstance": false,
            "dynamic_class": 255,
            "static_class": 255,
             "sim_meta": {
                "has_bbox_label": true
            },
           "conversion": {
                "cityscapes": 13,
                "wayve": 12,
                "wayve_v2": 3,
                "dreamer_mini": 2,
                "dreamer_v1": 0,
                "wayve_bev": 8,
                "universal": 176
            }
        }
    ]
}
