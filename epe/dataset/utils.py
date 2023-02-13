from collections import defaultdict
import csv
import logging
from pathlib import Path

import numpy as np
from skimage.transform import rescale
import torch
from tqdm import tqdm
import os

logger = logging.getLogger('epe.dataset.utils')


class Frame():
    def __init__(self, run_id, timestamp, camera_id='front-forward', is_sim=None, root_path=''):
        self.run_id = run_id
        self.timestamp = int(timestamp)
        self.camera_id = camera_id
        self.root_path = root_path

        if not is_sim:
            self.is_sim = 'ningaloo' in self.run_id
        else:
            self.is_sim = is_sim

    def __hash__(self) -> int:
        return hash((self.run_id, self.timestamp, self.camera_id))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Frame):
            return False
        return self.run_id == other.run_id and self.timestamp == other.timestamp and self.camera_id == other.camera_id

    def get_path(self, mode='rgb'):
        if not self.is_sim and mode == 'rgb':
            camera_id = self.camera_id
        else:
            camera_id = self.camera_id + f'--{mode}'

        ext = defaultdict(lambda: 'png')
        ext['rgb'] = 'jpeg'

        file_name = str(self.timestamp).zfill(16) + 'unixus.' + ext[mode]
        return os.path.join(self.root_path, self.run_id, 'cameras', camera_id, file_name)

    def __str__(self) -> str:
        return f'{self.run_id},{self.camera_id},{self.timestamp}'


def read_filelist(filelist_path):
    frames = []
    with open(filelist_path) as file:
        for i, line in enumerate(file):
            t = line.strip().split(',')
            if len(t) == 2:
                run_id = t[0]
                timestamp = t[1]
                frame = Frame(run_id, timestamp)
            elif len(t) == 3:
                run_id = t[0]
                camera_id = t[1]
                timestamp = t[2]
                frame = Frame(run_id, timestamp, camera_id)
            else:
                raise IOError()
            frames.append(frame)
    return frames


def read_azure_filelist(path_to_filelist, modes=['rgb'], dataset_name='urban-driving', is_sim=None, camera_id='front-forward'):
    paths = []
    with open(path_to_filelist) as file:
        for i, line in enumerate(file):
            t = line.strip().split(',')
            if len(t) == 2:
                run_id = t[0]
                timestamp = t[1]
            elif len(t) == 3:
                run_id = t[0]
                camera_id = t[1]
                timestamp = t[2]
            else:
                raise IOError()

            if is_sim is None:
                is_sim = 'ningaloo' in run_id

            ps = []
            for mode in modes:
                ext = 'jpeg' if mode in ['rgb'] else 'png'

                if mode == 'rgb':
                    camera = f'{camera_id}--rgb' if is_sim else camera_id
                else:
                    camera = f'{camera_id}--{mode}'
                ps.append(os.path.join(dataset_name, run_id,
                          'cameras', camera, f'{timestamp}unixus.{ext}'))

            paths.append(tuple(ps))
    return paths


def load_crops(path):
    """ Load crop info from a csv file.

    The file is expected to have columns run_id,camera,ts,r0,r1,c0,c1
    run_id -- run_id
    camera -- camera id name (e.g. 'front-forward')
    ts -- timestamp
    r0 -- top y coordinate
    r1 -- bottom y coordinate
    c0 -- left x coordinate
    c1 -- right x coordinate
    """

    path = Path(path)

    if not path.exists():
        logger.warn(
            f'Failed to load crops from {path} because it does not exist.')
        return []

    crops = []
    with open(path) as file:
        reader = csv.DictReader(file)
        for row in tqdm(reader):
            # print row keys
            crops.append((row['run_id'], row['camera'], row['ts'], int(row['r0']), int(
                row['r1']), int(row['c0']), int(row['c1'])))
            pass
        pass

    logger.debug(f'Loaded {len(crops)} crops.')
    return crops


def mat2tensor(mat):
    t = torch.from_numpy(mat).float()
    if mat.ndim == 2:
        return t.unsqueeze(2).permute(2, 0, 1)
    elif mat.ndim == 3:
        return t.permute(2, 0, 1)


def normalize_dim(a, d):
    """ Normalize a along dimension d."""
    return a.mul(a.pow(2).sum(dim=d, keepdim=True).clamp(min=0.00001).rsqrt())


def transform_identity(img):
    return img


def make_scale_transform(scale):
    return lambda img: rescale(img, scale, preserve_range=True, anti_aliasing=True, multichannel=True)


def make_scale_transform_w(target_width):
    return lambda img: rescale(img, float(target_width) / img.shape[1], preserve_range=True, anti_aliasing=True, multichannel=True)


def make_scale_transform_h(target_height):
    return lambda img: rescale(img, float(target_height) / img.shape[0], preserve_range=True, anti_aliasing=True, multichannel=True)
