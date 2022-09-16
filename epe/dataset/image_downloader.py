# %%
from azure_loader import AzureImageLoader
from random import choice
from tqdm import tqdm
from pyarrow.parquet import ParquetDataset
import os
from PIL import Image
import numpy as np
import cv2
import json
# %%
def semantic_conversion(seg, file):
    with open(file, 'r') as f:
        js = json.load(f)
    js = js['Classes']
    replace = {x['id']:x['conversion']['universal'] for x in js}

    func = lambda x: replace[x]
    func = np.vectorize(func)
    seg = func(seg)
    return seg
# %%
def image_match_desired_size(img, width_new, height_new):
    width, height = img.size

    old_aspect = width / height
    new_aspect = width_new / height_new

    # rescale on smaller dimension (aspect ratio wise)
    # then crop excess
    if new_aspect > old_aspect:
        scale = width_new / width
        height = int(scale * height)
        width = width_new
    else:
        scale = height_new / height
        height = height_new
        width = int(scale * width)

    # resizing to match desired height
    img = img.resize((width, height))

    # cropping to match desired width
    center_x = width // 2
    left = center_x - width_new//2
    right = center_x + width_new//2

    center_y = height // 2
    top = center_y - height_new//2
    bottom = center_y + height_new//2
    img = img.crop((left, top, right, bottom))

    return img

# %% Get Real run_ids
with open('/home/kacper/data/EPE/real_run_ids_2022_09_12.csv') as f:
    lines = f.readlines()[1:]
    real_run_ids, real_timestamps = zip(*[x.strip().split(",") for x in lines])
    real_timestamps = [int(x) for x in real_timestamps]
    real_data = list(zip(real_run_ids, real_timestamps))

# %% Get sim run_ids and timestamps
parquet_root = '/mnt/remote/wayve-datasets/databricks-users-datasets/vinh/synthetic_dataset_2022_08_26_9m/dataset=train'
parquet_files = sorted([os.path.join(parquet_root, x) for x in os.listdir(parquet_root) if x.endswith('.parquet')])

parquet_files = parquet_files[:1]

columns = [
    'run_id_noseginfix',
    'front-forward_image_timestamp_rgb',
    'front-forward_image_timestamp_depth',
]
dataset = ParquetDataset(parquet_files, memory_map=False, validate_schema=False)
sim_df = dataset.read_pandas(columns=columns).to_pandas()
sim_df = sim_df.sample(100)

# %%
EPE_HEIGHT = 540
EPE_WIDTH = 960
MAX_IMAGES = 100
camera = "front-forward"
loader = AzureImageLoader()
data_save_root = '/home/kacper/data/EPE'
semantic_conversion_classes_file = '/home/kacper/code/EPE/wayve_v3_mseg.json'
# %%
# Download SIM
for i in tqdm(range(MAX_IMAGES)):
    run_id = sim_df.iloc[i]['run_id_noseginfix']
    ts = sim_df.iloc[i]['front-forward_image_timestamp_rgb']
    for camera in ['front-forward']:
        for mode in ['image', 'depth', 'semseg']:
            mode_out = 'rgb' if mode == 'image' else mode
            output = loader.load(run_id, camera, ts, mode = mode)
            img = Image.open(output)
            ext_map = {'image': 'jpeg', 'semseg': 'png', 'depth': 'png'}

            img = image_match_desired_size(img, EPE_WIDTH, EPE_HEIGHT)
            img.save(os.path.join(data_save_root, "sim", mode_out, str(i).zfill(4) + f".{ext_map[mode]}"))

# %%
INTRINSICS = 1134.1, 0.0, 1020.2, 0.0, 1135.2, 640.39, 0.0, 0.0, 1.0
DISTORTION_COEFS = -0.039527, 0.0064852, -0.022026, 0.0095635, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
intrinsics = np.array(INTRINSICS, dtype=np.float32).reshape(3, 3)
distortion_coefs = np.array(DISTORTION_COEFS, dtype=np.float32)

for i in tqdm(range(MAX_IMAGES)):
    run_id, ts = choice(real_data)

    data = loader.load(run_id, camera, ts, mode="image")
    img = Image.open(data).convert("RGB")
    img_before = img

    img = np.array(img)

    # undistort
    alpha = 1
    img = np.asarray(img)
    new_intrinsics, _ = cv2.getOptimalNewCameraMatrix(
        intrinsics, distortion_coefs, (img.shape[1], img.shape[0]), alpha
    )
    img = cv2.undistort(img, intrinsics, distortion_coefs, None, new_intrinsics)
    img = Image.fromarray(img)

    img = image_match_desired_size(img, EPE_WIDTH, EPE_HEIGHT)
    img.save(os.path.join(data_save_root, "real", "rgb", str(i).zfill(4) + '.jpg'))
# %%
