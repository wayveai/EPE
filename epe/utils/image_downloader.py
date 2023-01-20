# %%
from pathlib import Path
from epe.dataset.azure_loader import AzureImageLoader, image_match_desired_size
from random import choice
from tqdm import tqdm
from pyarrow.parquet import ParquetDataset
import os
from PIL import Image
import numpy as np
import json
from IPython.display import display

from epe.dataset.utils import read_azure_filelist
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
azure_loader = AzureImageLoader()

sim_files_path = '/home/kacper/data/EPE/somers_town/sim_files.csv'
sim_root = '/home/kacper/data/datasets/'
sim_modes = ['rgb', 'segmentation', 'depth', 'segmentation-mseg', 'normal']
sim_files = read_azure_filelist(sim_files_path, sim_modes)

real_modes = ['rgb', 'segmentation-mseg']
real_files_path = '/home/kacper/data/EPE/somers_town/real_files.csv'
real_root = '/home/kacper/data/datasets/'
real_files = read_azure_filelist(real_files_path, real_modes)
failed_path = '/home/kacper/failed.txt'

def download_files(all_paths, root, shape=(960, 600), check_exist=False):
    for paths in tqdm(all_paths, total=len(all_paths)):
        for path in paths:
            save_path = os.path.join(root, path)
            if check_exist and os.path.exists(save_path):
                with open(failed_path, 'w+') as f:
                    f.write(path + '\n')
                continue
            os.makedirs(Path(save_path).parent, exist_ok=True)
            try:
                data = azure_loader.load_img_from_path_and_resize(path, *shape)
                data.save(os.path.join(root, path))
            except:
                with open(failed_path, 'w+') as f:
                    f.write(path + '\n')

# print("Downloading REAL")
# download_files(real_files, real_root, check_exist=True)
# print("Downloading SIM")
# download_files(sim_files, sim_root)
    
# %%
# # %% Get Real run_ids
# with open('/home/kacper/data/EPE/real_run_ids_2022_09_12.csv') as f:
#     lines = f.readlines()[1:]
#     real_run_ids, real_timestamps = zip(*[x.strip().split(",") for x in lines])
#     real_timestamps = [int(x) for x in real_timestamps]
#     real_data = list(zip(real_run_ids, real_timestamps))

# %% Get sim run_ids and timestamps
parquet_root = '/mnt/azure/wayveproddataset/databricks-users/datasets/kacper/somerstown/dataset=train'
parquet_files = sorted([os.path.join(parquet_root, x) for x in os.listdir(parquet_root) if x.endswith('.parquet')])

parquet_files = parquet_files[:]

columns = [
    'run_id_noseginfix',
    'front-forward_image_timestamp_rgb',
    'distance_travelled_m',
]
dataset = ParquetDataset(parquet_files, memory_map=False, validate_schema=False)
df = dataset.read_pandas(columns=columns).to_pandas()

# %%
excluded_runs = [
    'brizo/2021-09-13--09-39-36--session_2021_09_07_15_18_43_dgx-v100_becky_fb_iw_entron_finetune_entron',
    'nammu/2020-11-18--14-33-21--session_2020_11_12_18_57_26_vm-prod-training-05_piotr_flow_conv6_loss_balancing_to_speed_steering_cooldown50k',
    'nammu/2021-02-17--15-24-30--session_2021_02_06_19_14_43_bev-augscale-big-0-0_hannes_bev_aug_lat0.6_long_0_rot_0.3_interpolation_cosine_300k_w0.1_p0.5'
]

print(len(df))
df = df[~df['run_id_noseginfix'].isin(excluded_runs)]
print(len(df))
df = df.drop_duplicates('distance_travelled_m')
print(len(df))

# %%
df = df.iloc[::10]
print(len(df))

# %%
EPE_HEIGHT = 600
EPE_WIDTH = 960
MAX_IMAGES = 1
camera = "front-forward"
loader = AzureImageLoader()
data_save_root = '/home/kacper/data/datasets'
# %%
# Download REAL
for i in tqdm(range(len(df))):
    run_id = df.iloc[i]['run_id_noseginfix']
    ts = df.iloc[i]['front-forward_image_timestamp_rgb']
    for camera in ['front-forward']:
        output = loader.load(run_id, camera, ts, mode = 'rgb')
        img = Image.open(output)

        img = image_match_desired_size(img, EPE_HEIGHT, EPE_WIDTH)
        save_dir =os.path.join(data_save_root, run_id, 'cameras', 'front-forward--rgb')
        os.makedirs(save_dir, exist_ok=True)
        img.save( os.path.join(save_dir, f"{ts}unixus.jpeg"))
# %%
# Download SIM
# for i in tqdm(range(MAX_IMAGES)):
#     run_id = sim_df.iloc[i]['run_id_noseginfix']
#     ts = sim_df.iloc[i]['front-forward_image_timestamp_rgb']
#     for camera in ['front-forward']:
#         for mode in ['rgb', 'depth', 'segmentation']:
#             output = loader.load(run_id, camera, ts, mode = mode)
#             img = Image.open(output)
#             ext_map = {'rgb': 'jpeg', 'segmentation': 'png', 'depth': 'png'}

#             img = image_match_desired_size(img, EPE_WIDTH, EPE_HEIGHT)
#             img.save(os.path.join(data_save_root, "sim", mode, str(i).zfill(4) + f".{ext_map[mode]}"))


# %%
# for i in tqdm(range(MAX_IMAGES)):
#     run_id, ts = choice(real_data)

#     data = loader.load(run_id, camera, ts, mode="image")
#     img = Image.open(data).convert("RGB")
#     img_before = img

#     img = np.array(img)

#     # undistort
#     alpha = 1
#     img = np.asarray(img)
#     new_intrinsics, _ = cv2.getOptimalNewCameraMatrix(
#         intrinsics, distortion_coefs, (img.shape[1], img.shape[0]), alpha
#     )
#     img = cv2.undistort(img, intrinsics, distortion_coefs, None, new_intrinsics)
#     img = Image.fromarray(img)

#     img = image_match_desired_size(img, EPE_WIDTH, EPE_HEIGHT)
#     img.save(os.path.join(data_save_root, "real", "rgb", str(i).zfill(4) + '.jpg'))
# # %%