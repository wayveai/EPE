# %%
from importlib.machinery import PathFinder
from random import choice
from time import time
from numpy import isin
from tqdm import tqdm
from pyarrow.parquet import ParquetDataset
import os
from random import sample
import re

if __name__ == '__main__':
    SAMPLING_RATE = 5
    real_img_colection_path = '/home/kacper/data/EPE/somers_town/real_rgb_files.csv'
    real_dataset_path = '/home/kacper/data/EPE/somers_town/real_files.csv'
    sim_dataset_path = '/home/kacper/data/EPE/somers_town/sim_files.csv'
    pass
    # %%

    # # %% Get Real run_ids
    # with open('/home/kacper/data/EPE/real_limit_10000.csv') as f:
    #     lines = f.readlines()[1:]
    #     real_run_ids, real_timestamps = zip(*[x.strip().split(",") for x in lines])
    #     real_timestamps = [int(x) for x in real_timestamps]
    #     real_data = list(zip(real_run_ids, real_timestamps))

    # with open(real_img_colection_path, 'w') as f:
    #     for run_id, timestamp in real_data:
    #         timestamp = str(timestamp).zfill(12) + 'unixus.jpeg'
    #         path = f'{run_id}/cameras/front-forward/{timestamp}'
    #         f.write(path + "\n")

    # with open(real_dataset_path, 'w') as f:
    #     for run_id, timestamp in real_data:
    #         timestamp = str(timestamp).zfill(12)

    #         f.write(f'{run_id},{timestamp}\n')

    # %% Get sim run_ids and timestamps

    # parquet_root = '/mnt/remote/wayve-datasets/databricks-users-datasets/kyra/real_somers_town_2022_09_21/dataset=train'
    # parquet_files = sorted([os.path.join(parquet_root, x) for x in os.listdir(parquet_root) if x.endswith('.parquet')])

    # parquet_files = parquet_files

    # columns = [
    #     'run_id_noseginfix',
    #     'front-forward_image_timestamp_rgb'
    # ]
    # dataset = ParquetDataset(parquet_files, memory_map=False, validate_schema=False)
    # df = dataset.read_pandas(columns=columns).to_pandas()


    # with open(real_img_colection_path, 'w') as f:
    #     for i, x in df.iterrows():
    #         if i % SAMPLING_RATE == 0 :
    #             run_id = x.loc['run_id_noseginfix']
    #             timestamp = x.loc['front-forward_image_timestamp_rgb']
    #             timestamp = str(timestamp).zfill(12) + 'unixus.jpeg'
    #             path = f'{run_id}/cameras/front-forward--rgb/{timestamp}'
    #             f.write(path + "\n")

    # with open(real_dataset_path, 'w') as f:
    #     for i, x in df.iterrows():
    #         if i % SAMPLING_RATE == 0 :
    #             run_id = x.loc['run_id_noseginfix']
    #             timestamp = str(x.loc['front-forward_image_timestamp_rgb']).zfill(12)
    #             # rgb_path = f'{run_id}/cameras/front-forward--rgb/{timestamp}unixus.jpeg'
    #             f.write(f'{run_id},{timestamp}\n')
# %%
def uniform_sampling(files_csv_path, sampling_rate, data_root='/home/kacper/data/datasets', path_filter=lambda x: True, car_filter=lambda x: True):
    with open(files_csv_path, 'w') as f:
        print(data_root)
        for car in os.listdir(data_root):
            if car_filter(car):
                for run in os.listdir(os.path.join(data_root, car)):
                    camera_path = os.path.join(data_root, car, run, 'cameras', 'front-forward--rgb')
                    if path_filter(camera_path):
                        img_names = sorted(os.listdir(camera_path))
                        for i, img_name in enumerate(img_names):
                            if i % sampling_rate == 0:
                                run_id = os.path.join(car, run)
                                timestamp = re.search('\d+', img_name).group()
                                f.write(f'{run_id},{timestamp}\n')


        
# %%
def random_sampling(files_csv_path, p, data_root='/home/kacper/data/datasets', path_filter=lambda x: True, car_filter=lambda x: True):
    assert p <= 1 and p > 0

    with open(files_csv_path, 'w') as f:
        for car in os.listdir(data_root):
            if car_filter(car):
                for run in os.listdir(os.path.join(data_root, car)):
                    camera_path = os.path.join(data_root, car, run, 'cameras', 'front-forward--rgb')
                    if path_filter(camera_path):
                        img_names = os.listdir(camera_path)
                        img_names = sample(img_names, int(len(img_names) * p))
                        for i, img_name in enumerate(img_names):
                            run_id = os.path.join(car, run)
                            timestamp = re.search('\d+', img_name).group()
                            f.write(f'{run_id},{timestamp}\n')


# Sim Video
# sim_path_filter = lambda x: 'ningaloo--3_6_190--jaguaripacenoslipdynamics--ningaloo_av_2_0/2022-10-18--11-44-21--somerstown-aft-loop-anti-clockwise-v1--182909beaf375f6e--bf07930e' in x
# sim_car_filter = lambda car: 'ningaloo' in car and 'av_2_0' in car
# uniform_sampling('/home/kacper/data/EPE/somers_town/video_test.csv', 1, path_filter=sim_path_filter)

# # %% REAL 
# real_somers_town = '/home/kacper/code/EPE/datasets/somers_town/real_files.csv'
# real_car_filter = lambda car: car in ['brizo', 'nammu', 'neptune', 'nereus', 'sedna']
# uniform_sampling(real_somers_town, 1, car_filter=real_car_filter)

# %%

# %% SIM
# sim_somers_town = '/home/kacper/code/EPE/datasets/somers_town/sim_files.csv'
# sim_car_filter = lambda car: 'ningaloo' in car
# uniform_sampling(sim_somers_town, 8, car_filter=sim_car_filter)
# %%
