# %%
from random import choice
from time import time
from tqdm import tqdm
from pyarrow.parquet import ParquetDataset
import os

if __name__ == '__main__':
    SAMPLING_RATE = 10
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

    parquet_root = '/mnt/remote/wayve-datasets/databricks-users-datasets/kyra/real_somers_town_2022_09_21/dataset=train'
    parquet_files = sorted([os.path.join(parquet_root, x) for x in os.listdir(parquet_root) if x.endswith('.parquet')])

    parquet_files = parquet_files

    columns = [
        'run_id_noseginfix',
        'front-forward_image_timestamp_rgb'
    ]
    dataset = ParquetDataset(parquet_files, memory_map=False, validate_schema=False)
    df = dataset.read_pandas(columns=columns).to_pandas()


    with open(real_img_colection_path, 'w') as f:
        for i, x in df.iterrows():
            if i % SAMPLING_RATE == 0 :
                run_id = x.loc['run_id_noseginfix']
                timestamp = x.loc['front-forward_image_timestamp_rgb']
                timestamp = str(timestamp).zfill(12) + 'unixus.jpeg'
                path = f'{run_id}/cameras/front-forward--rgb/{timestamp}'
                f.write(path + "\n")

    with open(real_dataset_path, 'w') as f:
        for i, x in df.iterrows():
            if i % SAMPLING_RATE == 0 :
                run_id = x.loc['run_id_noseginfix']
                timestamp = str(x.loc['front-forward_image_timestamp_rgb']).zfill(12)
                # rgb_path = f'{run_id}/cameras/front-forward--rgb/{timestamp}unixus.jpeg'
                f.write(f'{run_id},{timestamp}\n')
# %%
files_csv_path = '/home/kacper/data/EPE/somers_town/sim_files.csv'
data_root = '/home/kacper/data/datasets/'
with open(files_csv_path, 'w') as f:
    for car in os.listdir(data_root):
        if 'ningaloo' in car:
            for run in os.listdir(os.path.join(data_root, car)):
                camera_path = os.path.join(data_root, car, run, 'cameras', 'front-forward--rgb')
                for i, img_name in enumerate(os.listdir(camera_path)):
                    if i % SAMPLING_RATE == 0:
                        run_id = os.path.join(car, run)
                        timestamp = img_name[:12]
                        f.write(f'{run_id},{timestamp}\n')

        
# %%
