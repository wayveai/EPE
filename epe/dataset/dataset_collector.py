# %%
from random import choice
from time import time
from tqdm import tqdm
from pyarrow.parquet import ParquetDataset
import os

if __name__ == '__main__':
    NUM_SAMPLES = 100
    real_img_colection_path = '/home/kacper/data/EPE/real_azure/real_rgb_files.csv'
    sim_img_colection_path = '/home/kacper/data/EPE/sim_azure/sim_rgb_files.csv'
    pass
    # %%

    # %% Get Real run_ids
    with open('/home/kacper/data/EPE/real_run_ids_2022_09_12.csv') as f:
        lines = f.readlines()[1:]
        real_run_ids, real_timestamps = zip(*[x.strip().split(",") for x in lines])
        real_timestamps = [int(x) for x in real_timestamps]
        real_data = list(zip(real_run_ids, real_timestamps))[:NUM_SAMPLES]

    with open(real_img_colection_path, 'w') as f:
        for run_id, timestamp in real_data:
            timestamp = str(timestamp).zfill(12) + 'unixus.jpeg'
            path = f'{run_id}/cameras/front-forward--rgb/{[timestamp]}'
            f.write(path + "\n")
    

    # %% Get sim run_ids and timestamps
    parquet_root = '/mnt/remote/wayve-datasets/databricks-users-datasets/vinh/synthetic_dataset_2022_08_26_9m/dataset=train'
    parquet_files = sorted([os.path.join(parquet_root, x) for x in os.listdir(parquet_root) if x.endswith('.parquet')])

    parquet_files = parquet_files[:1]

    columns = [
        'run_id_noseginfix',
        'front-forward_image_timestamp_rgb'
    ]
    dataset = ParquetDataset(parquet_files, memory_map=False, validate_schema=False)
    sim_df = dataset.read_pandas(columns=columns).to_pandas()
    sim_df = sim_df.sample(NUM_SAMPLES)


    with open(sim_img_colection_path, 'w') as f:
        for i, x in sim_df.iterrows():
            run_id = x.loc['run_id_noseginfix']
            timestamp = x.loc['front-forward_image_timestamp_rgb']
            timestamp = str(timestamp).zfill(12) + 'unixus.jpeg'
            path = f'{run_id}/cameras/front-forward--rgb/{timestamp}'
            f.write(path + "\n")
# %%
