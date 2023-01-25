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
# %%
parquet_root = '/mnt/azure/wayveproddataset/databricks-users/datasets/kacper/urban_driving_v2/dataset=train'
parquet_files = sorted([os.path.join(parquet_root, x) for x in os.listdir(parquet_root) if x.endswith('.parquet')])

dataset = ParquetDataset(parquet_files, memory_map=False, validate_schema=False)
df = dataset.read_pandas().to_pandas()
# %%
print(len(df))
# %%
print(df.iloc[0])
# sample n number of rows from the df pandas dataframe
n_sample = 30_000
df_sample = df.sample(n=n_sample)
print(len(df_sample))
# %%
# print columns of df
print(df.columns)

# %%
sim_csv_path = '/home/kacper/code/EPE/datasets/urban-driving/sim_files.csv'
with open(sim_csv_path, 'w') as f:
    for row in df_sample.iterrows():
        row = row[1]
        run_id =  row['run_id']
        for camera_id in ['front-forward', 'front-right-rightward', 'front-left-leftward']:
            key = camera_id.replace("-", "_") + "__image_timestamp_unixus"
            ts = row[key]

            f.write(','.join([run_id, camera_id, str(ts)]) + '\n')
# %%

