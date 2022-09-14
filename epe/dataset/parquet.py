# %%
from pyarrow.parquet import ParquetDataset
import os
# %%
parquet_root = '/mnt/remote/wayve-datasets/databricks-users-datasets/vinh/synthetic_dataset_2022_08_26_9m/dataset=train'
parquet_files = sorted([os.path.join(parquet_root, x) for x in os.listdir(parquet_root) if x.endswith('.parquet')])

parquet_files = parquet_files[:1]

columns = [
    'run_id_noseginfix',
    'front-forward_image_timestamp_rgb',
    'front-forward_image_timestamp_depth',
]
dataset = ParquetDataset(parquet_files, memory_map=False, validate_schema=False)

# %%
df = dataset.read_pandas(columns=columns).to_pandas()
df = df.sample(100)

# %%
df.iloc[0].run_id_noseginfix

# %%
print(df.iloc[0]['front-forward_image_timestamp_rgb'])
# %%
