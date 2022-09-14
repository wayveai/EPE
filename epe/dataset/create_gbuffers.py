# %%
import os
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
# %%
gbuffers = ['depth', 'depth']
base_path = Path('/home/kacper/data/EPE/sim')

out_dir_name = 'g_buffers_depth'
os.makedirs(base_path / out_dir_name, exist_ok=True)
# %%

g_files = []
g_file_names = []
for g in gbuffers:
    g_path = base_path / g
    files = sorted(list(g_path.iterdir()))
    if not g_file_names:
        g_file_names = [x.stem for x in files]
    files = map(str, files)


    g_files.append(files)

g_files = list(zip(*g_files))
# %%
for name, gf in tqdm(zip(g_file_names, g_files), total=len(g_file_names)):
    g_buffer_data = []
    for g in gf:
        if "depth" in g:
            # TODO: currently this assumes that depth data is jpg with range [0, 255]
            data = Image.open(g)
            data = np.asarray(data).astype(np.float32)
            data = data.min(axis=2)
            data /= 255
            g_buffer_data.append(data)
        else:
            raise NotImplementedError("currently not supporting this g-buffer data: ", g)

    g_buffer_data = np.stack(g_buffer_data) 

    np.savez_compressed(os.path.join(base_path, out_dir_name, f'{name}.npz'), data=g_buffer_data)

        

# %%
