# %%
import os
# %%
def remove_lines(file_path, excluded_keywords):
    temp_output = '/mnt/remote/data/users/kacper/datasets/somers-town_weather_v0/tmp_files.csv'
    with open(file_path, 'r') as file_in:
        with open(temp_output, "w") as file_out:
            file_out.writelines(filter(lambda line: not any([x in line for x in excluded_keywords]), file_in))

    os.replace(temp_output, file_path)

def keep_lines(file_path, out_file_path, keywords):
    with open(file_path, 'r') as file_in:
        with open(out_file_path, "w") as file_out:
            file_out.writelines(filter(lambda line: any([x in line for x in keywords]), file_in))

# %%

dark_worlds = [
'sedna/2022-03-28--09-39-44--session_2022_03_24_00_00_00_olympus_przemek_256x264x64__orientation_weight_3.0', 
'nereus/2022-08-22--09-25-43--session_2022_08_13_11_55_30_aml-v100-2nn-2_mike_temporal_resolution_832_320_3', 
'nereus/2021-08-17--10-31-13--session_2021_08_10_05_58_32_v100-4gpu_gunshi_bev-prod-maps-speed-limit-balancing', 
'neptune/2021-03-02--10-44-35--session_2021_02_15_13_57_50_vm-prod-training-dgx-02_gianluca_var_lw_010.2', 
'brizo/2022-08-12--12-56-56--session_2022_08_10_14_42_32_aml-v100-4nn-5_sofia_temporal_with_clothoids_drift_10_3cam', 
'brizo/2020-12-07--15-04-44--session_2020_11_15_20_47_53_vm-prod-training-dgx-05_corina_decoder_0', 
'nereus/2021-03-03--10-37-14--session_2021_03_01_22_00_00_host_zak_wayve_bev-2x_aug-pose_200k', 
'sedna/2022-03-08--09-34-17--session_2022_02_24_23_45_08_aml-a100-4nn-1_piotr_fleet_mf02_apece_mv_200ms_amp',
]

files = ['/mnt/remote/data/users/kacper/datasets/somers-town_weather_v0/real_files.csv']
for file in files:
    remove_lines(file, dark_worlds)
# %%

dark_worlds = [
'sedna/2022-03-28--09-39-44--session_2022_03_24_00_00_00_olympus_przemek_256x264x64__orientation_weight_3.0', 
'nereus/2022-08-22--09-25-43--session_2022_08_13_11_55_30_aml-v100-2nn-2_mike_temporal_resolution_832_320_3', 
'nereus/2021-08-17--10-31-13--session_2021_08_10_05_58_32_v100-4gpu_gunshi_bev-prod-maps-speed-limit-balancing', 
'neptune/2021-03-02--10-44-35--session_2021_02_15_13_57_50_vm-prod-training-dgx-02_gianluca_var_lw_010.2', 
'brizo/2022-08-12--12-56-56--session_2022_08_10_14_42_32_aml-v100-4nn-5_sofia_temporal_with_clothoids_drift_10_3cam', 
'brizo/2020-12-07--15-04-44--session_2020_11_15_20_47_53_vm-prod-training-dgx-05_corina_decoder_0', 
'nereus/2021-03-03--10-37-14--session_2021_03_01_22_00_00_host_zak_wayve_bev-2x_aug-pose_200k', 
'sedna/2022-03-08--09-34-17--session_2022_02_24_23_45_08_aml-a100-4nn-1_piotr_fleet_mf02_apece_mv_200ms_amp',
]

files = ['/mnt/remote/data/users/kacper/datasets/somers-town_weather_v0/sim_files.csv']
for file in files:
    remove_lines(file, dark_worlds)


# %%
# file = '/home/kacper/data/EPE/somers_town/sim_files.csv'
# runs = ['2022-10-05--13-11-03--somerstown-aft-loop-anti-clockwise-v1--f0e9b72c9fb9bb9f--5aae8a99']
# keep_lines(file, '/home/kacper/data/EPE/somers_town/video_test.csv', runs)

# %%
