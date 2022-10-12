python epe/matching/feature_based/collect_crops.py sim /home/kacper/data/EPE/somers_town/sim_files.csv --out_dir /home/kacper/data/EPE/somers_town --data_root /home/kacper/data/datasets
python epe/matching/feature_based/collect_crops.py real /home/kacper/data/EPE/somers_town/real_files.csv --out_dir /home/kacper/data/EPE/somers_town --data_root /home/kacper/data/datasets

python epe/matching/feature_based/find_knn.py /home/kacper/data/EPE/somers_town/crop_sim.npz /home/kacper/data/EPE/somers_town/crop_real.npz /home/kacper/data/EPE/somers_town/knn_sim-real.npz -k 10
python epe/matching/filter.py /home/kacper/data/EPE/somers_town/knn_sim-real.npz /home/kacper/data/EPE/somers_town/crop_sim.csv /home/kacper/data/EPE/somers_town/crop_real.csv 1.0 /home/kacper/data/EPE/somers_town/matched_crops_sim-real.csv

python epe/matching/compute_weights.py /home/kacper/data/EPE/somers_town/matched_crops_sim-real.csv 600 960 /home/kacper/data/EPE/somers_town/crop_weights_sim-real.npz