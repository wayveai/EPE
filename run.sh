touch /home/kacper/progress
python epe/matching/feature_based/collect_crops.py sim /home/kacper/code/EPE/datasets/somers_town/sim_files.csv --out_dir /home/kacper/code/EPE/datasets/somers_town --data_root /home/kacper/data/datasets
echo "sim crops collected" >> /home/kacper/progress
python epe/matching/feature_based/collect_crops.py real /home/kacper/code/EPE/datasets/somers_town/real_files.csv --out_dir /home/kacper/code/EPE/datasets/somers_town --data_root /home/kacper/data/datasets
echo "real crops collected" >> /home/kacper/progress

python epe/matching/feature_based/find_knn.py /home/kacper/code/EPE/datasets/somers_town/crop_sim.npz /home/kacper/code/EPE/datasets/somers_town/crop_real.npz /home/kacper/code/EPE/datasets/somers_town/knn_sim-real.npz -k 10
echo "knn done" >> /home/kacper/progress
python epe/matching/filter.py /home/kacper/code/EPE/datasets/somers_town/knn_sim-real.npz /home/kacper/code/EPE/datasets/somers_town/crop_sim.csv /home/kacper/code/EPE/datasets/somers_town/crop_real.csv 1.0 /home/kacper/code/EPE/datasets/somers_town/matched_crops_sim-real.csv
echo "filter done" >> /home/kacper/progress


python epe/matching/compute_weights.py /home/kacper/code/EPE/datasets/somers_town/matched_crops_sim-real.csv 600 960 /home/kacper/code/EPE/datasets/somers_town/crop_weights_sim-real.npz
echo "weights coimputed" >> /home/kacper/progress