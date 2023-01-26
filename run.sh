touch /home/kacper/progress
python epe/matching/feature_based/collect_crops.py sim /home/kacper/code/EPE/datasets/urban-driving/sim_files.csv --out_dir /home/kacper/code/EPE/datasets/urban-driving
echo "sim crops collected" >> /home/kacper/progress
python epe/matching/feature_based/collect_crops.py real /home/kacper/code/EPE/datasets/urban-driving/real_files.csv --out_dir  /home/kacper/code/EPE/datasets/urban-driving
echo "real crops collected" >> /home/kacper/progress

python epe/matching/feature_based/find_knn.py /home/kacper/code/EPE/datasets/urban-driving/crop_sim.npz /home/kacper/code/EPE/datasets/urban-driving/crop_real.npz /home/kacper/code/EPE/datasets/urban-driving/knn_sim-real.npz -k 5
echo "knn done" >> /home/kacper/progress
python epe/matching/filter.py /home/kacper/code/EPE/datasets/urban-driving/knn_sim-real.npz /home/kacper/code/EPE/datasets/urban-driving/crop_sim.csv /home/kacper/code/EPE/datasets/urban-driving/crop_real.csv 1.0 /home/kacper/code/EPE/datasets/urban-driving/matched_crops_sim-real.csv
echo "filter done" >> /home/kacper/progress


python epe/matching/compute_weights.py /home/kacper/code/EPE/datasets/urban-driving/matched_crops_sim-real.csv 456 960 /home/kacper/code/EPE/datasets/urban-driving/crop_weights_sim-real.npz
echo "weights computed" >> /home/kacper/progress