wandb docker -it -v /home/kacper/data/datasets:/home/kacper/data/datasets \
--no-dir \
-v /home/kacper/docker/wandb:/home/kacper/code/EPE/wandb \
-v /home/kacper/docker/log:/home/kacper/code/EPE/log \
-v /home/kacper/data/EPE/weights:/home/kacper/resume \
--runtime nvidia \
-e NVIDIA_VISIBLE_DEVICES=1 \
--shm-size 8G \
wayve.azurecr.io/sim2real_epe
