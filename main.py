from argparse import ArgumentParser
import subprocess
import os
import torch

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--config', type=str, help='path to directory containing train yaml configs')
	parser.add_argument('--session_path ', default='')
	parser.add_argument('--session_dir_name ', default='')

	args = parser.parse_args()

	commands = [[]]
	processes = []
	num_gpus = torch.cuda.device_count()
	print('num gpus: ', num_gpus)
	print('num train configs: ', len(os.listdir(args.config)))
	assert len(os.listdir(args.config)) <= num_gpus

	for i, train_config in enumerate(os.listdir(args.config)):
		config_name = os.path.join(args.config, train_config)

		cmd = ['python', '/app/run.py', '--log', 'info', '--log_dir',
		'/app/log', '--action', 'train', '--gpu', str(i), '--notes',config_name,
		'--config', config_name]
		p = subprocess.Popen(cmd)
		processes.append(p)

	for p in processes:
		p.wait()
