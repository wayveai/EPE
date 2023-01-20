import datetime
import logging
import random
from pathlib import Path
import sys
import os

import numpy as np
from scipy.io import savemat
import torch
from torch import autograd
from torchvision.utils import make_grid
import yaml
from tqdm import tqdm

import gc
import wandb
import re


def seed_worker(id):
	random.seed(torch.initial_seed() % np.iinfo(np.int32).max)
	np.random.seed(torch.initial_seed() % np.iinfo(np.int32).max)
	pass


def toggle_grad(model, requires_grad):
	for p in model.parameters():
		p.requires_grad_(requires_grad)
		pass
	pass


_logstr2level = {
	'critical': logging.CRITICAL,
	'error': logging.ERROR,
	'warn': logging.WARNING,
	'warning': logging.WARNING,
	'info': logging.INFO,
	'debug': logging.DEBUG
}


def parse_loglevel(loglevel_arg):
	level = _logstr2level.get(loglevel_arg.lower())
	if level is None:
		raise ValueError(
			f"log level given: {loglevel_arg}"
			f" -- must be one of: {' | '.join(_logstr2level.keys())}")

	return level


def init_logging(args):
	now = datetime.datetime.now()
	log_path = args.log_dir / f'{args.config.stem}_{datetime.date.today().isoformat()}_{now.hour}-{now.minute}-{now.second}.log'
	# with open(log_path, 'w') as f:
	# 	pass
	level = parse_loglevel(args.log)
	logging.basicConfig(level=level, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler()])	


class NetworkState:
	""" Capture (training) state of a network.

	"""

	def __init__(self, network, cfg, name='network_state'):

		self._log               = logging.getLogger(f'epe.experiment.{name}')
		self.network            = network
		self.iterations         = 0

		self._parse_config(cfg)
		pass


	def _parse_config(self, cfg):

		self._init_optimizer(dict(cfg.get('optimizer', {})))
		self._init_scheduler(dict(cfg.get('scheduler', {})))

		self.learning_rate = self.scheduler.get_last_lr()
		pass


	def _init_optimizer(self, cfg):

		self.learning_rate      = float(cfg.get('learning_rate', 0.001))
		self.clip_gradient_norm = float(cfg.get('clip_gradient_norm', -1))
		self.clip_weights       = float(cfg.get('clip_weights', -1))

		momentum           = float(cfg.get('momentum', 0.0))
		weight_decay       = float(cfg.get('weight_decay', 0.0001))
		adam_ams           = bool(cfg.get('adam_ams', False))
		adam_beta          = float(cfg.get('adam_beta', 0.9))
		adam_beta2         = float(cfg.get('adam_beta2', 0.999))
		optimizer          = str(cfg.get('type', 'adam'))

		self._log.debug(f'  learning rate : {self.learning_rate}')
		self._log.debug(f'  clip grad norm: {self.clip_gradient_norm}')
		self._log.debug(f'  clip_weights  : {self.clip_weights}')

		self._log.debug(f'  optimizer     : {optimizer}')

		if optimizer == 'adam':
			self._log.debug(f'    ams         : {adam_ams}')
			self._log.debug(f'    beta        : {adam_beta}')
			self._log.debug(f'    beta2       : {adam_beta2}')
			self.optimizer = torch.optim.Adam(params=self.network.parameters(), lr=self.learning_rate, betas=(adam_beta, adam_beta2), weight_decay=weight_decay, amsgrad=adam_ams)

		elif optimizer == 'adamw':
			self._log.debug(f'    ams         : {adam_ams}')
			self._log.debug(f'    beta        : {adam_beta}')
			self._log.debug(f'    beta2       : {adam_beta2}')
			self.optimizer = torch.optim.AdamW(params=self.network.parameters(), lr=self.learning_rate, betas=(adam_beta, adam_beta2), weight_decay=weight_decay, amsgrad=adam_ams)

		elif optimizer == 'sgd':
			self._log.debug(f'    momentum      : {momentum}')
			self._log.debug(f'    weight_decay  : {weight_decay}')
			self.optimizer = torch.optim.SGD(params=self.network.parameters(), lr=self.learning_rate, momentum=momentum, weight_decay=weight_decay)

		else:
			raise NotImplementedError

		pass


	def _init_scheduler(self, cfg):

		scheduler  = str(cfg.get('scheduler', 'step'))
		step       = int(cfg.get('step', 1000000))
		step_gamma = float(cfg.get('step_gamma', 1))
			
		if scheduler == 'step':
			self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step, gamma=step_gamma)
			
		elif scheduler == 'exp':
			# will produce  a learning rate of step_gamma at step
			gamma = step_gamma**(1.0/step)
			self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,  gamma=gamma)
			
		elif scheduler == 'cosine':			
			self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,  T_max=step, eta_min=self.learning_rate*step_gamma)		
		pass


	def load_from_dict(self, d):
		""" Initialize the network state from a dictionary."""

		self.network.load_state_dict(d['network'])
		# self._log.warn('NOT LOADING optimizer HERE')
		self.optimizer.load_state_dict(d['optimizer'])
		# self._log.warn('NOT LOADING scheduler HERE')
		self.scheduler.load_state_dict(d['scheduler'])
		self.iterations = d.get('iterations', 0)
		pass


	def save_to_dict(self):
		""" Save the network state to a disctionary."""

		return {\
			'network':self.network.state_dict(),
			'optimizer':self.optimizer.state_dict(),
			'scheduler':self.scheduler.state_dict()}, {
			'iterations':self.iterations}


	def prepare(self):
		self.optimizer.zero_grad(set_to_none=True)
		pass


	def update(self):
		if self.clip_gradient_norm > 0:
			# loss_infos['ggn'] = 
			# n = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_gradient_norm, norm_type=2)
			n = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_gradient_norm, norm_type='inf')
			if self._log.isEnabledFor(logging.DEBUG):
				self._log.debug(f'gradient norm: {n}')
			pass

		self.optimizer.step()						
		self.scheduler.step()
		lr = self.scheduler.get_last_lr()

		if lr != self.learning_rate:
			self._log.info(f'Learning rate set to {lr}.')
			self.learning_rate = lr
			pass

		if self.clip_weights > 0:
			for p in self.network.parameters():
				p.data.clamp_(-self.clip_weights, self.clip_weights)
				pass
			pass
				
		self.iterations += 1
		pass
	pass



class LogSync:
	def __init__(self, logger, log_interval):
		self.scalars = {}
		self._log          = logger
		self._log_interval = log_interval
		self._scalar_queue = {}
		self._delay        = 3


	def update(self, i, scalars):
		for k,v in scalars.items():
			if k not in self._scalar_queue:
				self._scalar_queue[k] = {}
				pass

			self._scalar_queue[k][i] = v.to('cpu', non_blocking=True)
			pass
		pass

	# def _update_gpu(self, scalars):
	# 	for k,v in scalars.items():
	# 		if k not in self.scalars:
	# 			self.scalars[k] = [torch.tensor([0.0], device=v.device), 0]
	# 			pass

	# 		self.scalars[k] = [self.scalars[k][0]+v, self.scalars[k][1]+1]
	# 		pass
	# 	pass

	def print(self, i):
		""" Print to screen. """

		if i % (20*self._log_interval) == 0:
			line = [f'{i:d} ']
			for t in self._scalar_queue.keys():
				line.append('%-4.4s ' % t)
				pass
			self._log.info('')
			self._log.info(''.join(line))


		if i % self._log_interval == 0:

			line = [f'{i:d} ']

			# Loss infos
			new_queue = {}
			for k,v in self._scalar_queue.items():
				valid = {j:float(vj) for j,vj in v.items() if i - j >= self._delay}
				if valid:
					vv = valid.values()
					line.append(f'{sum(vv)/len(vv):.2f} ')
					for vk in valid.keys():
						del v[vk]
						pass
					pass
				else:
					line.append('---- ')
					pass				
				pass

			self._log.info(''.join(line))			
			pass

def log_name_extractor(log_name):
	return re.search('[A-Za-z_\-]*', log_name)[0]


class BaseExperiment:
	""" Provide scaffold for common operations in an experiment.

	The class provides a scaffold for common operations required in running an experiment.
	It provides a training, validation, and testing loop, methods for loading and storing weights,
	and storing debugging info. It does not specify network architectures, optimizers, or datasets
	as they may vary a lot depending on the specific experiment or task.

	"""

	actions  = ['train', 'test', 'infer']
	networks = {}

	def __init__(self, args):
		"""Common set up code for all actions."""
		self.action       = args.action
		self._log         = logging.getLogger('main')		
		self.no_safe_exit = args.no_safe_exit
		self.collate_fn_train = None
		self.collate_fn_val   = None
		self.loader_fake = None

		self.device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu')

		self._load_config(args.config)
		self._parse_config()


		self._save_id = 0
		self._init_directories()
		self._init_dataset()
		self._init_network()
		self._init_network_state()

		if self.seed is not None:
			torch.manual_seed(self.seed)
			pass
		pass


	def _load_config(self, config_path):
		with open(config_path) as file:
			self.cfg = yaml.safe_load(file)

	def _parse_config(self):
		common_cfg = dict(self.cfg.get('common', {}))
		self.unpin           = bool(common_cfg.get('unpin', False))
		self.seed            = common_cfg.get('seed', None)
		self.batch_size      = int(common_cfg.get('batch_size', 1))
		self.num_loaders     = int(common_cfg.get('num_loaders', os.cpu_count() // torch.cuda.device_count() if torch.cuda.is_available() else 10))
		self._log_interval   = int(common_cfg.get('log_interval', 1))

		prof_cfg = dict(self.cfg.get('profile', {}))
		self._profile        = bool(prof_cfg.get('enable', False))
		self._profile_gpu    = bool(prof_cfg.get('gpu', False))
		self._profile_memory = bool(prof_cfg.get('memory', False))
		self._profile_stack  = bool(prof_cfg.get('stack', False))
		self._profile_path   = Path(prof_cfg.get('path', '.'))

		self._log.debug(f'  unpin        : {self.unpin}')
		self._log.debug(f'  seed         : {self.seed}')
		self._log.debug(f'  batch_size   : {self.batch_size}')
		self._log.debug(f'  num_loaders  : {self.num_loaders}')
		self._log.debug(f'  log_interval : {self._log_interval}')
		self._log.debug(f'  profile      : {self._profile}')

		self.shuffle_test    = bool(self.cfg.get('shuffle_test', False))
		self.shuffle_train   = bool(self.cfg.get('shuffle_train', True))
		
		self.weight_dir      = Path(self.cfg.get('weight_dir', './savegames/'))		
		if self.resume_id and self.resume_step:
			self.weight_init = f'{self.wandb_run.name}/{self.resume_step}'
		else:
			self.weight_init     = self.cfg.get('name_load', None)
		self.dbg_dir         = Path(self.cfg.get('out_dir', './out/'))
		self.result_ext      = '.jpg' 

		self._log.debug(f'  weight_dir   : {self.weight_dir}')
		self._log.debug(f'  name_load    : {self.weight_init}{" (will not load anything)" if self.weight_init is None else ""}')
		self._log.debug(f'  out_dir      : {self.dbg_dir}')

		train_cfg = dict(self.cfg.get('train', {}))
		self.max_epochs      = int(train_cfg.get('max_epochs', -1))
		self.max_iterations  = int(train_cfg.get('max_iterations', -1))		
		self.save_epochs     = int(train_cfg.get('save_epochs', -1))
		self.save_iterations = int(train_cfg.get('save_iterations', 100000))
		self.weight_save     = self.wandb_run.name
		self.no_validation   = bool(train_cfg.get('no_validation', False))
		self.val_interval    = int(train_cfg.get('val_interval', 20000))


		self._log.debug(f'  training config:')
		self._log.debug(f'    max_epochs      : {self.max_epochs}')
		self._log.debug(f'    max_iterations  : {self.max_iterations}')
		self._log.debug(f'    name_save       : {self.weight_save}')		
		self._log.debug(f'    save_epochs     : {self.save_epochs}')
		self._log.debug(f'    save_iterations : {self.save_iterations}')
		self._log.debug(f'    validation      : {"off" if self.no_validation else f"every {self.val_interval}"}')
		pass
		

	@property
	def i(self):
		raise NotImplementedError
		#return self._iterations
	

	def _init_directories(self):
		self.dbg_dir.mkdir(parents=True, exist_ok=True)
		(self.dbg_dir / self.weight_save).mkdir(parents=True, exist_ok=True)
		self.weight_dir.mkdir(parents=True, exist_ok=True)
		pass


	def _init_network(self):
		pass


	def _init_dataset(self):
		pass


	def _init_network_state(self):
		""" Initialize optimizer and scheduler for the network. """		
		pass

	def _train_network(self, batch):
		""" Run forward and backward pass of a network. """
		raise NotImplementedError


	def _should_stop(self, e, i):
		""" Check whether training stop criterion is reached. """

		if self.max_epochs > 0 and e >= self.max_epochs:
			return True

		if self.max_iterations > 0 and i >= self.max_iterations:
			return True

		return False


	def _should_save_epoch(self, e):
		return self.save_epochs > 0 and e % self.save_epochs == 0


	def _should_save_iteration(self, i):
		return self.save_iterations > 0 and i % self.save_iterations == 0


	def _dump(self, img_vars, other_vars={}, force=False):
		if force or ((self.i // 1000) % 5 == 0 and (self.i % 100 == 0)) or (self.i < 20000 and self.i % 100 == 0):
			d1 = {('i_%s' % k):v for k,v in img_vars.items() if v is not None}			
			d2 = {('o_%s' % k):v for k,v in other_vars.items()}
			self.save_dbg({**d1, **d2}, '%d' % self.i)

	def evaluate_test(self, batch, batch_id):
		raise NotImplementedError
		pass


	def evaluate_infer(self, sample):
		raise NotImplementedError
		pass


	def _load_sample(self):
		""" Loads a single example (preferably from self.args.input). """
		raise NotImplementedError
		return batch


	def _save_model(self, *, epoch=None, iterations=None, reason=None):
		raise NotImplementedError
		

	def _load_model(self):
		raise NotImplementedError
	

	def _profiler_schedule(self):
		def schedule(a):
			if a < 2:
				return torch.profiler.ProfilerAction.WARMUP
			elif a < 4:
				return torch.profiler.ProfilerAction.RECORD
			elif a == 4:
				return torch.profiler.ProfilerAction.RECORD_AND_SAFE
			else:
				return torch.profiler.ProfilerAction.NONE

		return schedule


	def dump_val(self, i, batch_id, img_vars):
		d = {('i_%s' % k):v for k,v in img_vars.items()}
		self.save_dbg(d, 'val_%d_%d' % (i,batch_id))
		pass
	

	def save_dbg(self, d, name=None):
		if name is None:
			name = 'dbg_%d' % self._save_id
			self._save_id += 1
			pass
		savemat(self.dbg_dir / self.weight_save / f'{name}.mat', \
			{k:d[k].detach().to('cpu').numpy() for k in d.keys()}, do_compression=True)
		pass


	def save_result(self, d, id):
		name = 'result_%d' % id
		savemat(self.dbg_dir / self.weight_save / f'{name}.mat', \
			{k:v.detach().cpu().numpy() for k,v in d.items()}, do_compression=True)
		pass


	def validate(self):
		counter = 0
		max_counter = 4	
		grids = []

		if len(self.dataset_fake) > 0 or True:
			torch.cuda.empty_cache()

			self.network.eval()

			toggle_grad(self.network.generator, False)
			toggle_grad(self.network.discriminator, False)

			with torch.no_grad():
				for bi, batch_fake in enumerate(self.loader):
					# last item of batch_fake is just index
					
					gen_vars = self._forward_generator_fake(batch_fake.to(self.device))
					if bi < max_counter:
						grid = None
						fake = batch_fake.img.detach()
						rec_fake, fake = gen_vars
						grid = make_grid(fake, rec_fake, nrow=2)
						grids.append(grid)

					del batch_fake
					# self.dump_val(i, bi, gen_vars)
					del gen_vars
					pass
				pass

			grids = make_grid(grids, nrow=1)
			grids = wandb.Image(grids)
			wandb.log({'valid/sim, sim2real': grids}, step=self.i)

			self.network.train()

			toggle_grad(self.network.generator, False)
			toggle_grad(self.network.discriminator, True)

			#del gen_vars
			torch.cuda.empty_cache()
			pass
		else:
			self._log.warning('Validation set is empty - Skipping validation.')
		pass


	def train(self):
		"""Train a network."""

		self.loader = torch.utils.data.DataLoader(self.dataset_train, \
			batch_size=self.batch_size, shuffle=self.shuffle_train, \
			num_workers=self.num_loaders, pin_memory=(not self.unpin), drop_last=True, collate_fn=self.collate_fn_train, worker_init_fn=seed_worker, prefetch_factor=3)

		if self.weight_init is not None:
			self._load_model()
			pass

		self.network.train()

		e = 0

		
		try:
			while not self._should_stop(e, self.i):
				NUM_PROGRESS_GRIDS = 4
				img_progress_grids = [None] * NUM_PROGRESS_GRIDS
				for batch in tqdm(self.loader, total=None, initial=self.i):
					gc.collect()
					torch.cuda.empty_cache()



					if self._should_stop(e, self.i):
						break

					log_scalar, log_img = self._train_network(batch.to(self.device))
					if self._log.isEnabledFor(logging.DEBUG):
						self._log.debug(f'GPU memory allocated: {torch.cuda.memory_allocated(device=self.device)}')
						pass
					

					# Log weigshts and gradient norms
					log = {}


					if len(log) > 0:
						wandb.log(log, step=self.i)
					
					# self._log_sync.update(self.i, log_scalar)

					# Generator Logs
					log = {}
					if self.i % 2 == 0:
						log = {f'train/generator/{log_name_extractor(k)}/' + k:v.to('cpu') for k,v in log_scalar.items()}
					# Discriminator Logs
					elif self.i % 2 == 1:
						log = {f'train/discriminator/{log_name_extractor(k)}/' + k:v.to('cpu') for k,v in log_scalar.items()}


					mod_index = ((self.i // 2) + NUM_PROGRESS_GRIDS - 1) % (self.val_interval // 2)
					if mod_index in range(NUM_PROGRESS_GRIDS) and self.i > 0:
						if len(log_img) == 3: # Take iamges off of cuda
							# should be the same as batch size?
							range_min = min(log_img['fake'].shape[0], log_img['rec_fake'].shape[0], log_img['real'].shape[0])
							grid = None
							for i in range(range_min):
								grid = make_grid([log_img['fake'][i], log_img['rec_fake'][i], log_img['real'][i]], nrow=3).cpu()
								img_progress_grids[mod_index] = grid

						if self.i % self.val_interval == 0 and all([x is not None for x in img_progress_grids]):
							grid = make_grid(img_progress_grids, nrow=1)
							grid = wandb.Image(grid)
							log['train/sim, sim2real, real'] = grid

					wandb.log(log, step=self.i)

					self._dump({**log_img}, force=self._log.isEnabledFor(logging.DEBUG))
					del log_img
					del batch
					

					if self._should_save_iteration(self.i):
						self._save_model(iterations=self.i)
						pass

					if self.i > 0 and self.i % self.val_interval == 0:
						self.validate()
						pass
					pass
					
				e += 1

				if self._should_save_epoch(e):
					self._save_model(epochs=e)
					pass
				pass
			pass
		except:
			if not self.no_safe_exit:
				self._save_model(iterations=self.i, reason='break')
				pass

			self._log.error(f'Unexpected error: {sys.exc_info()[0]}')
			raise
		pass


	def test(self):
		"""Test a network on a dataset."""
		self.loader_fake = torch.utils.data.DataLoader(self.dataset_fake_val, \
			batch_size=1, shuffle=(self.shuffle_test), \
			num_workers=self.num_loaders, pin_memory=True, drop_last=False, collate_fn=self.collate_fn_val, worker_init_fn=seed_worker)

		if self.weight_init is not None:
			self._load_model()
			pass

		self.network.eval()
		with torch.no_grad():
			for bi, batch_fake in enumerate(self.loader_fake):                
				print('batch %d' % bi)
				batch_fake = [f.to(self.device, non_blocking=True) for f in batch_fake[:-1]]
				self.save_result(self.evaluate_test(batch_fake, bi), bi)
				pass
			pass
		pass


	def infer(self):
		"""Run network on single example."""

		if self.weight_init is not None:
			self._load_model()
			pass

		self.network.train()
		# with torch.no_grad():
		self.evaluate_infer(self._load_sample())
			# pass
		pass


	@classmethod
	def add_arguments(cls, parser):
		# methods available at command line 
		
		parser.add_argument('--action', type=str, choices=cls.actions)
		parser.add_argument('--config', type=Path, help='Path to config file.')
		parser.add_argument('-log', '--log', type=str, default='info', choices=_logstr2level.keys())
		parser.add_argument('--log_dir', type=Path, default='./log/', help='Directory for log files.')
		parser.add_argument('--gpu', type=int, default=0, help='ID of GPU. Use -1 to run on CPU. Default: 0')
		parser.add_argument('--no_safe_exit', action='store_true', default=False, help='Do not save model if anything breaks.')
		parser.add_argument('--notes', type=str, default='')
		parser.add_argument('--resume_id', type=str, default=None)
		parser.add_argument('--resume_step', type=int, default=None)
		parser.add_argument('--disabled', default=False, action='store_true')
		parser.add_argument('--session_path ', default='')
		parser.add_argument('--session_dir_name ', default='')
		pass

	
	def run(self):
		self.__getattribute__(self.action)()
		pass
