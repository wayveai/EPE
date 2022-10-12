import torch
import torch.nn as nn

@torch.jit.script
def make_residual(img, x):
	epsilon=1e-08
	return torch.sigmoid(-torch.log((1 / (img.clamp(min=0.001, max=0.999) + epsilon)) - 1) + x)


class ResidualGenerator(nn.Module):
	def __init__(self, network):
		super(ResidualGenerator, self).__init__()
		self.network = network
		pass

	def forward(self, epe_batch):
		net_out = self.network(epe_batch)
		x = make_residual(epe_batch.img, net_out)
		return x

