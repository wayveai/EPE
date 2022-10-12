import torch
import torch.nn as nn

EPSILON=1e-08

class HingeLoss(nn.Module):
	def __init__(self):
		super(HingeLoss, self).__init__()
		pass

	def forward_gen(self, input):
		# should be 1 or higher
		return (1-input).clamp(min=0)

	def forward_real(self, input):
		# should be 1 higher
		return (1-input).clamp(min=0)

	def forward_fake(self, input):
		# should be 0 or lower
		return input.clamp(min=0)

@torch.jit.script
def _fw_ls_real(input):
	EPSILON = 1e-8
	input[input.isnan()] = EPSILON
	return (1-input).pow(2)

class LSLoss(nn.Module):
	def __init__(self):
		super(LSLoss, self).__init__()
		pass

	def forward_gen(self, input):
		# should be 1
		# return (1-input).pow(2)
		return _fw_ls_real(input)

	def forward_real(self, input):
		# should be 1
		# return (1-input).pow(2)
		return _fw_ls_real(input)

	def forward_fake(self, input):
		input[input.isnan()] = EPSILON
		return (input).pow(2)


class NSLoss(nn.Module):
	def __init__(self):
		super(NSLoss, self).__init__()
		pass

	def forward_gen(self, input):
		# should be 1
		return torch.nn.functional.softplus(1-input)

	def forward_real(self, input):
		# should be 1
		return torch.nn.functional.softplus(1-input)

	def forward_fake(self, input):
		# should be 0
		return torch.nn.functional.softplus(input)
