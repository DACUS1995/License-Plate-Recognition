import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

import pathlib
import os

ALPHABET_SIZE = 63

class TranscribeModel(nn.Module):
	def __init__(self):
		super(TranscribeModel, self).__init__()
		self.conv_block1 = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
			nn.MaxPool2d(kernel_size=(2, 2)),
			nn.BatchNorm2d(16),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
			nn.MaxPool2d(kernel_size=(2, 2)),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2, inplace=True)
		)
		
		self.linear_block1 = nn.Sequential(
			nn.Linear(8000, 512),
			nn.Dropout(p=0.5),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 512),
			nn.Dropout(p=0.5),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 630),
		)

		self.softmax = nn.LogSoftmax(dim=2)

	def forward(self, input):
		out = self.conv_block1(input)
		out = self.linear_block1(out.view(out.shape[0], -1))
		out = out.view(out.shape[0], 10, ALPHABET_SIZE)
		out = self.softmax(out)
		
		return out

	def load_params(self):
		print(pathlib.Path(__file__).parent.absolute())
		checkpoint = torch.load(
			os.path.join(pathlib.Path(__file__).parent.absolute(), "./ckp.pt")
		)
		self.load_state_dict(checkpoint["model_state_dict"])