import pathlib
import os
from PIL import Image
import string
from typing import Tuple
import datetime
import copy
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import datasets, models, transforms as T
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from train import ALPHABET, MAX_WORD_LENGTH

class CustomDataset(Dataset):
	def __init__(self, root_path, type="train"):
		self.root_path = root_path
		self.type = type
		self.images_paths = list(pathlib.Path(self.root_path + "./images").glob('*.png'))
		self.transforms = {
			'train' : T.Compose([
				T.RandomRotation(20),
				T.GaussianBlur(3),
				T.ToTensor()
			]),
			'valid' : T.Compose([
				T.ToTensor()
			])
		}
		global ALPHABET
		self.alphabet = ALPHABET
		self.alphabet_size = len(self.alphabet)

	def __getitem__(self, idx):
		image_path = self.images_paths[idx]
		sample_name = str(image_path).split(os.sep)[-1].split(".")[0]
		text_path = self.root_path + "/transcripts/" + str(int(sample_name) + 1) + ".txt"

		image = Image.open(image_path).convert("RGB")
		with open(text_path) as f:
			text = f.read()

		image = self.transforms[self.type](image)
		text_tensor = self.wordToTensor(text)
		return image, (text_tensor, len(text), text)
		

	def __len__(self):
		return len(self.images_paths)

	def letterToIndex(self, letter):
		return self.alphabet.find(letter)

	def letterToTensor(self, letter):
		tensor = torch.zeros(1, n_letters)
		tensor[0][letterToIndex(letter)] = 1
		return tensor

	def wordToTensor(self, word):
		tensor = torch.full((MAX_WORD_LENGTH,), len(ALPHABET) - 1)
		for li, letter in enumerate(word):
			tensor[li] = self.letterToIndex(letter)
		return tensor


class DataHandler:
	def __init__(self, run_config):
		self._training_dataset = None
		self._validation_dataset = None
		self._run_config = run_config

		self._load_datasets()
		
	def _load_datasets(self):
		self._training_dataset = CustomDataset("dataset/training")
		self._validation_dataset = CustomDataset("dataset/validation")

	def get_data_loaders(self) -> Tuple[DataLoader]:
		return (
			DataLoader(self._training_dataset, batch_size=self._run_config["batch_size"], shuffle=True, pin_memory=True, drop_last=True), 
			DataLoader(self._validation_dataset, batch_size=self._run_config["batch_size"], shuffle=False, pin_memory=True, drop_last=True)
		)

	def get_datasets(self) -> Tuple[Dataset]:
		return self._training_dataset, self._validation_dataset

	def get_datasets_sizes(self) -> Tuple[int]:
		return len(self._training_dataset), len(self._validation_dataset)