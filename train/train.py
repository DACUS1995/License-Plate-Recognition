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

from tqdm import tqdm
import pathlib
import os
from PIL import Image
import string
from typing import Tuple
import datetime
import copy
import time
import numpy as np


BATCH_SIZE = 32
WORKERS = 1
EPOCHS = 100
MAX_WORD_LENGTH = 10
ALPHABET = string.ascii_letters + string.digits + "_" #blank char for CTC
OUTPUT_SEQUENCE_LENGTH = 10
OUTPUT_STEP_SIZE = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running training on [{device}]")


log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)

# import after constants due to dependencies
from utils import *
from models import TranscribeModel
from data import DataHandler


def train_step(model, data, optimizer, loss_criterion):
	x_batch, (label_batch, label_length, label_text) = data
	x_batch, label_batch, label_length = x_batch.to(device), label_batch.to(device), label_length.to(device)
	optimizer.zero_grad()
	outputs = model(x_batch)

	outputs_permuted = outputs.permute((0, 2, 1))
	loss = loss_criterion(outputs_permuted, label_batch)
	loss.backward()
	optimizer.step()

	predictions = tensorToWordSync(outputs.detach())
	running_distance = batchLevenshteinDistance(predictions, label_text)
	return loss.item(), running_distance


@torch.no_grad()
def validation_step(model, data, loss_criterion):
	x_batch, (label_batch, label_length, label_text) = data
	x_batch, label_batch, label_length = x_batch.to(device), label_batch.to(device), label_length.to(device)
	outputs = model(x_batch)

	outputs_permuted = outputs.permute((0, 2, 1))
	loss = loss_criterion(outputs_permuted, label_batch)

	predictions = tensorToWordSync(outputs.detach())
	running_distance = batchLevenshteinDistance(predictions, label_text)
	return loss.item(), running_distance


def log_epoch(epoch_loss, epoch_distance, epoch, log_type="Training"):
	writer.add_scalar(f"Loss/{log_type}", epoch_loss, epoch)
	writer.add_scalar(f"Distance/{log_type}", epoch_distance, epoch)

	print(log_type + ' step => Loss: {:.4f} | Dist: {:.4f}'.format(
		epoch_loss, epoch_distance
	))

def save_checkpoint(model, optimizer, epoch):
	torch.save({
		"epoch": epoch,
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": optimizer.state_dict()
	}, "./checkpoints/ckp.pt")


def train_model():
	data_handler = DataHandler(run_config = {
		"batch_size": BATCH_SIZE,
		"workers": WORKERS
	})

	train_loader, validation_loader = data_handler.get_data_loaders()
	training_dataset_size, validation_dataset_size = data_handler.get_datasets_sizes()

	model = TranscribeModel()
	model.to(device)
	optimizer = optim.AdamW(
		model.parameters(), 
		lr=0.0001, 
		betas=(0.9, 0.999), 
		eps=1e-08, 
		weight_decay=1e-4
	)
	loss_criterion = nn.NLLLoss()
	scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=10, epochs=EPOCHS, anneal_strategy='linear')
	best_distance = float("inf")
	best_model_params = None

	since = time.time()
	for epoch in range(EPOCHS):
		print('Epoch {}/{}'.format(epoch, EPOCHS))
		print('-' * 10)

		model = model.train()
		training_loss = []
		running_loss = 0.0
		running_distance = 0

		loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}] progress")
		for i, data in enumerate(loop):
			loss, distance = train_step(model, data, optimizer, loss_criterion)
			
			running_loss += loss * BATCH_SIZE
			running_distance += distance
			training_loss.append(loss)
			loop.set_postfix(loss=loss)

		epoch_loss = running_loss / training_dataset_size
		epoch_distance = running_distance / training_dataset_size
		log_epoch(epoch_loss, epoch_distance, epoch, log_type="Training")

		scheduler.step()

		model = model.eval()
		validation_loss = []
		running_loss = 0.0
		running_distance = 0

		for i, data in enumerate(validation_loader):
			loss, distance = validation_step(model, data, loss_criterion)

			running_loss += loss * BATCH_SIZE
			running_distance += distance
			validation_loss.append(loss)

		epoch_loss = running_loss / validation_dataset_size
		epoch_distance = running_distance / validation_dataset_size
		log_epoch(epoch_loss, epoch_distance, epoch, log_type="Validation")

		if best_distance > epoch_distance:
			best_distance = epoch_distance
			save_checkpoint(model, optimizer, epoch)
			best_model_params = copy.deepcopy(model.state_dict())


	time_elapsed = time.time() - since
	print('-' * 10)
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60
	))
	print('Best validation Distance: {:4f}'.format(best_distance))


def main():
	train_model()


if __name__ == "__main__":
	main()