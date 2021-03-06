import torch
from torchvision import transforms

from models import *


class CustomModelHandler():
	__instance_map = {}
	
	@staticmethod
	def getInstance(model_name):
		if model_name not in CustomModelHandler.__instance_map:
			CustomModelHandler(model_name)
		return CustomModelHandler.__instance_map[model_name]

	def __init__(self, model_name):
		if model_name in CustomModelHandler.__instance_map:
			raise Exception("Singleton!")
		else:
			CustomModelHandler.__instance_map[model_name] = CustomModelHandler.__create_torch_model(model_name)

	@staticmethod
	def __create_torch_model(model_name):
		model_class = globals()[model_name]
		model = model_class()
		model.load_params()
		return model

	@staticmethod
	def apply_transforms(image):
		trans = transforms.Compose([
			transforms.Resize((20,100)),
			transforms.ToTensor()
		])

		return trans(image)
