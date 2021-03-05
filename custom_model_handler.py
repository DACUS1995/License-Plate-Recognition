import torch

from models import *


class CustomModelHandler():
	__instance_map = {}
	
	@staticmethod
	def getInstance(model_name):
		if model_name not in Detector.__instance_map:
			Detector(model_name)
		return Detector.__instance_map[model_name]

	def __init__(self, model_name):
		if model_name in Detector.__instance_map:
			raise Exception("Singleton!")
		else:
			Detector.__instance_map[model_name] = CustomModelHandler.__create_torch_model(model_name)

	@staticmethod
	def __create_torch_model(model_name):
		model = locals()[model_name]
		model.load_params()
		return model
