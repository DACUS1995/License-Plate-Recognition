import cv2
from PIL import Image
import numpy as np


class Detector():
	def __init__(self):
		pass

	
	def detect_single(self, image):
		cv2.imshow("image", image)
		cv2.waitKey(0)


	def preprocess_image(self, image):
		image = image[:, :, ::-1]
		return image


if __name__ == "__main__":
	det = Detector()

	image = Image.open("car.jpg").resize((800,800)).convert("RGB")
	image = np.array(image)
	
	image = det.preprocess_image(image)
	results = det.detect_single(image)