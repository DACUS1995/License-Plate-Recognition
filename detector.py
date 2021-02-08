import cv2
from PIL import Image
import numpy as np


class Detector():
	def __init__(self):
		pass

	def detect_single(self, image):
		coordinates = self.localize_plate(image)


	def localize_plate(self, image):
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray_image = cv2.bilateralFilter(gray_image, 13, 15, 15)

		# rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 5))
		# blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, rectKern)

		canny = cv2.Canny(gray_image, 170, 255, 1)
		contours = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
		contours = sorted(contours, key=cv2.contourArea, reverse = True)[:10]
		screenCnt = None

		for c in contours:
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.018 * peri, True)
			
			if len(approx) == 4:
				screenCnt = approx
				cv2.drawContours(gray_image, [screenCnt], -1, (0, 0, 255), 3)
				break


		cv2.imshow("image", gray_image)
		cv2.waitKey(0)

	def extract_characters(self, plate_image):
		pass

	def preprocess_image(self, image):
		image = image[:, :, ::-1]
		return image


if __name__ == "__main__":
	det = Detector()

	image = Image.open("car.jpg").resize((800,800)).convert("RGB")
	image = np.array(image)
	
	image = det.preprocess_image(image)
	results = det.detect_single(image)