import cv2
from PIL import Image
import numpy as np
import pytesseract

class Detector():
	def __init__(self):
		pass

	def detect_single(self, image):
		#TODO Add tests
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		x1, y1, x2, y2 = self.localize_plate(gray_image)
		license_plate = gray_image[y1:y2+1, x1:x2+1]
		license_plate_number = self.extract_characters(license_plate)
		return license_plate_number


	def localize_plate(self, gray_image):
		gray_image = cv2.bilateralFilter(gray_image, 13, 15, 15)

		# rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 5))
		# blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, rectKern)

		canny = cv2.Canny(gray_image, 170, 255, 1)
		contours = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
		contours = sorted(contours, key=cv2.contourArea, reverse = True)[:10]
		license_plate_contour = None

		for c in contours:
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.018 * peri, True)
			
			if len(approx) == 4:
				license_plate_contour = approx
				# cv2.drawContours(gray_image, [license_plate_contour], -1, (0, 0, 255), 3)
				break

		license_plate_contour = np.array(license_plate_contour).squeeze()
		x1 = min(license_plate_contour[:, 0])
		x2 = max(license_plate_contour[:, 0])
		y1 = min(license_plate_contour[:, 1])
		y2 = max(license_plate_contour[:, 1])

		return (x1, y1, x2, y2)

	def extract_characters(self, plate_image):
		#TODO separate the chars using the image histogram for x and y axis and then mnist for each of them
		pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract'
		text = pytesseract.image_to_string(plate_image, config='--psm 11')
		return text

	def preprocess_image(self, image):
		image = image[:, :, ::-1]
		return image


if __name__ == "__main__":
	det = Detector()

	image = Image.open("car.jpg").resize((800,800)).convert("RGB")
	image = np.array(image)
	
	image = det.preprocess_image(image)
	results = det.detect_single(image)
	print(results)